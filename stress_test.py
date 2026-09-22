#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字员工平台 · 并发压测

为什么要分四个阶段测：
  阶段 A 检索+权限+重排 —— 服务端的核心路径（重排是全局串行的，这里是真正的瓶颈）
  阶段 B HTTP 轻接口   —— 前端每几秒就在轮询（登录、索引状态、历史），178 人开着页面就是背景负载
  阶段 C 端到端问答     —— 走大模型，看真实用户体验（会消耗 token，级别要小）
  阶段 D 运行时/内存    —— 会话缓存和内存会不会随问答无限增长

用法（在项目根目录执行）：
  lora-from-scratch/.venv/bin/python stress_test.py --phase all
  lora-from-scratch/.venv/bin/python stress_test.py --phase retrieval --levels 1,10,20,50
  lora-from-scratch/.venv/bin/python stress_test.py --phase http --levels 20,50
  lora-from-scratch/.venv/bin/python stress_test.py --phase chat --levels 3,5,10
  lora-from-scratch/.venv/bin/python stress_test.py --phase runtime --admin testadmin:密码

注意：阶段 C 会真的调用大模型（花钱），默认级别小；阶段 A 需要服务在跑（它要读语料和索引）。
结果同时打印到屏幕并存成 并发压测报告_<时间>.md
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:8644"

# 会出现在真实提问里的几类问题：能答的、问权限敏感的、问了也没依据的
QUESTIONS = [
    "公司安全生产方针是什么？",
    "某低压工程的结算金额是多少？",
    "公司《员工手册》第七章薪酬管理里，绩效工资是怎么规定的？",
    "员工请假需要走什么流程？",
    "某项目的施工组织方案里管材规格是什么？",
    "公司差旅费报销标准是多少？",
    "天然气管道巡检的周期是多久？",
    "2023年度工资、考勤表里的数据怎么填？",
]

# 用真实工号（不同岗位的可见范围不同，这样并发时会走到不同的权限分支）
EMP_IDS = ["ENG002", "HR001", "FIN001", "QA002", "MKT002", "GM002", "RD004", "SCM001", "TST902", "TST903"]


# ─────────────────────────── 小工具 ───────────────────────────
def pct(xs, p):
    """百分位（毫秒）：p=95 表示 95% 的请求都比它快。"""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((p / 100.0) * len(xs) + 0.5)) - 1)
    return xs[max(0, k)]


def summarise(name, lat, errs, wall_s, extra=""):
    """打印一行结论，并返回 markdown 行。"""
    ok = len(lat)
    line = ("| %-22s | %5d | %5d | %8.0f | %8.0f | %8.0f | %7.1f | %s |"
            % (name, ok, len(errs), pct(lat, 50), pct(lat, 95), pct(lat, 99),
               (ok / wall_s) if wall_s else 0, extra or "—"))
    return line


def http_json(path, data=None, token=None, timeout=60):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE_URL + path,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def login(emp_id, password):
    return http_json("/api/login", {"emp_id": emp_id, "password": password})["token"]


def ask_stream(token, question, timeout=300):
    """走真实的问答接口（流式），返回 (答案, 是否有工具调用, 是否流里报错)。"""
    req = urllib.request.Request(
        BASE_URL + "/api/chat",
        data=json.dumps({"message": question, "role": "lucky"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    txt, tool, err, busy = "", None, None, False
    ttfb = None
    _t_start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if ttfb is None:
                ttfb = (time.time() - _t_start) * 1000      # 首字节：服务器多久才开始回话
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            if d.get("tool"):
                tool = d["tool"]
            if d.get("error"):
                err = d["error"]
            if d.get("busy"):
                busy = True
            txt += d.get("delta", "")
    return txt, tool, err, busy, (ttfb or 0)


def runtime_snapshot(admin_token):
    try:
        return http_json("/api/admin/runtime", token=admin_token, timeout=20)
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────── 阶段 A：检索+权限+重排 ───────────────────────
def phase_retrieval(levels, rounds, report):
    """在进程内直接调用生产代码的检索函数（不经过 HTTP，也不调大模型）。

    这样测的是最容易被并发压垮的部分：BM25 召回 → 权限过滤 → 重排（全局串行）。
    """
    print("\n【阶段 A】检索 + 权限 + 重排（不调大模型，进程内调用生产代码）")
    sys.path.insert(0, str(BASE_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location("kbapp", BASE_DIR / "webui" / "app.py")
    kbapp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kbapp)          # 会加载语料/索引（第一次要等几十秒）
    print("   生产模块加载完成，语料块 %d" % len(kbapp.chunks))

    rows = []
    for L in levels:
        lat, errs = [], []
        lock = threading.Lock()
        tasks = [(EMP_IDS[i % len(EMP_IDS)], QUESTIONS[i % len(QUESTIONS)])
                 for i in range(L * rounds)]      # 注意：取工号本身，别把下标传进去

        def work(t):
            eid, q = t
            t0 = time.time()
            try:
                hits = kbapp.retrieve_for_user(q, eid, k=5, info={})
                if hits is None:
                    raise RuntimeError("返回 None")
                with lock:
                    lat.append((time.time() - t0) * 1000)
            except Exception as e:
                with lock:
                    errs.append("%s: %s" % (type(e).__name__, e))

        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=L) as ex:
            list(ex.map(work, tasks))
        wall = time.time() - t0
        row = summarise("A 并发 %d（%d 次）" % (L, len(tasks)), lat, errs, wall)
        print("   " + row)
        if errs:
            print("      错误样例：%s" % errs[:2])
        rows.append(row)
    report.append("### 阶段 A：检索 + 权限 + 重排（无大模型）\n")
    report += rows
    report.append("")
    return rows


# ───────────────────────── 阶段 B：HTTP 轻接口 ─────────────────────────
def phase_http(levels, rounds, report, creds):
    print("\n【阶段 B】HTTP 轻接口（登录 / 索引状态 / 历史消息）")
    rows = []
    tokens = []
    for eid, pwd in creds:
        try:
            tokens.append(login(eid, pwd))
        except Exception as e:
            print("   登录 %s 失败：%s" % (eid, e))
    if not tokens:
        print("   跳过（没有可用账号）")
        return rows

    paths = [("/api/index_status", None), ("/api/history", None)]     # GET
    for L in levels:
        lat, errs = [], []
        lock = threading.Lock()

        def work(i):
            path, body = paths[i % len(paths)]
            t0 = time.time()
            try:
                if path == "/api/login":
                    http_json(path, {"emp_id": creds[0][0], "password": creds[0][1]}, timeout=30)
                else:
                    http_json(path, token=tokens[i % len(tokens)], timeout=30)
                with lock:
                    lat.append((time.time() - t0) * 1000)
            except Exception as e:
                with lock:
                    errs.append("%s %s" % (path, e))

        n = L * rounds
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=L) as ex:
            list(ex.map(work, range(n)))
        wall = time.time() - t0
        row = summarise("B 并发 %d（%d 次）" % (L, n), lat, errs, wall)
        print("   " + row)
        if errs:
            print("      错误样例：%s" % errs[:2])
        rows.append(row)
    report.append("### 阶段 B：HTTP 轻接口（前端会持续轮询的那些）\n")
    report += rows
    report.append("")
    return rows


# ─────────────────────── 阶段 C：端到端问答（走模型）───────────────────────
def phase_chat(levels, rounds, report, creds):
    print("\n【阶段 C】端到端问答（走大模型，会消耗 token）")
    rows = []
    tokens = []
    for eid, pwd in creds:
        try:
            tokens.append(login(eid, pwd))
        except Exception as e:
            print("   登录 %s 失败：%s" % (eid, e))
    if not tokens:
        return rows

    for L in levels:
        lat, errs, empty, tools = [], [], 0, 0
        lock = threading.Lock()

        def work(i):
            nonlocal empty, tools
            q = QUESTIONS[i % len(QUESTIONS)]
            tk = tokens[i % len(tokens)]
            t0 = time.time()
            try:
                txt, tool, err, is_busy, ttfb = ask_stream(tk, q)
                dt = (time.time() - t0) * 1000
                with lock:
                    if err:
                        errs.append("流错误: %s" % err)
                    elif not txt.strip():
                        empty += 1
                    else:
                        lat.append(dt)
                        if tool:
                            tools += 1
            except Exception as e:
                with lock:
                    errs.append("%s: %s" % (type(e).__name__, e))

        n = L * rounds
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=L) as ex:
            list(ex.map(work, range(n)))
        wall = time.time() - t0
        extra = "空答 %d ｜ 带工具 %d" % (empty, tools)
        row = summarise("C 并发 %d（%d 问）" % (L, n), lat, errs, wall, extra)
        print("   " + row)
        if errs:
            print("      错误样例：%s" % errs[:3])
        rows.append(row)
    report.append("### 阶段 C：端到端问答（含大模型）\n")
    report += rows
    report.append("")
    return rows


# ─────────────────────── 阶段 D：运行时与内存 ───────────────────────
def phase_runtime(report, admin_token, creds, rounds=20):
    print("\n【阶段 D】运行时与内存（会话缓存会不会无限涨）")
    rows = []
    before = runtime_snapshot(admin_token)
    print("   压测前：%s" % json.dumps(before, ensure_ascii=False))
    tokens = []
    for eid, pwd in creds[:2]:
        try:
            tokens.append(login(eid, pwd))
        except Exception:
            pass
    if tokens:
        # 连续追问：同一工号多轮，观察会话缓存和内存
        for i in range(rounds):
            try:
                ask_stream(tokens[i % len(tokens)], "公司安全生产方针是什么？" if i % 2 else "员工请假需要走什么流程？")
            except Exception as e:
                print("   第 %d 轮失败：%s" % (i, e))
    after = runtime_snapshot(admin_token)
    print("   压测后：%s" % json.dumps(after, ensure_ascii=False))
    if "rss_mb" in before and "rss_mb" in after:
        print("   内存：%.1f MB → %.1f MB（涨 %.1f MB / %d 轮）"
              % (before["rss_mb"], after["rss_mb"], after["rss_mb"] - before["rss_mb"], rounds))
        print("   会话缓存：%s → %s 条" % (before.get("conversations"), after.get("conversations")))
    report.append("### 阶段 D：运行时与内存\n")
    report.append("| 时点 | 会话缓存条数 | 内存 RSS(MB) | 线程数 | 在线登录态 |")
    report.append("|---|---|---|---|---|")
    for tag, snap in (("压测前", before), ("压测后", after)):
        report.append("| %s | %s | %s | %s | %s |" % (tag, snap.get("conversations"), snap.get("rss_mb"),
                                                     snap.get("threads"), snap.get("sessions")))
    report.append("")
    return rows



# ─────────────── 阶段 E：瞬时并发（"100 人同时提问会不会崩"）───────────────
def phase_burst(n, report, creds):
    """n 个请求在同一瞬间打进来：统计答完的、被闸门挡下的、报错的，以及服务端峰值与内存。

    闸门（CHAT_MAX_INFLIGHT）决定行为：
      设成 8   → 8 个真答，其余立刻收到"人有点多"的明确提示（等待时间可控）
      设成 >=n → 全部真答，等于没有闸门，可以看排队与内存的真实表现
    """
    print("\n【阶段 E】瞬时并发 %d 个问答" % n)
    tokens = []
    for eid, pwd in creds:
        try:
            tokens.append(login(eid, pwd))
        except Exception as e:
            print("   登录 %s 失败：%s" % (eid, e))
    if not tokens:
        return []
    admin_tk = None
    if len(creds) > 2:                      # 第 3 个账号约定为管理员，用来读运行时
        try:
            admin_tk = login(creds[2][0], creds[2][1])
        except Exception:
            pass
    snap_before = runtime_snapshot(admin_tk) if admin_tk else {}

    answered, busy, errs, lat, ttfb_ans, ttfb_busy = [], [], [], [], [], []
    send_ts, send_lock = [], threading.Lock()
    t_burst = time.time()
    barrier = threading.Barrier(n)          # 让 n 个线程尽量同一瞬间发出请求
    lock = threading.Lock()

    def work(i):
        tk = tokens[i % len(tokens)]
        q = QUESTIONS[i % len(QUESTIONS)]
        try:
            barrier.wait(timeout=60)
        except Exception:
            pass
        t0 = time.time()
        try:
            txt, tool, err, is_busy, ttfb = ask_stream(tk, q, timeout=300)
            dt = (time.time() - t0) * 1000
            with lock:
                if err:
                    errs.append("流错误 %s" % err)
                elif is_busy:
                    busy.append(dt); ttfb_busy.append(ttfb)
                elif txt.strip():
                    answered.append(dt); ttfb_ans.append(ttfb)
                else:
                    errs.append("空答")
        except Exception as e:
            with lock:
                errs.append("%s: %s" % (type(e).__name__, e))

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(work, range(n)))
    wall = time.time() - t0
    snap_after = runtime_snapshot(admin_tk) if admin_tk else {}

    print("   总耗时 %.1f 秒 ｜ 答完 %d ｜ 被闸门挡下 %d ｜ 出错 %d" % (wall, len(answered), len(busy), len(errs)))
    if send_ts:
        print("   请求实际发出的时间分布：中位 %.2f 秒 ｜ 最晚 %.2f 秒（越集中越接近同时）"
              % (sorted(send_ts)[len(send_ts)//2], max(send_ts)))
    if answered:
        print("   答完的延迟：P50 %.0f ms ｜ P95 %.0f ms ｜ 最慢 %.0f ms" % (pct(answered, 50), pct(answered, 95), max(answered)))
    if busy:
        print("   挡下的总延迟：P50 %.0f ms ｜ 其中首字节 P50 %.0f ms（首字节=用户多久看到回话）"
              % (pct(busy, 50), pct(ttfb_busy, 50)))
    if answered:
        print("   答完的首字节：P50 %.0f ms ｜ P95 %.0f ms" % (pct(ttfb_ans, 50), pct(ttfb_ans, 95)))
    if errs:
        print("   错误样例：%s" % errs[:3])
    if snap_before:
        print("   服务端：闸门上限 %s ｜ 峰值并发 %s ｜ 共挡下 %s ｜ 大模型失败 %s"
              % (snap_before.get("chat_max_inflight"), snap_after.get("chat_peak"),
                 snap_after.get("chat_shed"), snap_after.get("chat_llm_failed")))
        print("   内存：%.1f MB → %.1f MB ｜ 线程：%s → %s ｜ 会话缓存：%s → %s 条"
              % (snap_before.get("rss_mb", 0), snap_after.get("rss_mb", 0), snap_before.get("threads"),
                 snap_after.get("threads"), snap_before.get("conversations"), snap_after.get("conversations")))
    report.append("### 阶段 E：瞬时并发 %d 个问答\n" % n)
    report.append("| 指标 | 值 |")
    report.append("|---|---|")
    report.append("| 总耗时(秒) | %.1f |" % wall)
    report.append("| 答完 | %d |" % len(answered))
    report.append("| 被闸门挡下（立刻明确提示） | %d |" % len(busy))
    report.append("| 出错 | %d |" % len(errs))
    if answered:
        report.append("| 答完延迟 P50/P95/最慢 (ms) | %.0f / %.0f / %.0f |" % (pct(answered, 50), pct(answered, 95), max(answered)))
    if snap_after:
        report.append("| 服务端峰值并发 / 累计挡下 / 大模型失败 | %s / %s / %s |" % (snap_after.get("chat_peak"),
                                                                        snap_after.get("chat_shed"),
                                                                        snap_after.get("chat_llm_failed")))
        report.append("| 内存 RSS (MB) 前后 | %.1f → %.1f |" % (snap_before.get("rss_mb", 0), snap_after.get("rss_mb", 0)))
    report.append("")
    return [answered, busy, errs]


def main():
    global BASE_URL          # 必须在用到 BASE_URL 之前声明
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "retrieval", "http", "chat", "runtime", "burst"])
    ap.add_argument("--n", type=int, default=100, help="阶段 E 的瞬时并发数")
    ap.add_argument("--levels", default="1,10,20,50", help="并发级别，逗号分隔")
    ap.add_argument("--rounds", type=int, default=5, help="每个并发级别每人问几轮")
    ap.add_argument("--base", default=BASE_URL)
    ap.add_argument("--admin", default="", help="管理员账号:密码（用于读运行时指标）")
    ap.add_argument("--creds", default="TST902:Tst902pass,TST903:Tst903pass",
                    help="普通账号:密码，逗号分隔（阶段 B/C/D 用）")
    a = ap.parse_args()

    BASE_URL = a.base
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    creds = [tuple(x.split(":", 1)) for x in a.creds.split(",") if ":" in x]

    admin_token = None
    if a.admin and ":" in a.admin:
        eid, pwd = a.admin.split(":", 1)
        try:
            admin_token = login(eid, pwd)
        except Exception as e:
            print("管理员登录失败：%s" % e)

    report = ["# 数字员工平台 · 并发压测报告", "",
              "时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
              "地址：%s ｜ 并发级别：%s ｜ 每级轮数：%d" % (BASE_URL, levels, a.rounds), "",
              "| 场景 | 成功 | 失败 | P50(ms) | P95(ms) | P99(ms) | 吞吐(次/秒) | 备注 |",
              "|---|---|---|---|---|---|---|---|"]

    if a.phase in ("all", "retrieval"):
        phase_retrieval(levels, a.rounds, report)
    if a.phase in ("all", "http"):
        phase_http(levels, a.rounds, report, creds)
    if a.phase in ("all", "chat"):
        phase_chat(levels, min(a.rounds, 2), report, creds)      # 想测更高并发就 --levels 20,30
    if a.phase == "burst":
        phase_burst(a.n, report, creds + ([(a.admin.split(":", 1)[0], a.admin.split(":", 1)[1])] if a.admin and ":" in a.admin else []))
    if a.phase in ("all", "runtime") and admin_token:
        phase_runtime(report, admin_token, creds)

    out = BASE_DIR / ("并发压测报告_%s.md" % time.strftime("%Y%m%d_%H%M%S"))
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n报告已写入：%s" % out)


if __name__ == "__main__":
    main()
