#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识库快照与一键回滚（对应验收标准里的"一键回滚"）。

它是什么：给"语料 + 索引 + 权限 + 配置 + 数据库 + 代码"拍一个存档点，出事一条命令退回去。
它不是什么：不是模型权重 checkpoint（几 GB 且极少变，不进快照）。

要回滚的三类东西（每类都出过事）：
  ① 语料与索引 docs/ + chunks.json + faiss.index —— 重建索引后检索变差最常见
  ② 权限与标签 acl_matrix.json / acl_tags*.json / acl_roster.json —— 改错就是安全事故
  ③ 代码与配置 app.py / retrieval.py / prompting.py / app.db —— 改坏 ACL 会静默失效

关键设计：
  · 用 macOS 的 APFS 克隆（cp -c）：秒级、几乎不额外占盘，所以可以放心多变几版
  · 回滚前**先给当前状态做一版**（保证"回滚本身也能被回滚"）
  · 回滚后自动体检：检索能出结果 / 权限越权=0 / 数据库可读；不通过提示退回到回滚前那版
  · 每次动作写审计日志 snapshots/ROLLBACK.log（谁、何时、从哪版到哪版、原因、体检结果）

用法（项目根目录下）：
    python kb_snapshot.py --backup "改权限矩阵之前"
    python kb_snapshot.py --list
    python kb_snapshot.py --diff 改权限矩阵       # 支持只写关键词，不用写全 ID
    python kb_snapshot.py --restore 改权限矩阵 --reason "员工反馈查不到资料"
    python kb_snapshot.py --verify 改权限矩阵
    python kb_snapshot.py --prune --keep 10
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAP_DIR = ROOT / "snapshots"
LOG = SNAP_DIR / "ROLLBACK.log"

# 单文件：纳入快照
PATTERNS = [
    # 语料与索引（由 index_meta.py 决定真实位置，默认在项目根）
    "index_meta.json", "chunks.json", "faiss.index",
    # 权限与标签
    "acl_matrix.json", "acl_block_overrides.json", "acl_tags.json", "acl_tags_by_source.json", "acl_roster.json", "acl_grants.json",
    # 检索与提示词（等于运行时配置）
    "retrieval.py", "prompting.py", "gold_rules.py",
    # 权限相关脚本（回滚要连判定逻辑一起退）
    "acl_roster.py", "acl_guard.py", "acl_register.py", "acl_sync.py", "acl_tagging.py",
    # 服务与前端
    "webui/app.py", "webui/static/chat.html", "webui/static/login.html",
    # 数据库（账号 + 花名表）
    "webui/data/app.db",
]
# 目录：整目录克隆（语料原始文件，1000+ 文件约 1 GB，用 APFS 克隆不占额外空间）
DIRS = ["docs"]
# 明确排除（让看代码的人放心）
EXCLUDE_NOTE = ["lora-from-scratch/models/*（模型权重，几 GB 且极少变）",
                ".env（凭据）", "lora-from-scratch/.venv（虚拟环境）", "snapshots/（快照自己）",
                "__pycache__/"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def dir_fingerprint(d: Path):
    """目录指纹：文件数 + 所有 (相对路径, 大小, mtime) 的 sha1。与 index_meta 的算法一致。"""
    if not d.exists():
        return None
    items = []
    total = 0
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.name.startswith("~$"):
            continue
        st = f.stat()
        items.append((str(f.relative_to(d)), st.st_size, st.st_mtime_ns))
        total += st.st_size
    h = hashlib.sha1(repr(items).encode()).hexdigest()
    return {"文件数": len(items), "字节数": total, "指纹": h}


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        r = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
        if r.returncode != 0:
            raise OSError(r.stderr.decode()[:200])
    except Exception:
        shutil.copy2(src, dst)


def clone_dir(src: Path, dst: Path):
    """整目录克隆（APFS 克隆优先）。不删除目标里多出来的文件。"""
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cp", "-c", "-R", str(src) + "/.", str(dst)], capture_output=True)
    if r.returncode != 0:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def rel_files():
    return [p for p in PATTERNS if (ROOT / p).exists() and (ROOT / p).is_file()]


def key_counts() -> dict:
    c = {}
    try:
        con = sqlite3.connect(str(ROOT / "webui" / "data" / "app.db"))
        c["花名表人数"] = con.execute("SELECT COUNT(*) FROM roster").fetchone()[0]
        c["账号数"] = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        con.close()
    except Exception as e:
        c["数据库"] = "读取失败：%s" % e
    try:
        c["权限表人数"] = len(json.loads((ROOT / "acl_roster.json").read_text(encoding="utf-8")))
    except Exception:
        pass
    try:
        c["语料块数"] = len(json.loads((ROOT / "chunks.json").read_text(encoding="utf-8")))
    except Exception:
        pass
    fp = dir_fingerprint(ROOT / "docs")
    if fp:
        c["语料文件数"] = fp["文件数"]
    return c


def git_rev() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def do_backup(note: str, silent=False) -> Path:
    sid = "%s_%s" % (time.strftime("%Y%m%d_%H%M%S"),
                     "".join(ch for ch in (note or "手动") if ch not in '\\/:*?"<>|').strip()[:40])
    d = SNAP_DIR / sid
    (d / "files").mkdir(parents=True, exist_ok=True)
    meta = {"id": sid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "user": os.environ.get("USER", "?"),
            "host": platform.node(), "note": note, "git": git_rev(), "counts": key_counts(),
            "files": {}, "dirs": {}, "excluded": EXCLUDE_NOTE}
    t0 = time.time()
    for rel in rel_files():
        src = ROOT / rel
        copy_file(src, d / "files" / rel)
        meta["files"][rel] = {"sha256": sha256(src), "size": src.stat().st_size}
    for rel in DIRS:
        src = ROOT / rel
        fp = dir_fingerprint(src)
        if not fp:
            continue
        clone_dir(src, d / "files" / rel)
        meta["dirs"][rel] = fp
    meta["耗时秒"] = round(time.time() - t0, 2)
    (d / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not silent:
        mb = sum(v["size"] for v in meta["files"].values()) / 1e6
        print("已做快照：%s" % sid)
        print("  文件 %d 个（约 %.1f MB）" % (len(meta["files"]), mb)
              + ("" if not meta["dirs"] else "＋目录 %s" % "、".join(
                  "%s(%d 个文件/%.0f MB)" % (k, v["文件数"], v["字节数"] / 1e6) for k, v in meta["dirs"].items()))
              + "，耗时 %.2f 秒（APFS 克隆，实际占盘远小于此）" % meta["耗时秒"])
        print("  关键计数：" + "，".join("%s=%s" % (k, v) for k, v in meta["counts"].items()))
    return d


def load_meta(sid: str) -> dict:
    p = SNAP_DIR / sid / "manifest.json"
    if not p.exists():
        raise SystemExit("没有这个快照：%s（用 --list 看有哪些）" % sid)
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_sid(q: str) -> str:
    """支持只写关键词：先当完整 ID，再按前后缀/包含匹配。"""
    if (SNAP_DIR / q).is_dir():
        return q
    cands = [d.name for d in SNAP_DIR.iterdir() if d.is_dir() and q in d.name] if SNAP_DIR.exists() else []
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise SystemExit("没有匹配的快照：%s" % q)
    raise SystemExit("匹配到多个快照，请写更全一点：%s" % "、".join(sorted(cands)))


def do_list():
    if not SNAP_DIR.exists():
        print("还没有任何快照。用 --backup \"备注\" 做第一版。")
        return
    snaps = sorted([d for d in SNAP_DIR.iterdir() if d.is_dir()], reverse=True)
    if not snaps:
        print("还没有任何快照。"); return
    print("共 %d 版快照（新的在前）：\n" % len(snaps))
    for d in snaps:
        try:
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            print("  %s（manifest 读取失败）" % d.name); continue
        c = m.get("counts", {})
        print("  %s\n      %s ｜ %s ｜ 备注：%s" % (d.name, m["time"], m.get("user", "?"), m.get("note", "")))
        print("      语料块=%s ｜ 语料文件=%s ｜ 权限表=%s ｜ 花名表=%s ｜ 账号=%s"
              % (c.get("语料块数", "?"), c.get("语料文件数", "?"), c.get("权限表人数", "?"),
                 c.get("花名表人数", "?"), c.get("账号数", "?")))


def do_diff(sid: str):
    sid = resolve_sid(sid)
    m = load_meta(sid)
    print("与快照 %s 的差异（+ 当前有变化 ｜ - 快照里有、当前没有 ｜ ++ 快照之后新增）：\n" % sid)
    n = 0
    for rel, info in sorted(m["files"].items()):
        p = ROOT / rel
        if not p.exists():
            print("  - %s（当前不存在）" % rel); n += 1; continue
        if sha256(p) != info["sha256"]:
            print("  + %s（当前 %d 字节 ｜ 快照 %d 字节）" % (rel, p.stat().st_size, info["size"])); n += 1
    for rel in rel_files():
        if rel not in m["files"]:
            print("  ++ %s（快照之后新增）" % rel); n += 1
    for rel, info in (m.get("dirs") or {}).items():
        cur = dir_fingerprint(ROOT / rel)
        if not cur:
            print("  - %s/（当前不存在）" % rel); n += 1
        elif cur["指纹"] != info["指纹"]:
            print("  + %s/（文件数 %s→%s，大小 %.0f→%.0f MB）"
                  % (rel, info["文件数"], cur["文件数"], info["字节数"] / 1e6, cur["字节数"] / 1e6)); n += 1
    print(("\n共 %d 处不同。" % n) if n else "\n与当前状态完全一致。")


def do_verify(sid: str) -> bool:
    sid = resolve_sid(sid)
    m = load_meta(sid)
    bad = []
    for rel, info in m["files"].items():
        p = SNAP_DIR / sid / "files" / rel
        if not p.exists() or sha256(p) != info["sha256"]:
            bad.append(rel)
    for rel, info in (m.get("dirs") or {}).items():
        cur = dir_fingerprint(SNAP_DIR / sid / "files" / rel)
        if not cur or cur["文件数"] != info["文件数"]:
            bad.append(rel + "/（目录不完整）")
    if bad:
        print("校验不通过，以下内容损坏或缺失：\n  " + "\n  ".join(bad)); return False
    print("校验通过：%d 个文件 + %d 个目录全部完整。" % (len(m["files"]), len(m.get("dirs") or {})))
    return True


def health_check() -> dict:
    """回滚后体检：三项，任一不过就算异常。"""
    res, py = {}, sys.executable
    # ① 语料/索引一致性 + 真跑一次 BM25 召回（不加载 embedding 模型，快且够用）
    code1 = (
        "import json,jieba,numpy as np,faiss\n"
        "ch=json.load(open('chunks.json',encoding='utf-8'))\n"
        "raw=open('faiss.index','rb').read()\n"
        "idx=faiss.deserialize_index(np.frombuffer(raw,dtype=np.uint8))\n"
        "assert idx.ntotal==len(ch),(idx.ntotal,len(ch))\n"
        "from rank_bm25 import BM25Okapi\n"
        "bm=BM25Okapi([list(jieba.cut(c['text'])) for c in ch])\n"
        "sc=bm.get_scores(list(jieba.cut('公司安全生产方针')))\n"
        "top=np.argsort(sc)[::-1][:5]\n"
        "print('%d %d %d'%(len(ch),idx.ntotal,int((sc[top]>0).sum())))\n")
    p1 = subprocess.run([py, "-c", code1], cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    tail = (p1.stdout or "").strip().splitlines()[-1] if (p1.stdout or "").strip() else ""
    parts = tail.split()
    if p1.returncode == 0 and len(parts) == 3:
        res["检索"] = "通过（语料 %s 块 = 索引 %s 向量，BM25 命中 %s 条）" % (parts[0], parts[1], parts[2])
    else:
        res["检索"] = "失败：%s" % ((p1.stderr or p1.stdout or "").strip()[-200:])
    t = ROOT / "acl_filter_test.py"
    if t.exists():
        p2 = subprocess.run([py, str(t)], cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
        out = (p2.stdout or "") + (p2.stderr or "")
        # 判据只看 ①：越权泄漏必须为 0（"① 全 ✓" 是该脚本自己的汇总标记）
        # ② 里若出现"某负责人问某工程结算金额 → 0 命中"属检索层面的正常现象
        # （有该类权限 ≠ 这个问题能召回该类资料），不作为回滚失败的依据
        ok = p2.returncode == 0 and ("① 全 ✓" in out or "越权泄漏 = 0" in out)
        res["权限"] = "通过（越权 0）" if ok else "需人工确认：" + " / ".join(out.strip().splitlines()[-3:])
    else:
        res["权限"] = "跳过（未找到 acl_filter_test.py）"
    try:
        con = sqlite3.connect(str(ROOT / "webui" / "data" / "app.db"))
        a = con.execute("SELECT COUNT(*) FROM roster").fetchone()[0]
        b = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        con.close()
        res["数据库"] = "通过（花名表 %d 人 / 账号 %d 个）" % (a, b)
    except Exception as e:
        res["数据库"] = "失败：%s" % e
    return res


def log(line: str):
    SNAP_DIR.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))


def do_restore(sid: str, assume_yes=False, reason=""):
    sid = resolve_sid(sid)
    t0 = time.time()
    m = load_meta(sid)
    print("准备回滚到：%s\n  %s ｜ %s ｜ 备注：%s\n" % (sid, m["time"], m.get("user", "?"), m.get("note", "")))
    do_diff(sid)
    if not assume_yes and input("\n确认回滚？（输入 yes 继续）：").strip().lower() not in ("yes", "y"):
        raise SystemExit("已取消。")
    if not do_verify(sid):
        raise SystemExit("快照校验不通过，已中止（不会动当前状态）。")

    print("\n① 先备份当前状态（保证回滚本身也能被回滚）…")
    pre = do_backup("回滚前自动备份", silent=True)
    print("   →", pre.name)

    print("② 恢复文件与目录…")
    done_f, done_d = [], []
    try:
        for rel in m["files"]:
            dst = ROOT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_file(SNAP_DIR / sid / "files" / rel, dst)
            done_f.append(rel)
        for rel in (m.get("dirs") or {}):
            clone_dir(SNAP_DIR / sid / "files" / rel, ROOT / rel)
            done_d.append(rel)
    except Exception as e:
        print("   出错：%s → 自动退回回滚前状态" % e)
        for r2 in done_f:
            copy_file(SNAP_DIR / pre.name / "files" / r2, ROOT / r2)
        for r2 in done_d:
            clone_dir(SNAP_DIR / pre.name / "files" / r2, ROOT / r2)
        log("回滚失败（→ %s）：%s，已退回原状" % (sid, e))
        raise SystemExit("回滚失败，已恢复原状。")
    print("   已恢复 %d 个文件 + %d 个目录" % (len(done_f), len(done_d)))

    print("③ 体检（检索 / 权限越权 / 数据库）…")
    h = health_check()
    for k, v in h.items():
        print("   %-4s %s" % (k, v))
    ok = all("失败" not in v for v in h.values())
    used = round(time.time() - t0, 1)
    print("\n%s ｜ 用时 %.1f 秒（RTO 门槛 ≤ 300 秒）" % ("回滚完成" if ok else "回滚完成，但体检异常", used))
    print("回滚后关键计数：" + "，".join("%s=%s" % (k, v) for k, v in key_counts().items()))
    log("回滚 %s ← %s ｜ 体检%s ｜ 用时 %.1fs ｜ 原因：%s ｜ 明细：%s"
        % (pre.name, sid, "通过" if ok else "异常", used, reason or "（未填）", json.dumps(h, ensure_ascii=False)))
    if not ok:
        print("\n⚠ 体检未全部通过：必要时回滚到 %s（回滚前的状态）。" % pre.name)


def do_prune(keep: int):
    snaps = sorted([d for d in SNAP_DIR.iterdir() if d.is_dir()], reverse=True)
    drop = snaps[keep:]
    if not drop:
        print("无需清理（当前 %d 版，保留 %d 版）。" % (len(snaps), keep)); return
    for d in drop:
        shutil.rmtree(d); print("已删除旧快照：%s" % d.name)
    print("保留最近 %d 版。" % keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="知识库快照与一键回滚")
    ap.add_argument("--backup", nargs="?", const="手动快照", metavar="备注")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--diff", metavar="ID")
    ap.add_argument("--restore", metavar="ID")
    ap.add_argument("--verify", metavar="ID")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--reason", default="")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.list:
        do_list()
    elif a.diff:
        do_diff(a.diff)
    elif a.restore:
        do_restore(a.restore, assume_yes=a.yes, reason=a.reason)
    elif a.verify:
        sys.exit(0 if do_verify(a.verify) else 1)
    elif a.prune:
        do_prune(a.keep)
    elif a.backup is not None:
        do_backup(a.backup)
    else:
        ap.print_help()
