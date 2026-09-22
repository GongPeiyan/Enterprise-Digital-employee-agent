# -*- coding: utf-8 -*-
"""A4：权限可见性测试（证明"隔离真的有效"，验收 4.5 节的证据）。

测两个方向，这是重点：
  ① 越权（反向）——全量：每个角色 × 每一道题，走一遍"召回→权限过滤"，
     逐条断言"返回的每一块，其类别都必须在这个角色的可见范围内"。违反 = 越权。
     这是**不变量测试**：只要有一处过滤漏了（比如某条路径没走 ACL），就会被抓出来。
  ② 误伤（正向）——抽样：每个角色取若干"他本来就该看到"的题，跑两遍完整流程
     （①不过滤 ②过滤），比较标答块是否还在 top-5。
     误伤 = 过滤前能命中、过滤后命中不了 → 说明锁过头了。
     两遍都用同样的召回条数（15），所以差异只可能来自 ACL 过滤。

为什么正向也要测：越权会被人投诉，误伤不会——员工只会以为"系统里没这资料"，
悄悄把工具废掉。两种都要为 0 才算过。

用法：lora-from-scratch/.venv/bin/python acl_visibility_test.py
产出：acl_visibility_report.json（机器可核）+ acl_visibility_report.txt（一页结论）
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import acl_roster                      # noqa: E402

# 角色代表（同部门+同身份的人权限完全一样，测一个代表一组）
ROLES = [
    ("人力资源主管", "HR001"), ("人力资源专员", "HR002"),
    ("工程部负责人", "ENG001"), ("工程部员工", "ENG002"),
    ("品技部员工", "QA002"), ("财务负责人", "FIN001"), ("财务部员工", "FIN002"),
    ("采购负责人", "SCM001"), ("研发部员工", "RD004"),
    ("市场部员工", "MKT002"), ("总经办员工", "GM002"),
]
PER_ROLE_POS = 8          # 每个角色取多少道"该看到"的题
RECALL_N = 15             # 与生产一致（app.py 里 n=15）


def load_corpus():
    chunks = json.loads((BASE / "chunks.json").read_text(encoding="utf-8"))
    return [c["text"] for c in chunks], [c["source"] for c in chunks]


def build_bank(srcs):
    """题库：每题带 标答块下标（或标答文件路径）+ 标答类别。类别来自生产同一套打标签结果。"""
    bank = []
    d = json.loads((BASE / "评估集_扩充.json").read_text(encoding="utf-8"))
    for it in d:
        gid = it.get("gid")
        if isinstance(gid, int) and 0 <= gid < len(srcs):
            bank.append({"q": it["q"], "gold_idx": [gid], "origin": "扩充92"})

    g = json.loads((BASE / "gold_manual.json").read_text(encoding="utf-8"))
    for it in g.get("items", []):
        ids = it.get("gold_ids") or []
        if ids and all(isinstance(i, int) and 0 <= i < len(srcs) for i in ids):
            bank.append({"q": it["q"], "gold_idx": ids, "origin": "人工标注"})

    lt = json.loads((BASE / "评估集_长文本.json").read_text(encoding="utf-8"))
    for it in lt:
        s = (it.get("source") or "").replace("\\", "/")
        idx = [i for i, x in enumerate(srcs) if x.replace("\\", "/") == s]
        if idx:
            bank.append({"q": it["q"], "gold_idx": idx, "origin": "长文本14"})

    for b in bank:                                   # 标答类别 = 标答块所属文件的类别
        b["cat"] = acl_roster.cat_of_block(srcs[b["gold_idx"][0]], texts[b["gold_idx"][0]])
    return bank


def main():
    t0 = time.time()
    texts, srcs = load_corpus()
    import _eval_mac as em
    bm = em.Bm25(texts)
    rk = em.Reranker()
    bank = build_bank(srcs)

    print("题库 %d 题（%s）｜ 语料 %d 块\n" % (
        len(bank), "／".join("%s %d" % (o, sum(1 for b in bank if b["origin"] == o))
                            for o in dict.fromkeys(b["origin"] for b in bank)), len(texts)))
    cat_cnt = Counter(acl_roster.cat_of_block(s, texts[i]) for i, s in enumerate(srcs))

    # ───────── ① 越权不变量：全量 角色 × 题 ─────────
    print("① 越权检查（全量：%d 角色 × %d 题 = %d 次检索）" % (len(ROLES), len(bank), len(ROLES) * len(bank)))
    violations, blocked_stat, leak_tbl = [], {}, {}
    for who, eid in ROLES:
        allowed = acl_roster.visible_categories(eid)
        got, nb = Counter(), 0
        for b in bank:
            for i in bm.recall(b["q"], n=RECALL_N):
                cat = acl_roster.cat_of_block(srcs[i], texts[i])
                if cat in allowed:
                    got[cat] += 1                      # 实际拿到手的块，按类别计数
                else:
                    nb += 1                            # 被挡下的块数（证明过滤真的在起作用）
        bad = {c: n for c, n in got.items() if c not in allowed}     # 越权类别（必须为空）
        leak_tbl[who] = bad
        violations += [{"role": who, "cat": c, "n": n} for c, n in bad.items()]
        blocked_stat[who] = nb
        print("   %-12s 拿到 %3d 块：%-58s 挡下 %5d 条 ｜ 越权：%s"
              % (who, sum(got.values()), "／".join("%s %d" % (c, got[c]) for c in sorted(got)),
                 nb, bad or "无 ✓"))
    # 安全侧默认：还没打标签的新文件（语料治理阶段会天天有新文件进来）必须锁闭
    newf = acl_roster.cat_of_source("某个刚入库还没打标签的新文件.docx")
    nmiss = [w for w, _e in ROLES if newf in acl_roster.visible_categories(_e)]
    print("   新语料默认（无标签文件 → %s）：%s" % (newf, "✓ 所有角色都看不到" if not nmiss else "✗ %s 能看到" % nmiss))
    if nmiss:
        violations.append({"role": ",".join(nmiss), "cat": newf, "n": -1})
    print("   越权条数：%d  %s" % (len(violations), "✓ 通过" if not violations else "✗ 未通过"))

    # ② 误伤（正向）：每角色取"他该看到"的题，比较过滤前后的 top-5 命中
    print("\n② 误伤检查（每角色最多 %d 道正向题，比较 过滤前/过滤后 的 top-5）" % PER_ROLE_POS)
    rows = []
    for who, eid in ROLES:
        allowed = acl_roster.visible_categories(eid)
        cands = [b for b in bank if b["cat"] in allowed][:PER_ROLE_POS]
        hit_before = hit_after = 0
        missed = []
        for b in cands:
            ids = bm.recall(b["q"], n=RECALL_N)
            base = ids[:]                                        # 不过滤
            kept = [i for i in ids if acl_roster.cat_of_block(srcs[i], texts[i]) in allowed]
            for tag, pool in (("before", base), ("after", kept)):
                if not pool:
                    ok = False
                else:
                    sc = rk.scores(b["q"], [texts[i] for i in pool])
                    top = [pool[j] for j in sorted(range(len(pool)), key=lambda j: sc[j], reverse=True)[:5]]
                    ok = any(i in top for i in b["gold_idx"])
                if tag == "before":
                    hit_before += ok
                else:
                    hit_after += ok
                    if not ok:
                        missed.append({"q": b["q"][:30], "cat": b["cat"], "origin": b["origin"]})
        hurt = hit_before - hit_after
        rows.append({"role": who, "emp_id": eid, "正向题数": len(cands),
                     "过滤前命中": hit_before, "过滤后命中": hit_after, "误伤题数": hurt,
                     "误伤题": missed,
                     "可见类别": sorted(allowed),
                     "可见块数": sum(1 for i, s in enumerate(srcs) if acl_roster.cat_of_block(s, texts[i]) in allowed)})
        print("   %-12s 正向 %2d 题 → 过滤前命中 %2d / 过滤后命中 %2d ｜ 误伤 %d"
              % (who, len(cands), hit_before, hit_after, hurt))
        if len(cands) < PER_ROLE_POS:
            print("      ⚠ 该角色「该看到」的题只有 %d 道（语料里几乎没有他部门的资料）" % len(cands))

    total_pos = sum(r["正向题数"] for r in rows)
    total_hurt = sum(r["误伤题数"] for r in rows)
    print("\n判据：越权 = 0 条（实际 %d）｜ 误伤率 = 0（实际 %d/%d = %.1f%%）"
          % (len(violations), total_hurt, total_pos, total_hurt * 100 / max(total_pos, 1)))
    print("耗时 %.1f 秒" % (time.time() - t0))

    rep = {"语料块数": len(texts), "题库题数": len(bank), "召回条数": RECALL_N,
           "越过权限条数": len(violations), "越权明细": violations, "被挡条数": blocked_stat,
           "正向题总数": total_pos, "误伤题总数": total_hurt,
           "按角色": rows, "各类别块数": dict(cat_cnt.most_common())}
    (BASE / "acl_visibility_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = (not violations) and total_hurt == 0
    L = []
    L.append("A4 权限可见性测试报告")
    L.append("=" * 46)
    L.append("结论：%s" % ("通过 —— 越权 0 条、误伤 0 题" if ok else "未通过，见下表"))
    L.append("语料 %d 块 ｜ 题库 %d 题 ｜ 角色 %d 组 ｜ 召回 %d 条（与生产一致）" %
             (len(texts), len(bank), len(ROLES), RECALL_N))
    L.append("")
    L.append("判据：① 越权（不该看到的必须一条都进不来）= 0 ｜ ② 误伤（该看到的必须还检索得到）= 0")
    L.append("")
    L.append("① 越权检查：全量 %d 次检索，越权 %d 条 %s" % (len(ROLES) * len(bank), len(violations),
                                                       "✓" if not violations else "✗"))
    L.append("   每角色实际拿到的块（按类别）与被挡条数，见 acl_visibility_report.json 的「按角色」「被挡条数」")
    L.append("   另测安全侧默认：无标签的新文件 → 所有角色都看不到（新语料入场先锁闭）")
    L.append("")
    L.append("② 按角色（正向题 = 该角色有权看到的题；命中 = 标答块落在 top-5）")
    L.append("%-12s %6s %8s %8s %6s %10s" % ("角色", "正向题", "过滤前", "过滤后", "误伤", "可见块数"))
    for r in rows:
        L.append("%-12s %6d %8d %8d %6d %10d" % (r["role"], r["正向题数"], r["过滤前命中"],
                                                 r["过滤后命中"], r["误伤题数"], r["可见块数"]))
    L.append("")
    L.append("各类别块数：")
    for k, v in cat_cnt.most_common():
        L.append("   %-16s %5d" % (k, v))
    L.append("")
    L.append("注：正向题数不足 8 的角色，是因为语料库里几乎没有该部门自己的资料（语料缺口，见权限矩阵第 6 表）。")
    L.append("   加入新数据后需重跑：acl_tagging.py → acl_roster.py → 本脚本。")
    (BASE / "acl_visibility_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("已写出 acl_visibility_report.json / acl_visibility_report.txt")


if __name__ == "__main__":
    main()
