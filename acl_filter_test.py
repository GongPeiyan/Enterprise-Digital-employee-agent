# -*- coding: utf-8 -*-
"""A3 验证 / A4 雏形：按真实语料模拟"某工号问某题，能看到/看不到哪些块"。

注意：本地没有 sentence_transformers（生产用的是它），所以本脚本验证的是**过滤逻辑**
（acl_roster 的判定 + 按路径查类别），不是 app.py 的接线。接线需要在一个装了
sentence_transformers 的环境里跑一次（你的 Windows 那台，或本地装依赖）。

做三件事：
  ① 反向（越权）：让"不该看到的人"问敏感题，检查敏感类别一条都进不来
  ② 正向（误伤）：让"该看到的人"问同类题，检查标答块没被误挡
  ③ 输出每个角色的可见块数，核对是否符合矩阵

用法：lora-from-scratch/.venv/bin/python acl_filter_test.py [--role 市场部员工]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import acl_roster                      # noqa: E402


def load():
    import _eval_mac as em
    chunks = json.loads((BASE / "chunks.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    srcs = [c["source"] for c in chunks]
    return em, texts, srcs


def retrieve(texts, srcs, bm, q, allowed, n=15):
    """完整复刻 app.py 里 A3 之后的检索：召回 → 权限过滤 → 重排 → top5。"""
    ids = bm.recall(q, n=n)
    kept, blocked = [], []
    for i in ids:
        cat = acl_roster.cat_of_block(srcs[i], texts[i])
        (kept if cat in allowed else blocked).append((i, cat))
    return kept, blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=0, help="每题试几个块（默认取全部命中）")
    a = ap.parse_args()

    em, texts, srcs = load()
    bm = em.Bm25(texts)
    roster = json.loads((BASE / "acl_roster.json").read_text(encoding="utf-8"))

    # 每个文件属于哪一类（和生产一样，按路径查）
    cat_cnt = Counter(acl_roster.cat_of_block(s, texts[i]) for i, s in enumerate(srcs))
    print("语料块 %d ｜ 各类别块数：" % len(texts))
    for k, v in cat_cnt.most_common():
        print("   %-16s %5d" % (k, v))

    # ③ 各角色可见块数（与矩阵核对）
    print("\n③ 各角色可见块数（按文件路径判定，同生产逻辑）")
    reps = [("人力资源主管", "HR001"), ("人力资源专员", "HR002"), ("工程部员工", "ENG002"),
            ("工程部负责人", "ENG001"), ("品技部员工", "QA002"), ("财务部员工", "FIN002"),
            ("财务负责人", "FIN001"), ("采购负责人", "SCM001"), ("研发部员工", "RD004"),
            ("市场部员工", "MKT002"), ("总经办员工", "GM002")]
    for name, eid in reps:
        allowed = acl_roster.visible_categories(eid)
        n = sum(1 for i, s in enumerate(srcs) if acl_roster.cat_of_block(s, texts[i]) in allowed)
        print("   %-14s %-8s 可见 %5d 块 (%4.1f%%)  类别：%s"
              % (name, eid, n, n * 100 / len(srcs), "／".join(sorted(allowed))))

    # ①② 越权 / 误伤 用例
    print("\n① 越权检查：不该看到的人问敏感题，敏感类别必须 0 条")
    cases = [
        ("薪酬福利管理制度的绩效工资怎么规定的？", "MKT002", "市场部员工", ["B 薪酬人事", "B2 个人薪酬明细"]),
        ("土建组1-12月工资汇总表里张三的工资是多少？", "ENG002", "工程部员工", ["B2 个人薪酬明细"]),
        ("某低压工程的结算金额是多少？", "RD004", "研发部员工", ["C 结算采购"]),
        ("施工组织方案里的管材规格是什么？", "MKT002", "市场部员工", ["D 项目技术"]),
    ]
    bad = 0
    for q, eid, who, forbidden in cases:
        allowed = acl_roster.visible_categories(eid)
        kept, blocked = retrieve(texts, srcs, bm, q, allowed)
        leak = [c for _i, c in kept if c in forbidden]
        print("   %s %s(%s) 问「%s」→ 放进 %d 条 / 挡下 %d 条；泄漏 %d 条"
              % ("✓" if not leak else "✗", who, eid, q[:16], len(kept), len(blocked), len(leak)))
        bad += 1 if leak else 0

    print("\n② 误伤检查：该看到的人问同类题，必须仍有命中（否则就是锁过头了）")
    for q, eid, who in [
        ("薪酬福利管理制度的绩效工资怎么规定的？", "HR001", "人力资源主管"),
        ("某低压工程的结算金额是多少？", "FIN001", "财务负责人"),
        ("施工组织方案里的管材规格是什么？", "ENG002", "工程部员工"),
        ("安全操作规程里对动火作业怎么要求的？", "MKT002", "市场部员工"),
    ]:
        allowed = acl_roster.visible_categories(eid)
        kept, blocked = retrieve(texts, srcs, bm, q, allowed)
        print("   %s %s(%s) 问「%s」→ 命中 %d 条（挡下 %d 条）"
              % ("✓" if kept else "✗", who, eid, q[:16], len(kept), len(blocked)))

    print("\n判据：越权泄漏 = 0（上面 ① 全 ✓）｜ 误伤 = 该看到的必须还有命中（② 全 ✓）")
    print("本地结论：%s" % ("过滤逻辑正确，可以进 A4 全量测试" if bad == 0 else "存在越权泄漏，需修规则"))


if __name__ == "__main__":
    main()
