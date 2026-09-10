# -*- coding: utf-8 -*-
"""检索评估脚本：输入策略组合 + 评估集，输出 MRR/Recall/Precision 对比表。

用法（每次扩充数据、重建索引后跑，让数据决定当前最优策略，而不是靠记忆）：
    .venv\\Scripts\\python.exe eval_retrieval.py
    .venv\\Scripts\\python.exe eval_retrieval.py --strategies bm25,hybrid,vector
    .venv\\Scripts\\python.exe eval_retrieval.py --rerank off --n 20
    .venv\\Scripts\\python.exe eval_retrieval.py --evals 评估集_混合.json,评估集_新增.json

与生产一致：复用 retrieval.py 的 candidate_ids / rerank_scores / search。
"""
import os
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import argparse, json, re
from pathlib import Path
import numpy as np, jieba, faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import retrieval

BASE = Path("E:/数字员工项目")


def norm(s):
    return (s.replace(" ", "").replace("\u3000", "").replace("\n", "").replace("\r", "")
             .replace("\t", "").replace(",", "").replace("，", ""))


def metrics(ids, rel):
    if not rel:
        return 0.0, 0.0, 0.0
    mrr = 0.0
    for rank, i in enumerate(ids, 1):
        if i in rel:
            mrr = 1.0 / rank
            break
    rec = 1.0 if any(i in rel for i in ids) else 0.0
    prec = sum(1 for i in ids if i in rel) / len(ids)
    return mrr, rec, prec


def main():
    ap = argparse.ArgumentParser(description="检索策略评估")
    ap.add_argument("--strategies", default="bm25,vector,hybrid",
                    help="逗号分隔的检索策略：bm25/vector/hybrid")
    ap.add_argument("--rerank", default="both", choices=["both", "on", "off"],
                    help="both=测有无rerank两档；on=只测rerank；off=只测无rerank")
    ap.add_argument("--n", type=int, default=20, help="召回条数")
    ap.add_argument("--k", type=int, default=10, help="top-k")
    ap.add_argument("--evals", default="评估集_混合.json,评估集_新增.json",
                    help="评估集文件，逗号分隔")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    evals_files = [s.strip() for s in args.evals.split(",")]

    need_embed = any(s in ("vector", "hybrid") for s in strategies)
    need_rerank = args.rerank in ("both", "on")
    print(f"加载模型 ...（embed={need_embed}, rerank={need_rerank}）", flush=True)
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5") if need_embed else None
    reranker = CrossEncoder("BAAI/bge-reranker-base") if need_rerank else None
    print("模型就绪", flush=True)

    data = json.loads((BASE / "chunks.json").read_text(encoding="utf-8"))
    texts = [d["text"] for d in data]
    with open(BASE / "faiss.index", "rb") as f:
        fa = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
    bm25 = BM25Okapi([list(jieba.cut(t)) for t in texts])
    evals = []
    for name in evals_files:
        p = BASE / name
        if p.exists():
            evals += json.loads(p.read_text(encoding="utf-8"))
    norm_texts = [norm(t) for t in texts]
    print(f"语料 {len(texts)} 块，测试集 {len(evals)} 题\n", flush=True)

    use_rerank_list = [True, False] if args.rerank == "both" else [args.rerank == "on"]

    print(f"{'配置':<22} {'MRR@10':>8} {'Recall@10':>10} {'Prec@10':>12}", flush=True)
    for strat in strategies:
        for use_rerank in use_rerank_list:
            agg = [0.0, 0.0, 0.0]
            n_used = 0
            for it in evals:
                nv = norm(it["a"])
                rel = {i for i, nt in enumerate(norm_texts) if nv in nt}
                if not rel:
                    continue
                if use_rerank:
                    ids = retrieval.search(it["q"], bm25, texts, reranker, fa, embed_model,
                                           k=args.k, n=args.n, strategy=strat)
                else:
                    ids = retrieval.candidate_ids(it["q"], bm25, fa, embed_model,
                                                  n=args.k, strategy=strat)
                mm, rc, pc = metrics(ids, rel)
                agg[0] += mm; agg[1] += rc; agg[2] += pc
                n_used += 1
            if n_used:
                agg = [x / n_used for x in agg]
                label = f"{strat}{' + rerank' if use_rerank else ''}"
                print(f"{label:<22} {agg[0]:>8.4f} {agg[1]:>10.4f} {agg[2]:>12.4f}  (n={n_used})", flush=True)

    print("\n提示：ground-truth 用'答案包含'判据，金额答案有精度坑（四舍五入 vs 全精度浮点）。", flush=True)


if __name__ == "__main__":
    main()
