# -*- coding: utf-8 -*-
"""
阶段12：检索消融 —— BM25 / 向量 / 混合(RRF) × 有无rerank，测 MRR/Recall/Precision
- 语料：rebuild 后的单层400索引（chunks.json + faiss.index）
- ground-truth：答案包含（chunk 归一化文本含答案）
- 6 配置 × 3 指标
用法：.venv\\Scripts\\python.exe stage12_retrieval.py
"""
import os
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import json, re
from pathlib import Path
import faiss, numpy as np, jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE = Path("E:/数字员工项目")
INDEX = BASE / "faiss.index"
CHUNKS = BASE / "chunks.json"
K = 10

print("加载模型 ...", flush=True)
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")
print("模型就绪", flush=True)

# 载入索引
with open(INDEX, "rb") as f:
    fa = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
data = json.loads(CHUNKS.read_text(encoding="utf-8"))
texts = [d["text"] for d in data]
print(f"语料 {len(texts)} 块", flush=True)
bm25 = BM25Okapi([list(jieba.cut(t)) for t in texts])

def norm(s):
    return (s.replace(" ","").replace("\u3000","").replace("\n","").replace("\r","").replace("\t","").replace(",","").replace("，",""))
norm_texts = [norm(t) for t in texts]

# 载入测试集
evals = []
for name in ["评估集_混合.json", "评估集_新增.json"]:
    p = BASE / name
    if p.exists():
        evals += json.loads(p.read_text(encoding="utf-8"))
print(f"测试集 {len(evals)} 题", flush=True)

# 逐题算 ground-truth（相关块 = 含答案的块）
def relevant_set(nv):
    return {i for i, nt in enumerate(norm_texts) if nv in nt}

# 检索函数
def retrieve(q, method, rerank_flag, k=K):
    n = 20 if rerank_flag else k
    if method == "vector":
        qv = embed_model.encode([q]).astype("float32")
        _, ids = fa.search(qv, n); ids = ids[0].tolist()
    elif method == "bm25":
        scores = bm25.get_scores(list(jieba.cut(q)))
        ids = np.argsort(scores)[::-1][:n].tolist()
    else:  # hybrid
        qv = embed_model.encode([q]).astype("float32")
        _, vid = fa.search(qv, n); vid = vid[0].tolist()
        scores = bm25.get_scores(list(jieba.cut(q)))
        bid = np.argsort(scores)[::-1][:n].tolist()
        rrf = {}
        for r,i in enumerate(vid): rrf[i] = rrf.get(i,0)+1/(60+r)
        for r,i in enumerate(bid): rrf[i] = rrf.get(i,0)+1/(60+r)
        ids = sorted(rrf, key=rrf.get, reverse=True)[:n]
    if rerank_flag:
        sc = reranker.predict([(q, texts[i]) for i in ids])
        ids = [i for i,_ in sorted(zip(ids, sc), key=lambda x:x[1], reverse=True)[:k]]
    return ids[:k]

# 指标
def metrics(ids, rel):
    if not rel:
        return 0.0, 0.0, 0.0
    # MRR
    mrr = 0.0
    for rank, i in enumerate(ids, 1):
        if i in rel:
            mrr = 1.0/rank; break
    # Recall(Hit)
    rec = 1.0 if any(i in rel for i in ids) else 0.0
    # Precision
    prec = sum(1 for i in ids if i in rel) / len(ids)
    return mrr, rec, prec

methods = ["bm25", "vector", "hybrid"]
print(f"\n{'配置':<16} {'MRR@10':>8} {'Recall@10':>10} {'Precision@10':>12}", flush=True)
results = {}
for m in methods:
    for rr in [False, True]:
        label = f"{m}{'+rerank' if rr else ''}"
        agg = [0.0, 0.0, 0.0]
        n_used = 0
        for it in evals:
            nv = norm(it["a"])
            rel = relevant_set(nv)
            if not rel:
                continue  # 答案不在语料，跳过
            ids = retrieve(it["q"], m, rr)
            mm, rc, pc = metrics(ids, rel)
            agg[0]+=mm; agg[1]+=rc; agg[2]+=pc
            n_used += 1
        if n_used:
            agg = [x/n_used for x in agg]
            results[label] = agg
            print(f"{label:<16} {agg[0]:>8.4f} {agg[1]:>10.4f} {agg[2]:>12.4f}  (n={n_used})", flush=True)

print("\n===== 结论 =====", flush=True)
print(f"混合检索 vs 纯BM25/纯向量 的 MRR 差:", flush=True)
for m in ["bm25","vector"]:
    for rr in [False, True]:
        base = results.get(f"hybrid{'+rerank' if rr else ''}")
        other = results.get(f"{m}{'+rerank' if rr else ''}")
        if base and other:
            print(f"  hybrid{'rerank' if rr else ''} - {m}{'rerank' if rr else ''} = {base[0]-other[0]:+.4f}", flush=True)
