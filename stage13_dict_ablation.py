
# -*- coding: utf-8 -*-
"""阶段13：分词词典消融 —— 默认jieba vs 自定义词典(gas_dict.txt) × 召回20/50，测 BM25+rerank 的 MRR/Recall/Precision"""
import os
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 关键：强制离线，跳过联网检查 modules.json
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import json, re, time
from pathlib import Path
import faiss, numpy as np, jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

BASE = Path("E:/数字员工项目")
CHUNKS = BASE / "chunks.json"
K = 10
LOG = BASE / "_stage13.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("加载 reranker ...")
reranker = CrossEncoder("BAAI/bge-reranker-base")
log("reranker 就绪")

data = json.loads(CHUNKS.read_text(encoding="utf-8"))
texts = [d["text"] for d in data]
log(f"语料 {len(texts)} 块")

def norm(s):
    return (s.replace(" ","").replace("\u3000","").replace("\n","").replace("\r","")
             .replace("\t","").replace(",","").replace("，",""))
norm_texts = [norm(t) for t in texts]

# 两个独立分词器
default_tok = jieba.Tokenizer()
custom_tok = jieba.Tokenizer()
custom_tok.load_userdict(str(BASE / "gas_dict.txt"))
log("分词器就绪（默认 + 自定义词典）")

# 构建两个 BM25
log("构建 BM25（默认分词）...")
bm25_default = BM25Okapi([list(default_tok.cut(t)) for t in texts])
log("构建 BM25（自定义词典）...")
bm25_custom = BM25Okapi([list(custom_tok.cut(t)) for t in texts])
log("两个 BM25 就绪")

# 测试集（与 stage12 一致）
evals = []
for name in ["评估集_混合.json", "评估集_新增.json"]:
    p = BASE / name
    if p.exists():
        evals += json.loads(p.read_text(encoding="utf-8"))
log(f"测试集 {len(evals)} 题")

def relevant_set(nv):
    return {i for i, nt in enumerate(norm_texts) if nv in nt}

def retrieve(q, bm25, tok, n):
    scores = bm25.get_scores(list(tok.cut(q)))
    ids = np.argsort(scores)[::-1][:n].tolist()
    sc = reranker.predict([(q, texts[i]) for i in ids])
    ids = [i for i,_ in sorted(zip(ids, sc), key=lambda x:x[1], reverse=True)[:K]]
    return ids

def metrics(ids, rel):
    if not rel:
        return 0.0, 0.0, 0.0
    mrr = 0.0
    for rank, i in enumerate(ids, 1):
        if i in rel:
            mrr = 1.0/rank; break
    rec = 1.0 if any(i in rel for i in ids) else 0.0
    prec = sum(1 for i in ids if i in rel) / len(ids)
    return mrr, rec, prec

configs = [
    ("默认分词 n=20", bm25_default, default_tok, 20),
    ("自定义词典 n=20", bm25_custom, custom_tok, 20),
    ("默认分词 n=50", bm25_default, default_tok, 50),
    ("自定义词典 n=50", bm25_custom, custom_tok, 50),
]

log("\n配置               MRR@10   Recall@10  Prec@10")
results = {}
for label, bm25, tok, n in configs:
    agg = [0.0,0.0,0.0]
    n_used = 0
    for it in evals:
        nv = norm(it["a"])
        rel = relevant_set(nv)
        if not rel:
            continue
        ids = retrieve(it["q"], bm25, tok, n)
        mm, rc, pc = metrics(ids, rel)
        agg[0]+=mm; agg[1]+=rc; agg[2]+=pc
        n_used += 1
    if n_used:
        agg = [x/n_used for x in agg]
        results[label] = agg
        log(f"{label:<18} {agg[0]:.4f}  {agg[1]:.4f}  {agg[2]:.4f}  (n={n_used})")

log("\n===== 词典增益（同 n 对比）=====")
for n in [20, 50]:
    base_label = f"默认分词 n={n}"
    cust_label = f"自定义词典 n={n}"
    if base_label in results and cust_label in results:
        d = results[cust_label][0] - results[base_label][0]
        log(f"  n={n}: 自定义 - 默认 = {d:+.4f} (MRR)")

log("DONE")
