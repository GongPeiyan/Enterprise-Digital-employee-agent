
# -*- coding: utf-8 -*-
"""阶段13b：自定义词典 vs 已知 baseline(0.860) —— 跑 自定义n=20 和 自定义n=50"""
import os
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
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
LOG = BASE / "_stage13b.log"

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

custom_tok = jieba.Tokenizer()
custom_tok.load_userdict(str(BASE / "gas_dict.txt"))
log("自定义词典分词器就绪")

log("构建 BM25（自定义词典）...")
bm25_custom = BM25Okapi([list(custom_tok.cut(t)) for t in texts])
log("BM25 就绪")

evals = []
for name in ["评估集_混合.json", "评估集_新增.json"]:
    p = BASE / name
    if p.exists():
        evals += json.loads(p.read_text(encoding="utf-8"))
log(f"测试集 {len(evals)} 题")

def relevant_set(nv):
    return {i for i, nt in enumerate(norm_texts) if nv in nt}

def retrieve(q, n):
    scores = bm25_custom.get_scores(list(custom_tok.cut(q)))
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

log("\n配置               MRR@10   Recall@10  Prec@10")
log("默认分词 n=20(baseline) 0.8600  0.9400  0.2320  (n=50)  [引用 stage12]")
results = {}
for n in [20, 50]:
    agg = [0.0,0.0,0.0]
    n_used = 0
    per_question = []
    for it in evals:
        nv = norm(it["a"])
        rel = relevant_set(nv)
        if not rel:
            continue
        ids = retrieve(it["q"], n)
        mm, rc, pc = metrics(ids, rel)
        agg[0]+=mm; agg[1]+=rc; agg[2]+=pc
        n_used += 1
        per_question.append((it["q"], mm))
    if n_used:
        agg = [x/n_used for x in agg]
        results[n] = agg
        log(f"自定义词典 n={n:<4}     {agg[0]:.4f}  {agg[1]:.4f}  {agg[2]:.4f}  (n={n_used})")

log("\n===== 词典增益（n=20）=====")
log(f"  自定义 {results[20][0]:.4f} - 默认 0.8600 = {results[20][0]-0.8600:+.4f} (MRR)")
if 50 in results:
    log(f"\n===== 召回条数影响（自定义词典）=====")
    log(f"  n=20 {results[20][0]:.4f} vs n=50 {results[50][0]:.4f} = {results[50][0]-results[20][0]:+.4f} (MRR)")

log("DONE")
