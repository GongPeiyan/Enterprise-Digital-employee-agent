
# -*- coding: utf-8 -*-
"""阶段14：表格去重修复后的完整 BM25+rerank 验证（对比修复前 0.860/0.940/0.232）"""
import os
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import json, re, time
from pathlib import Path
import numpy as np, jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

BASE = Path("E:/数字员工项目")
K = 10
LOG = BASE / "_stage14.log"

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("加载 reranker ...")
reranker = CrossEncoder("BAAI/bge-reranker-base")
log("reranker 就绪")

data = json.loads((BASE / "chunks.json").read_text(encoding="utf-8"))
texts = [d["text"] for d in data]
log(f"语料 {len(texts)} 块")

def norm(s):
    return (s.replace(" ","").replace("\u3000","").replace("\n","").replace("\r","")
             .replace("\t","").replace(",","").replace("，",""))
norm_texts = [norm(t) for t in texts]

bm25 = BM25Okapi([list(jieba.cut(t)) for t in texts])
log("BM25 就绪")

evals = []
for name in ["评估集_混合.json", "评估集_新增.json"]:
    p = BASE / name
    if p.exists():
        evals += json.loads(p.read_text(encoding="utf-8"))
log(f"测试集 {len(evals)} 题")

def relevant_set(nv):
    return {i for i, nt in enumerate(norm_texts) if nv in nt}

def retrieve(q, n=20):
    scores = bm25.get_scores(list(jieba.cut(q)))
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

agg = [0.0,0.0,0.0]
n_used = 0
miss = []
for it in evals:
    nv = norm(it["a"])
    rel = relevant_set(nv)
    if not rel:
        continue
    ids = retrieve(it["q"])
    mm, rc, pc = metrics(ids, rel)
    agg[0]+=mm; agg[1]+=rc; agg[2]+=pc
    n_used += 1
    if not rc:
        miss.append(it["q"])

agg = [x/n_used for x in agg]
log(f"\n修复后 BM25+rerank n=20:  MRR {agg[0]:.4f}  Recall {agg[1]:.4f}  Prec {agg[2]:.4f}  (n={n_used})")
log(f"修复前 baseline:           MRR 0.8600  Recall 0.9400  Prec 0.2320")
log(f"差异:                      MRR {agg[0]-0.86:+.4f}  Recall {agg[1]-0.94:+.4f}  Prec {agg[2]-0.232:+.4f}")
if miss:
    log(f"\n仍漏 {len(miss)} 题:")
    for q in miss:
        log(f"  ✗ {q}")
log("DONE")
