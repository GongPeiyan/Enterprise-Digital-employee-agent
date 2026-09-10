# -*- coding: utf-8 -*-
"""
阶段4：父子文档检索（small-to-big）
- 小块(约200字符)负责召回，大块(约800字符)负责回答
- 索引只建在小块上，召回后映射回父块喂给 LLM
- 纯本地评估：对比「单层切块(现状400字符)」vs「父子切块」的答案关键词命中率

设计说明（对应你的想法）：
- 你原设计"小块150token/大块800token"，这里用字符数实现。
  中文 1 字 ≈ 1 token（DeepSeek 中文 tokenizer 近似），
  小块取 200 字符是因为 150 字符太碎(约75汉字)会伤召回。
- 你问"LLM写chunk说明做embedding vs 延迟切分"，本步都不用：
  摘要 embedding 有 query/chunk 语义错位问题；延迟切分需长上下文模型，进阶再做。
"""
import os
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import json, re
from pathlib import Path
import pymupdf, faiss, numpy as np, jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE = Path("E:/数字员工项目")
PDF = BASE / "docs" / "vbd_paper.pdf"
EVAL = BASE / "评估集.json"

embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ================= 1. 文档解析（行优先 + 标签前置，复用你的表格修复） =================
def extract_lines(page, y_tol=3.0):
    def is_data(w):
        return bool(re.match(r'^[\d±—✓\-\.\(\)]', w))
    words = page.get_text("words")
    words.sort(key=lambda w: (round(w[1], 1), w[0]))
    raw_lines, cur, cur_y = [], [], None
    for w in words:
        y = w[1]
        if cur_y is None or abs(y - cur_y) > y_tol:
            if cur:
                raw_lines.append(cur)
            cur, cur_y = [w[4]], y
        else:
            cur.append(w[4])
    if cur:
        raw_lines.append(cur)
    result = []
    for ws in raw_lines:
        dc = sum(1 for w in ws if is_data(w))
        if dc >= 2 and len(ws) - dc >= 1:
            labels = [w for w in ws if not is_data(w)]
            datas = [w for w in ws if is_data(w)]
            result.append(" ".join(labels + datas))
        else:
            result.append(" ".join(ws))
    return result

def load_pdf(path):
    doc = pymupdf.open(str(path))
    lines = []
    for page in doc:
        lines.extend(extract_lines(page))
    return lines

# ================= 2. 切块：单层 vs 父子 =================
def split_chunks_flat(lines, chunk_size=400):
    """现状：单层固定大小切块"""
    chunks, current, cur_len = [], [], 0
    for line in lines:
        if current and cur_len + len(line) > chunk_size:
            chunks.append("\n".join(current))
            current, cur_len = [], 0
        current.append(line)
        cur_len += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks

def split_chunks_parent_child(lines, child_size=200, parent_size=800):
    """父子切块：先切大块(父)，每个父块内再切小块(子)。返回 (子块列表, 子→父映射, 父块列表)"""
    # 先按 parent_size 切父块
    parents, cur, cur_len = [], [], 0
    for line in lines:
        if cur and cur_len + len(line) > parent_size:
            parents.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        parents.append("\n".join(cur))
    # 每个父块内按 child_size 切子块
    child_chunks, child_to_parent = [], []
    for pi, ptext in enumerate(parents):
        ccur, ccur_len = [], 0
        for ln in ptext.split("\n"):
            if ccur and ccur_len + len(ln) > child_size:
                child_chunks.append("\n".join(ccur))
                child_to_parent.append(pi)
                ccur, ccur_len = [], 0
            ccur.append(ln)
            ccur_len += len(ln)
        if ccur:
            child_chunks.append("\n".join(ccur))
            child_to_parent.append(pi)
    return child_chunks, child_to_parent, parents

# ================= 3. 双索引 + 混合检索 =================
def build_indexes(chunks):
    vectors = np.array(embed_model.encode(chunks)).astype("float32")
    faiss_idx = faiss.IndexFlatL2(vectors.shape[1])
    faiss_idx.add(vectors)
    bm25 = BM25Okapi([list(jieba.cut(t)) for t in chunks])
    return faiss_idx, bm25

def hybrid_search(query, faiss_idx, bm25, n=20):
    q_vec = embed_model.encode([query]).astype("float32")
    _, vec_ids = faiss_idx.search(q_vec, n)
    vec_ids = vec_ids[0].tolist()
    bm25_ids = np.argsort(bm25.get_scores(list(jieba.cut(query))))[::-1][:n].tolist()
    rrf = {}
    for rank, i in enumerate(vec_ids):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
    for rank, i in enumerate(bm25_ids):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
    return sorted(rrf, key=rrf.get, reverse=True)

def rerank_top(query, candidates, chunks, k):
    pairs = [(query, chunks[i]) for i in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [i for i, s in ranked[:k]]

# ================= 4. 评估：答案关键词命中率 =================
STOP = set("的了和是在与或等及（）()【】,，.。:：;；-—%％、/")
def answer_keywords(ans):
    words = [w for w in jieba.lcut(ans)
             if len(w) >= 2 and w.strip() and not all(ch in STOP for ch in w)]
    return words

def hit_rate(context_text, keywords):
    if not keywords:
        return 1.0
    return sum(1 for w in keywords if w in context_text) / len(keywords)

# ================= 5. 主流程 =================
print("加载 PDF ...")
lines = load_pdf(PDF)
print(f"提取 {len(lines)} 行")

evals = json.loads(EVAL.read_text(encoding="utf-8"))
print(f"评估集 {len(evals)} 条\n")

# ---------- 方案A：单层切块（现状） ----------
print("=" * 60)
print("方案A：单层切块 400 字符（现状）")
print("=" * 60)
flat = split_chunks_flat(lines, 400)
fa, bm = build_indexes(flat)
print(f"切出 {len(flat)} 块")

# ---------- 方案B：父子切块 ----------
print("=" * 60)
print("方案B：父子切块 200/800 字符")
print("=" * 60)
childs, child2parent, parents = split_chunks_parent_child(lines, 200, 800)
fc, bmc = build_indexes(childs)
print(f"父块 {len(parents)} 个，子块 {len(childs)} 个，平均每父块 {len(childs)/max(len(parents),1):.1f} 个子块")

# ---------- 评估对比 ----------
K = 3
print("\n" + "=" * 60)
print(f"评估：top-{K} 检索，答案关键词命中率（本地，不调 API）")
print("=" * 60)

hitA_list, hitB_list = [], []
detail = []
for item in evals:
    q, a = item["q"], item["a"]
    kws = answer_keywords(a)
    # 方案A：单层
    candA = hybrid_search(q, fa, bm)[:20]
    topA = rerank_top(q, candA, flat, K)
    ctxA = "\n".join(flat[i] for i in topA)
    hitA = hit_rate(ctxA, kws)
    # 方案B：父子（子块召回 -> 映射父块）
    candB = hybrid_search(q, fc, bmc)[:20]
    topB_children = rerank_top(q, candB, childs, K)
    parent_ids = []
    for ci in topB_children:
        pi = child2parent[ci]
        if pi not in parent_ids:
            parent_ids.append(pi)
    ctxB = "\n".join(parents[pi] for pi in parent_ids)
    hitB = hit_rate(ctxB, kws)
    hitA_list.append(hitA)
    hitB_list.append(hitB)
    detail.append((item["id"], item["cat"], round(hitA,3), round(hitB,3), len(kws)))

print(f"\n{'id':>3} {'类别':<8} {'关键词数':>6} {'单层命中':>8} {'父子命中':>8}")
for i, cat, hA, hB, nkw in detail:
    flag = "▲" if hB > hA else ("▼" if hB < hA else "=")
    print(f"{i:>3} {cat:<8} {nkw:>6} {hA:>8.3f} {hB:>8.3f} {flag}")

print(f"\n平均命中率：单层={sum(hitA_list)/len(hitA_list):.3f}  父子={sum(hitB_list)/len(hitB_list):.3f}")
print(f"改进：(父子-单层) = {sum(hitB_list)/len(hitB_list)-sum(hitA_list)/len(hitA_list):+.3f}")
