# -*- coding: utf-8 -*-
# 阶段2：复杂 RAG —— 混合检索 + 重排 + query改写
# 运行：.venv\Scripts\python.exe stage2_rag.py

import os
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import pymupdf
import faiss
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# ========== 0. 配置 ==========
PDF_PATH = "E:/数字员工项目/vbd_paper.pdf"
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY：请先设置环境变量，或在项目根目录 .env 里配置")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ========== 1. 加载 + 切块（同阶段1） ==========
def load_pdf(path):
    doc = pymupdf.open(path)
    return "".join(page.get_text() for page in doc)

def split_chunks(text, chunk_size=400, overlap=50):
    chunks = []              
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# ========== 2. 两个模型 + 双索引 ==========
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")      # 向量：粗排用
reranker = CrossEncoder("BAAI/bge-reranker-base")                 # 精排用

def build_indexes(chunks):
    # 向量索引（faiss）
    vectors = np.array(embed_model.encode(chunks)).astype("float32")
    dim = vectors.shape[1]
    faiss_idx = faiss.IndexFlatL2(dim)
    faiss_idx.add(vectors)
    # BM25 关键词索引（先 jieba 分词）
    tokenized = [list(jieba.cut(c)) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return faiss_idx, bm25

# ========== 3. query 改写 ==========
def rewrite_query(query):
    prompt = (
        "把下面这个问题改写成一个适合检索的、明确的问句。\n"
        "如果问题本身已经很明确，就原样返回。\n"
        f"问题：{query}\n改写后："
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=100,
    )
    return resp.choices[0].message.content.strip()

# ========== 4. 混合检索（RRF 融合向量 + BM25） ==========
def hybrid_search(query, faiss_idx, bm25, chunks, n=20):
    # 向量检索 top-n
    q_vec = embed_model.encode([query]).astype("float32")
    _, vec_ids = faiss_idx.search(q_vec, n)
    vec_ids = vec_ids[0].tolist()
    # BM25 检索 top-n
    tokens = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokens)
    bm25_ids = np.argsort(bm25_scores)[::-1][:n].tolist()
    # RRF 融合：按"排名"打分，两种结果合并
    rrf = {}
    for rank, i in enumerate(vec_ids):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
    for rank, i in enumerate(bm25_ids):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
    return sorted(rrf, key=rrf.get, reverse=True)  # 返回候选下标，按 RRF 分排序

# ========== 5. 重排：对候选精排出最相关的 3 块 ==========
def rerank(query, cand_ids, chunks, k=3):
    cand_ids = cand_ids[:20]  # 最多精排 20 个候选
    pairs = [(query, chunks[i]) for i in cand_ids]
    scores = reranker.predict(pairs)  # 逐个算"query 和这块的相关性"
    ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, s in ranked[:k]]

# ========== 6. 生成 ==========
def answer(query, context):
    prompt = (
        "你根据下面【参考资料】回答问题，资料里没有的就说不知道，不要编。\n"
        f"【参考资料】\n{context}\n【问题】\n{query}"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=1000,
    )
    return resp.choices[0].message.content

# ========== 主流程 ==========
text = load_pdf(PDF_PATH)
chunks = split_chunks(text)
faiss_idx, bm25 = build_indexes(chunks)
print(f"✅ 加载完成：{len(chunks)} 块，向量索引 + BM25 索引都建好了")

while True:
    q = input("\n问：")
    if q.strip().lower() == "exit":
        break
    q2 = rewrite_query(q)
    cand_ids = hybrid_search(q2, faiss_idx, bm25, chunks)
    top = rerank(q2, cand_ids, chunks, k=3)
    print("AI：", answer(q, "\n\n".join(top)))
