# -*- coding: utf-8 -*-
# 阶段1：最简 RAG —— 把一份文档喂进去，能问答
# 跑之前：把一份中文 PDF 放到 E:/数字员工项目 下，改下面 PDF_PATH
# 运行：.venv\Scripts\python.exe stage1_rag.py

import os
# 国内下载 HuggingFace 模型的三个开关（必须在 import 模型库之前设好）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import fitz  # pymupdf，读 PDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ========== 0. 配置 ==========
PDF_PATH = "E:/数字员工项目/Variational_Bayesian_Distributed_Q_Learning_with_Adaptive_Annealing_for_Offline_Reinforcement_Learning__1_.pdf"   # ← 改成你的 PDF 文件名
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY：请先设置环境变量，或在项目根目录 .env 里配置")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ========== 1. 加载：读 PDF，抽出纯文本 ==========
def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ========== 2. 切块：把长文切成小块（先最简单版，阶段2再优化） ==========
def split_chunks(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# ========== 3+4. 向量化 + 建索引 ==========
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

def build_index(chunks):
    vectors = model.encode(chunks)           # 每块文字 -> 一个向量
    vectors = np.array(vectors).astype("float32")
    dim = vectors.shape[1]                    # 向量的维度
    index = faiss.IndexFlatL2(dim)            # 用欧氏距离找最近的向量
    index.add(vectors)                        # 把所有向量存进去
    return index

# ========== 5. 检索：找和问题最像的几块 ==========
def retrieve(query, index, chunks, k=3):
    q_vec = model.encode([query]).astype("float32")
    distances, ids = index.search(q_vec, k)   # 返回最近的 k 块的下标
    return [chunks[i] for i in ids[0]]

# ========== 6. 生成：把检索结果塞给 LLM 回答 ==========
def answer(query, context):
    prompt = (
        "你根据下面【参考资料】回答问题，资料里没有的就说不知道，不要编。\n"
        f"【参考资料】\n{context}\n"
        f"【问题】\n{query}"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )
    return resp.choices[0].message.content

# ========== 主流程 ==========
text = load_pdf(PDF_PATH)
chunks = split_chunks(text)
index = build_index(chunks)
print(f"✅ 文档加载完成，切成 {len(chunks)} 块")

while True:
    q = input("\n问：")
    if q.strip().lower() == "exit":
        break
    context = "\n\n".join(retrieve(q, index, chunks))
    print("AI：", answer(q, context))
