# -*- coding: utf-8 -*-
# 阶段3：完整数字员工 = 多文档RAG + 索引持久化 + 长期记忆
# 用法：把 PDF 都丢进 docs/ 目录，首次运行自动预处理并存盘，之后秒开。
#       新文档用 add_document() 增量添加，不用重新处理老文档。
# 运行：.venv\Scripts\python.exe stage3.py

import os, json, re
from pathlib import Path

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

# ================= 配置 =================
DOCS_DIR = "E:/数字员工项目/docs"      # 所有 PDF 放这
INDEX_FILE = "E:/数字员工项目/faiss.index"
CHUNKS_FILE = "E:/数字员工项目/chunks.json"
MEMORY_FILE = "E:/数字员工项目/memories.json"

api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY：请先设置环境变量，或在项目根目录 .env 里配置")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ================= 1. 文档解析（行优先 + 标签前置） =================
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
        data_count = sum(1 for w in ws if is_data(w))
        if data_count >= 2 and len(ws) - data_count >= 1:
            labels = [w for w in ws if not is_data(w)]
            datas = [w for w in ws if is_data(w)]
            result.append(" ".join(labels + datas))
        else:
            result.append(" ".join(ws))
    return result

def load_pdf(path):
    doc = pymupdf.open(path)
    lines = []
    for page in doc:
        lines.extend(extract_lines(page))
    return lines

def split_chunks(lines, chunk_size=400):
    chunks, current, current_len = [], [], 0
    for line in lines:
        if current and current_len + len(line) > chunk_size:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks

# ================= 2. 索引持久化（多文档，累积式） =================
def build_index(texts):
    vectors = np.array(embed_model.encode(texts)).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def save_index(index, chunks, sources):
    # faiss 的 write_index 底层 C++ 不支持中文路径，改用 serialize + Python 写文件
    with open(INDEX_FILE, "wb") as f:
        f.write(faiss.serialize_index(index).tobytes())
    data = [{"text": t, "source": s} for t, s in zip(chunks, sources)]
    json.dump(data, open(CHUNKS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def load_or_build():
    """有缓存就加载（秒开），没缓存就首次处理 docs/ 里所有 PDF"""
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        with open(INDEX_FILE, "rb") as f:
            index = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
        data = json.load(open(CHUNKS_FILE, encoding="utf-8"))
        chunks = [d["text"] for d in data]
        sources = [d["source"] for d in data]
        return index, chunks, sources
    chunks, sources = [], []
    for pdf in sorted(Path(DOCS_DIR).glob("*.pdf")):
        for c in split_chunks(load_pdf(str(pdf))):
            chunks.append(c)
            sources.append(pdf.name)
    index = build_index(chunks)
    save_index(index, chunks, sources)
    return index, chunks, sources

def add_document(pdf_path):
    """新文档增量添加：只处理新文档，老文档不动"""
    index, chunks, sources = load_or_build()
    new_chunks = split_chunks(load_pdf(pdf_path))
    vectors = np.array(embed_model.encode(new_chunks)).astype("float32")
    index.add(vectors)
    chunks.extend(new_chunks)
    sources.extend([Path(pdf_path).name] * len(new_chunks))
    save_index(index, chunks, sources)
    print(f"✅ 新文档 {Path(pdf_path).name} 已加入（{len(new_chunks)} 块）")

# ================= 3. 检索（改写→混合检索→重排） =================
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

def hybrid_search(query, faiss_idx, bm25, n=20):
    q_vec = embed_model.encode([query]).astype("float32")
    _, vec_ids = faiss_idx.search(q_vec, n)
    vec_ids = vec_ids[0].tolist()
    tokens = list(jieba.cut(query))
    bm25_ids = np.argsort(bm25.get_scores(tokens))[::-1][:n].tolist()
    rrf = {}
    for rank, i in enumerate(vec_ids):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
    for rank, i in enumerate(bm25_ids):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
    return sorted(rrf, key=rrf.get, reverse=True)

def retrieve_top_ids(query, faiss_idx, bm25, chunks, k=3):
    """完整检索链，返回 top-k 的下标"""
    q2 = rewrite_query(query)
    cand_ids = hybrid_search(q2, faiss_idx, bm25)[:20]
    pairs = [(q2, chunks[i]) for i in cand_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
    return [i for i, s in ranked[:k]]

# ================= 4. 长期记忆 =================
def load_memories():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE, encoding="utf-8"))
    return []

def save_memories(mems):
    json.dump(mems, open(MEMORY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def extract_memory(user_msg, assistant_msg):
    prompt = (
        "从下面这段对话里，提取一条\"值得长期记住的事实\"（用户的名字、身份、偏好、重要决定、正在做的事）。\n"
        "如果没有值得记的，就只回复两个字：无\n"
        f"用户：{user_msg}\nAI：{assistant_msg}\n值得记住的事实："
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

# ================= 主流程 =================
index, chunks, sources = load_or_build()
bm25 = BM25Okapi([list(jieba.cut(t)) for t in chunks])

memories = load_memories()
memory_text = "\n".join(f"- {m}" for m in memories) if memories else "（暂无）"
messages = [{"role": "system", "content": f"你是用户的数字员工。你记得关于用户的这些事：\n{memory_text}"}]

print(f"✅ 知识库就绪：{len(chunks)} 块，来自 {len(set(sources))} 个文档")
print(f"✅ 长期记忆：{len(memories)} 条")
print("输入 exit 退出")

while True:
    q = input("\n问：")
    if q.strip().lower() == "exit":
        break
    top_ids = retrieve_top_ids(q, index, bm25, chunks)
    context = "\n\n".join(f"【来自 {sources[i]}】\n{chunks[i]}" for i in top_ids)
    messages.append({"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】\n{q}"})
    resp = client.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.3, max_tokens=1000
    )
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print("AI：", reply)
    fact = extract_memory(q, reply)
    if fact and fact != "无":
        memories.append(fact)
        save_memories(memories)
        print(f"  [🧠 记住了一条：{fact}]")
