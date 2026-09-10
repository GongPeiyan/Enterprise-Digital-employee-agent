# -*- coding: utf-8 -*-
"""
阶段6：按文档类型路由切分 + 三种切分 MRR 对比

三种策略：
  1. 单层400       —— 现状基线（按 400 字符累积多行）
  2. 父子200/800   —— small-to-big（子块召回→父块回答）
  3. 路由          —— 表格(xlsx/pdf)按行一块，长文档(docx/txt/md)用父子

ground-truth：答案数值在语料里唯一(1~3块)的干净题(16条)。
语料：干净题答案所在的 5 个文档 + 其所在文件夹里的全部文件（干扰项）。

用法：.venv\\Scripts\\python.exe stage6_routed_chunking.py
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
DOCS = BASE / "docs"
CHUNKS_FILE = BASE / "chunks.json"
EVAL_FILE = BASE / "评估集.json"
SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".xlsx"}
K = 10

print("加载模型 ...", flush=True)
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")
print("模型就绪", flush=True)

# ============ 解析 ============
def extract_lines(page, y_tol=3.0):
    def is_data(w):
        return bool(re.match(r'^[\d±—✓\-\.\(\)]', w))
    words = page.get_text("words")
    words.sort(key=lambda w: (round(w[1], 1), w[0]))
    raw_lines, cur, cur_y = [], [], None
    for w in words:
        y = w[1]
        if cur_y is None or abs(y - cur_y) > y_tol:
            if cur: raw_lines.append(cur)
            cur, cur_y = [w[4]], y
        else:
            cur.append(w[4])
    if cur: raw_lines.append(cur)
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

def load_document(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(str(path))
    elif ext in [".txt", ".md"]:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    elif ext == ".docx":
        import docx
        d = docx.Document(str(path))
        lines = [para.text for para in d.paragraphs if para.text.strip()]
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                lines.append(" | ".join(cells))
        return lines
    elif ext == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        lines = []
        for ws in wb.worksheets:
            lines.append(f"[工作表: {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(cells):
                    lines.append(" | ".join(cells))
        wb.close()
        return lines
    return []

# ============ 三种切分 ============
def split_flat(lines, size=400):
    chunks, cur, clen = [], [], 0
    for ln in lines:
        if cur and clen + len(ln) > size:
            chunks.append("\n".join(cur)); cur, clen = [], 0
        cur.append(ln); clen += len(ln)
    if cur: chunks.append("\n".join(cur))
    return chunks

def split_parent_child(lines, child=200, parent=800):
    parents, cur, clen = [], [], 0
    for ln in lines:
        if cur and clen + len(ln) > parent:
            parents.append("\n".join(cur)); cur, clen = [], 0
        cur.append(ln); clen += len(ln)
    if cur: parents.append("\n".join(cur))
    children, c2p = [], []
    for pi, p in enumerate(parents):
        cc, cclen = [], 0
        for ln in p.split("\n"):
            if cc and cclen + len(ln) > child:
                children.append("\n".join(cc)); c2p.append(pi); cc, cclen = [], 0
            cc.append(ln); cclen += len(ln)
        if cc: children.append("\n".join(cc)); c2p.append(pi)
    return children, c2p, parents

def chunk_rows(lines):
    """表格按行：一行一块，跳过工作表标记和空行"""
    chunks = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("[工作表:"):
            continue
        chunks.append(ln)
    return chunks

# ============ 索引/检索 ============
def build_faiss(texts):
    v = np.array(embed_model.encode(texts)).astype("float32")
    idx = faiss.IndexFlatL2(v.shape[1]); idx.add(v); return idx

def build_bm25(texts):
    return BM25Okapi([list(jieba.cut(t)) for t in texts])

def hybrid_search(q, fa, bm, n=20):
    qv = embed_model.encode([q]).astype("float32")
    _, vid = fa.search(qv, n); vid = vid[0].tolist()
    bid = np.argsort(bm.get_scores(list(jieba.cut(q))))[::-1][:n].tolist()
    rrf = {}
    for r, i in enumerate(vid): rrf[i] = rrf.get(i,0) + 1/(60+r)
    for r, i in enumerate(bid): rrf[i] = rrf.get(i,0) + 1/(60+r)
    return sorted(rrf, key=rrf.get, reverse=True)

def rerank_top(q, cands, texts, k):
    sc = reranker.predict([(q, texts[i]) for i in cands])
    r = sorted(zip(cands, sc), key=lambda x:x[1], reverse=True)
    return [i for i,s in r[:k]]

def norm(s):
    return (s.replace(" ","").replace("\u3000","").replace("\n","").replace("\r","")
             .replace("\t","").replace(",","").replace("，",""))

def answer_value(ans):
    nums = re.findall(r'\d+(?:\.\d+)?', ans)
    if not nums: return None
    return max(nums, key=lambda n:(len(n), '.' in n))

# ============ 1. 干净题 + 范围文档 ============
evals = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
full = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
full_norm = [norm(c["text"]) for c in full]
full_src = [c["source"] for c in full]

clean = []
for it in evals:
    if "无此信息" in it["a"] or it["a"].startswith("文档未"): continue
    v = answer_value(it["a"])
    if v is None: continue
    nv = norm(v)
    if len(nv) < 4 and "." not in v: continue
    idx = [i for i,nt in enumerate(full_norm) if nv in nt]
    if 1 <= len(idx) <= 3:
        clean.append((it, v))

golden_docs = set()
for it, v in clean:
    nv = norm(v)
    for i, nt in enumerate(full_norm):
        if nv in nt:
            golden_docs.add(full_src[i])

# 范围文档 = 黄金文档所在文件夹里的全部支持文件
golden_folders = set(Path(d).parent for d in golden_docs)
corpus_files = []
for f in sorted(DOCS.rglob("*")):
    if not f.is_file() or f.name.startswith("~$"): continue
    if f.suffix.lower() not in SUPPORTED: continue
    if f.relative_to(DOCS).parent in golden_folders:
        corpus_files.append(f)

print(f"干净题 {len(clean)} 条；黄金文档 {len(golden_docs)} 个；扩后语料 {len(corpus_files)} 个文件", flush=True)

# ============ 2. 解析 + 三种切分 ============
flat_chunks, pc_children, pc_c2p, pc_parents, row_chunks = [], [], [], [], []
for f in corpus_files:
    rel = str(f.relative_to(DOCS))
    try:
        lines = load_document(str(f))
    except Exception as e:
        print(f"  解析失败 {rel}: {type(e).__name__}", flush=True); continue
    flat_chunks.extend(split_flat(lines))
    c, c2p, p = split_parent_child(lines)
    base = len(pc_parents)
    pc_children.extend(c); pc_c2p.extend([base+x for x in c2p]); pc_parents.extend(p)
    # 路由：表格按行，长文档父子
    if f.suffix.lower() in (".xlsx", ".pdf"):
        row_chunks.extend(chunk_rows(lines))
    else:
        cc, cc2p, pp = split_parent_child(lines)
        # 路由里的长文档分支也用父子，这里把父块直接当"行块"用（简化：父块即检索单元）
        row_chunks.extend(pp)

print(f"单层 {len(flat_chunks)} 块 | 父子 子{len(pc_children)}/父{len(pc_parents)} | 路由 {len(row_chunks)} 块", flush=True)

# ============ 3. 建三种索引 ============
print("编码中 ...", flush=True)
fa_f = build_faiss(flat_chunks); bm_f = build_bm25(flat_chunks)
fa_c = build_faiss(pc_children); bm_c = build_bm25(pc_children)
fa_r = build_faiss(row_chunks); bm_r = build_bm25(row_chunks)

flat_norm = [norm(t) for t in flat_chunks]
pc_parent_norm = [norm(t) for t in pc_parents]
row_norm = [norm(t) for t in row_chunks]

# ============ 4. MRR ============
def rr_flat(q, v):
    nv = norm(v)
    cand = hybrid_search(q, fa_f, bm_f)[:20]
    for rank, i in enumerate(rerank_top(q, cand, flat_chunks, K), 1):
        if nv in flat_norm[i]: return 1.0/rank
    return 0.0

def rr_pc(q, v):
    nv = norm(v)
    cand = hybrid_search(q, fa_c, bm_c)[:20]
    seen, rp = set(), []
    for ci in rerank_top(q, cand, pc_children, K):
        pi = pc_c2p[ci]
        if pi not in seen:
            seen.add(pi); rp.append(pi)
    for rank, pi in enumerate(rp, 1):
        if nv in pc_parent_norm[pi]: return 1.0/rank
    return 0.0

def rr_row(q, v):
    nv = norm(v)
    cand = hybrid_search(q, fa_r, bm_r)[:20]
    for rank, i in enumerate(rerank_top(q, cand, row_chunks, K), 1):
        if nv in row_norm[i]: return 1.0/rank
    return 0.0

print(f"\n{'id':>3} {'类别':<3} {'答案值':<12} {'单层RR':>7} {'父子RR':>7} {'路由RR':>7}", flush=True)
r1, r2, r3 = [], [], []
for it, v in clean:
    q = it["q"]
    a, b, c = rr_flat(q,v), rr_pc(q,v), rr_row(q,v)
    r1.append(a); r2.append(b); r3.append(c)
    print(f"{it['id']:>3} {it['cat']:<3} {v:<12} {a:>7.4f} {b:>7.4f} {c:>7.4f}", flush=True)

m1 = sum(r1)/len(r1); m2 = sum(r2)/len(r2); m3 = sum(r3)/len(r3)
print(f"\n===== MRR@10 结果 =====", flush=True)
print(f"单层400   = {m1:.4f}", flush=True)
print(f"父子200/800 = {m2:.4f}  ({m2-m1:+.4f})", flush=True)
print(f"路由(按行)  = {m3:.4f}  ({m3-m1:+.4f})", flush=True)
print(f"命中题数: 单层 {sum(1 for x in r1 if x>0)}/{len(r1)}  父子 {sum(1 for x in r2 if x>0)}/{len(r2)}  路由 {sum(1 for x in r3 if x>0)}/{len(r3)}", flush=True)
