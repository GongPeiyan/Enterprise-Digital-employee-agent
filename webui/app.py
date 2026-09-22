# -*- coding: utf-8 -*-
"""数字员工 WebUI 后端：花名表 + 工号登录 + 记忆隔离 + RAG"""
import os, json, re, hashlib, secrets, sqlite3, shutil, threading, time, subprocess
from pathlib import Path
from collections import OrderedDict
from fastapi import FastAPI, Request, UploadFile, File
from starlette.concurrency import run_in_threadpool   # 把阻塞调用丢进线程，别堵事件循环
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse

os.environ.setdefault("HF_HOME",
                    "E:/hf_cache" if os.name == "nt" else str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 强制离线：否则每次加载模型都联网重试 31s+
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import pymupdf, faiss, numpy as np, jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import uvicorn
import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根，供 import retrieval / index_meta
import retrieval
import index_meta
from roles import ROLES, ROLE_ORDER, ROLE_BRIEF   # 角色卡片统一放 roles.py
from prompting import system_for, fmt_refs, build_rag_prompt   # 作答协议与拼法（生产/实验共用）
import acl_roster              # A3：部门/岗位级权限过滤（矩阵 + 员工表）
import acl_guard               # A5：回答后置兜底校验
import acl_register            # 注册姓名核对（花名表比对）
import prompting as _pr        # A5：统一拒答话术的唯一来源

# ================= 配置 =================
# 项目路径：默认取本文件所在目录（Windows 上就是 E:/数字员工项目/webui，行为不变）
# Mac/其他机器上不用改代码；要用别的目录时设环境变量 KB_BASE
BASE = Path(os.environ.get("KB_BASE", str(Path(__file__).resolve().parent)))
DATA = BASE / "data"
DB = DATA / "app.db"
INDEX_FILE = None      # 稍后由 index_meta 赋值
CHUNKS_FILE = None     # 稍后由 index_meta 赋值
DOCS_DIR = None        # 稍后由 index_meta 赋值
SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx", ".xlsx"}  # 向量库支持的文件格式

# ================= 读取项目根 .env（key 不写死在代码里） =================
def _load_env_file(path):
    """极简 .env 解析：KEY=VALUE，忽略注释/空行；已存在的环境变量优先（便于临时覆盖）。"""
    pth = Path(path)
    if not pth.exists():
        return
    for line in pth.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v

BASE_ROOT = Path(__file__).resolve().parent.parent          # 项目根 E:/数字员工项目
_load_env_file(BASE_ROOT / ".env")

# 知识库相关路径统一由 index_meta 提供（KB_DOCS_DIR / KB_INDEX_FILE / KB_CHUNKS_FILE 可覆盖）
DOCS_DIR = str(index_meta.docs_dir())
INDEX_FILE = str(index_meta.index_file())
CHUNKS_FILE = str(index_meta.chunks_file())

# 生产检索口径：进 prompt 的条数（2026-09-21 由 3 改为 5）
# 依据：生成层 A/B（50 题 × 3 次采样 = 150 条/组）答案正确率 0.927 → 0.940，
#       差异不显著（McNemar p=0.5，仅 1 道题受益），但无回退、拒答数不变、成本可忽略；
#       检索层 @5：命中率 0.92 → 0.96、MRR 0.8323 → 0.8747。评估口径同步到 @5。
MODEL = "deepseek-chat"
TOP_K = 5

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit(
        "❌ 未找到 DEEPSEEK_API_KEY。请在 " + str(BASE_ROOT / ".env") +
        " 里加一行：DEEPSEEK_API_KEY=sk-xxxxx（或设为系统环境变量后重启）"
    )

# 自动重建开关：KB_AUTO_REBUILD=0 关闭；KB_WATCH_INTERVAL 为指纹检查间隔（秒）
KB_AUTO_REBUILD = os.getenv("KB_AUTO_REBUILD", "1").lower() not in ("0", "false", "no")
AUTO_REBUILD_INTERVAL = int(os.getenv("KB_WATCH_INTERVAL", "60"))
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com",
                timeout=float(os.environ.get("KB_LLM_TIMEOUT", "60")), max_retries=1)   # 单次调用默认 60 秒超时：断流/超时要能报错，不能挂死占住线程
# 模型位置可配置：默认用 HF 缓存里的官方名；本地已有权重时用 KB_EMBED_MODEL / KB_RERANK_MODEL 指过去
embed_model = SentenceTransformer(os.getenv("KB_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))
reranker = CrossEncoder(os.getenv("KB_RERANK_MODEL", "BAAI/bge-reranker-base"))

# ================= RAG：文档解析（行优先 + 标签前置） =================
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
    doc = pymupdf.open(path)
    lines = []
    for page in doc:
        lines.extend(extract_lines(page))
    return lines

def split_chunks(lines, chunk_size=400):
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

# ================= RAG：索引持久化 =================
def build_index(texts):
    vectors = np.array(embed_model.encode(texts)).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def save_index(index, chunks, sources):
    with open(INDEX_FILE, "wb") as f:
        f.write(faiss.serialize_index(index).tobytes())
    data = [{"text": t, "source": s} for t, s in zip(chunks, sources)]
    json.dump(data, open(CHUNKS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def load_or_build():
    """递归扫描 docs/ 下所有支持格式的文档，自动发现新文件并入库。
    - 首次（无索引）：全量扫描重建
    - 之后：只处理「相对路径不在已存 source 里」的新文件，增量追加
    - 跳过 ~$ 开头的 Word/WPS 临时锁文件
    - source 用相对 docs/ 的路径，避免不同子目录同名文件冲突
    - 注意：同名文件内容更新暂不触发重处理，需改内容请改名或删索引重建
    """
    doc_dir = Path(DOCS_DIR)
    UNSUPPORTED = {".doc", ".wps", ".xls", ".ppt", ".pptx", ".csv"}
    files, unsupported = [], []
    if doc_dir.exists():
        for f in sorted(doc_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.name.startswith("~$"):
                continue  # Word/WPS 临时锁文件，跳过
            ext = f.suffix.lower()
            if ext in SUPPORTED_EXTS:
                files.append(f)
            elif ext in UNSUPPORTED:
                unsupported.append(f)
    if unsupported:
        print(f"⚠ 跳过 {len(unsupported)} 个暂不支持的格式（建议转 .docx/.xlsx）：")
        for f in unsupported[:15]:
            print(f"  - {f.relative_to(doc_dir)}")
        if len(unsupported) > 15:
            print(f"  ... 其余 {len(unsupported)-15} 个略")

    # 已有索引则加载，否则建空
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        with open(INDEX_FILE, "rb") as f:
            index = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
        data = json.load(open(CHUNKS_FILE, encoding="utf-8"))
        chunks = [d["text"] for d in data]
        sources = [d["source"] for d in data]
        existing = set(sources)
    else:
        index, chunks, sources = None, [], []
        existing = set()

    # 找出新文件（相对路径不在 source 里）
    new_files = [f for f in files if str(f.relative_to(doc_dir)) not in existing]
    if new_files:
        print(f"发现 {len(new_files)} 个新文档，入库中...")
        new_chunks, new_sources = [], []
        for f in new_files:
            rel = str(f.relative_to(doc_dir))
            try:
                lines = load_document(str(f))
                cs = split_chunks(lines)
                if not cs:
                    print(f"  ⚠ {rel}: 无内容，跳过")
                    continue
                new_chunks.extend(cs)
                new_sources.extend([rel] * len(cs))
                print(f"  {rel}: {len(cs)} 块")
            except Exception as e:
                print(f"  ⚠ {rel}: 读取失败跳过 ({type(e).__name__})")
        if not new_chunks:
            # 坑：纯表格/扫描件 docx 抽不出文本，这些文件每次启动都会被当成"新文件"，
            # 若不加这层判断，encode([]) 得到一维数组、index.add() 直接崩（2026-09-10 实测踩到）
            print("  ⚠ 这些新文件都抽不出可用文本（多为纯表格/扫描件），本次跳过入库")
        elif index is None:
            # 首次：全量建
            index = build_index(new_chunks)
            chunks, sources = new_chunks, new_sources
            save_index(index, chunks, sources)
            print(f"入库完成，知识库共 {len(chunks)} 块")
        else:
            # 增量：追加向量（flat 索引可安全 add）
            vectors = np.array(embed_model.encode(new_chunks)).astype("float32")
            if len(vectors.shape) != 2:      # 再加一层保险
                print("  ⚠ 向量维度异常，跳过入库")
            else:
                index.add(vectors)
                chunks.extend(new_chunks)
                sources.extend(new_sources)
                save_index(index, chunks, sources)
                print(f"入库完成，知识库共 {len(chunks)} 块")
    return index, chunks, sources

# ================= 文档加载（支持多格式）+ 动态加入知识库 =================
def load_document(path):
    """按文件类型加载文本，返回行列表"""
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
                cells = []
                prev = None
                for c in row.cells:
                    t = c.text.strip()
                    if t != prev:
                        cells.append(t)
                    prev = t
                if any(cells):
                    lines.append(" | ".join(cells))
        return lines
    elif ext == ".xlsx":
        # Excel：按工作表、按行读，单元格用 " | " 连接（行优先，避免列优先数字张冠李戴）
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

def add_to_knowledge_base(filepath, filename):
    """把上传的文档切块、向量化、加入知识库，返回加入的块数"""
    global bm25
    lines = load_document(filepath)
    if not lines:
        return 0
    new_chunks = split_chunks(lines)
    if not new_chunks:
        return 0
    vectors = np.array(embed_model.encode(new_chunks)).astype("float32")
    index.add(vectors)                       # 追加向量（index 是可变对象）
    chunks.extend(new_chunks)                # 追加文本块
    sources.extend([filename] * len(new_chunks))  # 追加来源标记
    save_index(index, chunks, sources)       # 持久化
    bm25 = BM25Okapi([list(jieba.cut(t)) for t in chunks])  # 重建 BM25
    return len(new_chunks)

# ================= 个人知识库（每个用户独立索引，懒加载缓存） =================
user_kbs = {}   # emp_id -> (faiss索引, chunks, sources, bm25)

def user_kb_dir(emp_id):
    d = DATA / "kb" / emp_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_user_kb(emp_id):
    """懒加载用户的个人知识库"""
    if emp_id in user_kbs:
        return user_kbs[emp_id]
    d = user_kb_dir(emp_id)
    idx_file = d / "faiss.index"
    chunks_file = d / "chunks.json"
    if idx_file.exists() and chunks_file.exists():
        with open(idx_file, "rb") as f:
            idx = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
        data = json.load(open(chunks_file, encoding="utf-8"))
        uchunks = [x["text"] for x in data]
        usources = [x["source"] for x in data]
    else:
        dim = index.d   # 从全局索引拿向量维度
        idx = faiss.IndexFlatL2(dim)
        uchunks, usources = [], []
    ubm25 = BM25Okapi([list(jieba.cut(t)) for t in uchunks]) if uchunks else None
    user_kbs[emp_id] = (idx, uchunks, usources, ubm25)
    return user_kbs[emp_id]

def add_to_user_kb(emp_id, filepath, filename):
    """把上传文档加入该用户的个人知识库"""
    idx, uchunks, usources, _ = load_user_kb(emp_id)
    lines = load_document(filepath)
    new_chunks = split_chunks(lines)
    if not new_chunks:
        return 0
    vectors = np.array(embed_model.encode(new_chunks)).astype("float32")
    idx.add(vectors)
    uchunks.extend(new_chunks)
    usources.extend([filename] * len(new_chunks))
    d = user_kb_dir(emp_id)
    with open(d / "faiss.index", "wb") as f:
        f.write(faiss.serialize_index(idx).tobytes())
    json.dump([{"text": t, "source": s} for t, s in zip(uchunks, usources)],
              open(d / "chunks.json", "w", encoding="utf-8"), ensure_ascii=False)
    user_kbs[emp_id] = (idx, uchunks, usources, BM25Okapi([list(jieba.cut(t)) for t in uchunks]))
    return len(new_chunks)

# ================= RAG：检索 =================
def retrieve_top_ids(query, faiss_idx, bm25, chunks, k=TOP_K):
    cand = retrieval.candidate_ids(query, bm25, faiss_idx, embed_model, n=20)
    scores = retrieval.rerank_scores(query, [chunks[i] for i in cand], reranker)
    ranked = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)
    return [i for i, s in ranked[:k]]

ACL_BLOCKED = {}      # emp_id → 累计被权限挡下的条数（A4 权限可见性测试统计用）

def _acl_note(emp_id, query, blocked):
    """记录一次"被挡下几条"，先只打日志；阶段 B 会接成正式的审计日志。"""
    ACL_BLOCKED[emp_id] = ACL_BLOCKED.get(emp_id, 0) + len(blocked)
    print("[acl] %s 问题「%s」被权限挡下 %d 条（类别：%s）"
          % (emp_id, query[:20], len(blocked), ",".join(sorted(set(blocked)))), flush=True)




ACL_FORBID = None      # 敏感类别（薪酬/结算/财务）的独有特征词，启动后算一次
def acl_forbid_terms():
    """从敏感类别的语料里抽出「不该出现在回答里」的特征词（金额数字、文件名）。"""
    global ACL_FORBID
    if ACL_FORBID is None:
        idx = [i for i in range(len(chunks))
               if acl_roster.cat_of_block(sources[i], chunks[i]) in acl_guard.SENSITIVE_CATS]
        ACL_FORBID = acl_guard.extract_forbidden([chunks[i] for i in idx],
                                                 [sources[i] for i in idx], max_terms=40)
        print("[acl] 兜底词表就绪：%d 条（来自 %d 个敏感块）" % (len(ACL_FORBID), len(idx)), flush=True)
    return ACL_FORBID


def retrieve_for_user(query, emp_id, k=TOP_K, info=None):
    """合并全局知识库 + 个人知识库检索，返回 [(text, source)]。

    A3 权限过滤：全局库的每个块先查它的类别（acl_tags_by_source.json），
    再看这个工号能不能看该类（acl_matrix.json × 员工表，由 acl_roster 判定）。
    个人知识库是员工自己上传的，不参与过滤。
    emp_id 为空时判为无权限（fail-closed），宁可少给也不能越权。
    """
    if emp_id and is_admin(emp_id):
        allowed = acl_roster.all_categories()     # 管理员（语料管理者）全库可见
    else:
        allowed = acl_roster.visible_categories(emp_id) if emp_id else set()
    if not emp_id:
        print("[acl] 警告：retrieve_for_user 未收到 emp_id，按无权限处理", flush=True)

    candidates, blocked = [], []
    g_ids = retrieval.candidate_ids(query, bm25, index, embed_model, n=15)
    for i in g_ids:
        if i >= len(chunks):
            continue
        cat = acl_roster.cat_of_block(sources[i], chunks[i])   # 先按块判（混载文件的例外），再回退文件级
        if cat in allowed:
            candidates.append((chunks[i], sources[i]))
        else:
            blocked.append(cat)                        # 挡下并计数（不把内容带出去）
    if blocked:
        _acl_note(emp_id, query, blocked)
    if info is not None:                      # B1：把"挡下了什么"记下来，便于区分权限问题/检索问题
        info["n_blocked"] = len(blocked)
        info["blocked_cats"] = [c for c in blocked if c]

    uidx, uchunks, usources, ubm25 = load_user_kb(emp_id)
    if len(uchunks) > 0:                              # 个人库：不过滤
        u_ids = retrieval.candidate_ids(query, ubm25, uidx, embed_model, n=15)
        for i in u_ids:
            if i < len(uchunks):
                candidates.append((uchunks[i], usources[i]))

    seen, dedup = set(), []
    for t, s in candidates:
        if t not in seen:
            seen.add(t)
            dedup.append((t, s))
    if not dedup:
        if info is not None:
            info["n_cand"] = 0
            info["sources"] = []
        return []
    texts = [t for t, s in dedup[:20]]
    scores = retrieval.rerank_scores(query, texts, reranker)
    ranked = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    result = [(dedup[i][0], dedup[i][1]) for i in ranked[:k]]
    if info is not None:
        info["n_cand"] = len(dedup)
        info["sources"] = [src for _t, src in result]
    return result


def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS roster(emp_id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS users(emp_id TEXT PRIMARY KEY, pwd_hash TEXT, salt TEXT)")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
    if "is_admin" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    if "disabled" not in cols:      # 停用标记：不删账号也能让人登不进来（可逆）
        conn.execute("ALTER TABLE users ADD COLUMN disabled INTEGER DEFAULT 0")
    conn.execute("CREATE TABLE IF NOT EXISTS qa_audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, emp_id TEXT, role TEXT, question TEXT, n_cand INTEGER, n_blocked INTEGER, blocked_cats TEXT, sources TEXT, tools TEXT, answer TEXT, refused INTEGER, guarded TEXT, latency_ms INTEGER, model TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT, topic TEXT, fact TEXT, ts TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT, role TEXT, content TEXT, ts TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS knowledge(id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, status TEXT, created_by TEXT, ts TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS skills(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, prompt TEXT, author TEXT, ts TEXT)")
    # 登录态持久化：否则服务一重启旧 token 全失效，浏览器却还留着 → 登录页与聊天页来回跳
    conn.execute("CREATE TABLE IF NOT EXISTS tokens(token TEXT PRIMARY KEY, emp_id TEXT, ts REAL)")
    # 初始化管理员 admin / admin123
    if not conn.execute("SELECT 1 FROM users WHERE emp_id='admin'").fetchone():
        salt = secrets.token_hex(16)
        conn.execute("INSERT INTO users(emp_id, pwd_hash, salt, is_admin) VALUES('admin',?,?,1)", (hash_pwd("admin123", salt), salt))
    else:
        conn.execute("UPDATE users SET is_admin=1 WHERE emp_id='admin'")
    conn.commit()
    conn.close()

def hash_pwd(password, salt):
    return hashlib.sha256((salt + password).encode()).hexdigest()

def is_admin(emp_id):
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT is_admin FROM users WHERE emp_id=?", (emp_id,)).fetchone()
    conn.close()
    return bool(row and row[0])

# ================= 角色（三个数字员工的人设） =================
# 角色卡片（system prompt / 侧栏顺序 / 一句话定位）统一在项目根的 roles.py，改角色只改那一处


def detect_role(message):
    """判断用户消息是否在召唤某个角色，是则返回角色名，否则 None"""
    names = [n for n in ROLE_ORDER if n != "管家"]     # 白名单直接来自 roles.py，加角色不用再改这里
    if not any(n in message for n in names):
        return None
    roster = "、".join(f"{n}（{ROLE_BRIEF[n]}）" for n in names)
    prompt = (
        f"判断用户这句话是否在召唤某个数字员工。员工：{roster}\n"
        f"如果用户在召唤某个员工（例如'妙妙在吗'、'叫码码来'、'安安帮我看看考勤'），只回复对应的名字（{'/'.join(names)}）。\n"
        "如果只是提到名字但不是召唤（例如'妙妙是谁'），回复 none。\n"
        f"用户的话：{message}\n你的判断："
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=10,
    )
    r = resp.choices[0].message.content.strip()
    for name in names:
        if name in r:
            return name
    return None

# ================= 多 agent 编排（管家调度） =================
def orchestrate_plan(task):
    """管家拆解任务，返回 [(专家名, 子任务描述)]"""
    experts = "、".join(k for k in ROLES if k != "管家")
    prompt = (
        "你是任务调度员。请把用户的任务拆解成几个子任务，分配给最合适的专家。\n"
        f"可用的专家：{experts}\n"
        "规则：每个子任务分配给最合适的专家；简单任务可能只需要 1 个专家。\n"
        "输出格式（每行一个子任务）：专家名|子任务描述\n"
        f"用户任务：{task}\n"
        "拆解结果："
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=500,
    )
    text = resp.choices[0].message.content.strip()
    plan = []
    for line in text.split("\n"):
        if "|" in line:
            expert, sub = line.split("|", 1)
            expert = expert.strip()
            sub = sub.strip()
            if expert in ROLES and expert != "管家" and sub:
                plan.append((expert, sub))
    return plan

def expert_run(role_prompt, sub_task, emp_id=None):
    """专家 agent 执行子任务，返回结果。

    2026-09-21 改：专家原先**完全不查资料库**（只凭模型自己的知识写东西），
    现在把公司知识库的检索结果一并注入，真正做到「每个角色都能用知识库」。
    传入 emp_id 时按该用户权限检索；检索失败自动降级为无资料作答，不阻断流程。
    """
    msgs = [{"role": "system", "content": role_prompt}]
    top = []
    top = []                      # A5：先兜底初始化，避免 emp_id 为空时变量未定义
    if emp_id:
        try:
            top = retrieve_for_user(sub_task, emp_id, k=TOP_K)
        except Exception as e:
            print(f"  ⚠ 专家检索失败，降级为无资料作答：{e}")
    if not top:
        # A5：没有任何"他有权看"的依据 → 按统一话术拒答，绝不无资料作答（否则就是编造）
        return _pr.REFUSAL
    msgs.append({"role": "user", "content": build_rag_prompt(sub_task, top)})
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=msgs,
        temperature=0.5, max_tokens=2000,
    )
    return resp.choices[0].message.content

def summarize(task, results):
    """管家汇总各专家结果"""
    parts = "\n\n".join(f"【{expert}】\n{r}" for expert, r in results)
    prompt = (
        f"你是任务调度员。用户的任务是：{task}\n\n"
        f"各专家完成的结果如下：\n{parts}\n\n"
        "请把这些结果整合成一份完整、连贯、专业的最终答复给用户。直接输出最终答复，不要提及'管家'或'专家'这些过程。"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5, max_tokens=3000,
    )
    return resp.choices[0].message.content

# ================= 记忆（按工号隔离 + 同主题覆盖） =================
def load_memories(emp_id):
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT fact FROM memories WHERE emp_id=? ORDER BY id", (emp_id,)).fetchall()
    conn.close()
    return [r[0] for r in rows]

def audit_qa(emp_id, role, q, info, tools, answer, refused, latency_ms, guarded=""):
    """B1 问答审计：每次问答写一条可倒查的记录（写失败只告警，绝不影响问答本身）。"""
    # 模型按协议自己说"没有依据"时也要算拒答（原实现只标了"无候选直接拒答"那条快路径）
    try:
        _refusal_head = _pr.REFUSAL[:12]
        if not refused and answer and _refusal_head and _refusal_head in answer:
            refused = 1
            guarded = (guarded + "、model-refusal").strip("、")
    except Exception:
        pass
    try:
        con = sqlite3.connect(DB)
        con.execute("INSERT INTO qa_audit(ts, emp_id, role, question, n_cand, n_blocked,"
                    " blocked_cats, sources, tools, answer, refused, guarded, latency_ms, model)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%d %H:%M:%S"), emp_id, role, (q or "")[:500],
                     info.get("n_cand"), info.get("n_blocked"),
                     "、".join(sorted(set(c for c in (info.get("blocked_cats") or []) if c)))[:200],
                     json.dumps(info.get("sources") or [], ensure_ascii=False)[:2000],
                     json.dumps(tools or [], ensure_ascii=False)[:600],
                     (answer or "")[:4000], 1 if refused else 0, guarded, latency_ms, MODEL))
        con.commit()
        con.close()
    except Exception as e:
        print("[audit] 写审计失败：%s" % e, flush=True)

def upsert_memory(emp_id, topic, fact):
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM memories WHERE emp_id=? AND topic=?", (emp_id, topic))
    conn.execute("INSERT INTO memories(emp_id, topic, fact, ts) VALUES(?,?,?,datetime('now'))", (emp_id, topic, fact))
    conn.commit()
    conn.close()

def extract_memory(user_msg, assistant_msg):
    prompt = (
        "从下面这段对话里，提取值得长期记住的、关于用户的事实。\n"
        "以下这些【一定要记，宁可多记不可漏记】：\n"
        "- 用户的个人信息：名字、职业、身份、年龄、生日、所在城市等\n"
        "- 用户的偏好和喜好：爱吃什么喝什么、习惯、讨厌什么（比如'我爱喝美式'）\n"
        "- 用户正在做的事、目标、重要决定\n"
        "输出格式：主题|事实（例如：咖啡偏好|爱喝美式不加糖）。每行一条。\n"
        "只有确实没出现任何关于用户的信息时，才回复：无\n"
        f"用户：{user_msg}\nAI：{assistant_msg}"
    )
    resp = client.chat.completions.create(model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=200)
    return resp.choices[0].message.content.strip()

# ================= 知识沉淀（提炼公共知识 + 向量化进全局库） =================
def extract_knowledge(user_msg, assistant_msg):
    prompt = (
        "判断下面这段对话里，有没有值得全公司共享的公共知识。\n"
        "公共知识 = 关于公司业务、流程、制度、系统的通用知识（比如'报销流程怎么走''某系统怎么登录'）。\n"
        "不是个人隐私（薪资、私人偏好、个人信息）。\n"
        "如果有，输出格式：问题|答案（问题提炼成标准问法，答案是标准答案）。\n"
        "如果没有公共知识，只回复：无\n"
        f"用户：{user_msg}\nAI：{assistant_msg}"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

def save_knowledge(question, answer, emp_id):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO knowledge(question, answer, status, created_by, ts) VALUES(?,?,?,?,datetime('now'))",
                 (question, answer, "pending", emp_id))
    conn.commit()
    conn.close()

def add_text_to_global(text, source):
    """把一段文本直接向量化加入全局知识库（RAG 可检索）"""
    global bm25
    vectors = np.array(embed_model.encode([text])).astype("float32")
    index.add(vectors)
    chunks.append(text)
    sources.append(source)
    save_index(index, chunks, sources)
    bm25 = BM25Okapi([list(jieba.cut(t)) for t in chunks])
    return 1

# ================= 对话历史持久化 =================
def save_message(emp_id, role, content):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO messages(emp_id, role, content, ts) VALUES(?,?,?,datetime('now'))", (emp_id, role, content))
    conn.commit()
    conn.close()

def load_history(emp_id):
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT role, content FROM messages WHERE emp_id=? ORDER BY id", (emp_id,)).fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]

# ================= 工具（Function Calling） =================
TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取当前日期和时间",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算数学表达式，例如 123*456 或 (1+2)*3",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "要计算的数学表达式"}
        }, "required": ["expression"]}
    }},
]

def safe_calc(expr):
    """安全数学求值：只允许数字与 + - * / // % ** 和括号。
    原实现用 eval()，提问者可以借它执行任意代码（服务器上什么都能干），必须换掉。"""
    import ast as _ast
    import operator as _op
    _ops = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul, _ast.Div: _op.truediv,
            _ast.FloorDiv: _op.floordiv, _ast.Mod: _op.mod, _ast.Pow: _op.pow,
            _ast.USub: _op.neg, _ast.UAdd: _op.pos}

    def _ev(n):
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, _ast.BinOp) and type(n.op) in _ops:
            left, right = _ev(n.left), _ev(n.right)
            if isinstance(n.op, _ast.Pow) and isinstance(right, (int, float)) and abs(right) > 100:
                raise ValueError("指数过大")
            v = _ops[type(n.op)](left, right)
            if isinstance(v, (int, float)) and abs(v) > 1e18:
                raise ValueError("结果过大")
            return v
        if isinstance(n, _ast.UnaryOp) and type(n.op) in _ops:
            return _ops[type(n.op)](_ev(n.operand))
        raise ValueError("只支持数字与 + - * / // %% ** 运算")

    return _ev(_ast.parse(expr if isinstance(expr, str) else str(expr), mode="eval").body)

def run_tool_loop(call_messages, max_rounds=3, force=None):
    """跑工具调用循环。force=指定工具名时强制调用它（算数/时间这类明确的请求，别指望模型自觉）。"""
    made = []
    for _round in range(max_rounds):
        _tc = ({"type": "function", "function": {"name": force}}
               if (force and not made) else "auto")
        resp = client.chat.completions.create(model="deepseek-chat", messages=call_messages,
                                              tools=TOOLS, max_tokens=200, tool_choice=_tc)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        call_messages.append({"role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = execute_tool(tc.function.name, args)
            made.append({"name": tc.function.name, "result": result})
            call_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return made

def execute_tool(name, args):
    if name == "get_current_time":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif name == "calculate":
        try:
            return str(safe_calc(args.get("expression", "")))
        except Exception as e:
            return f"计算错误: {e}"
    return "未知工具"

# ================= 会话（每工号独立） =================
TOKEN_TTL_DAYS = 30   # 登录态有效期（天）；这段时间内重启服务都不用重新登录
sessions = {}       # token -> emp_id（启动时从库恢复，跨重启有效）
# ─────────────────── 并发与记忆的闸门（C 阶段加固，可用环境变量覆盖）───────────────────
# 大模型吞吐实测约 0.75 答/秒：没有闸门时，30 人同时提问、最后一个人要静默等 40 秒。
# 所以这里不排队等，而是"超了就明确告诉你稍后再试"，等待时间永远可控。
CHAT_MAX_INFLIGHT = int(os.environ.get("KB_CHAT_MAX_INFLIGHT", "8"))    # 同时最多处理几个问答
CHAT_WAIT_S = float(os.environ.get("KB_CHAT_WAIT", "5"))               # 满员时最多排队等几秒（有界队列，不无限等）
CHAT_KEEP_ROUNDS = int(os.environ.get("KB_CHAT_KEEP_ROUNDS", "10"))     # 每个会话在内存里保留最近几轮（一轮=一问一答）
CHAT_MAX_CONV = int(os.environ.get("KB_CHAT_MAX_CONV", "500"))          # 会话缓存最多几个键，超出淘汰最久未用
CHAT_SLOTS = threading.Semaphore(CHAT_MAX_INFLIGHT)
CHAT_STAT = {"inflight": 0, "peak": 0, "shed": 0, "llm_failed": 0}

conversations = OrderedDict()  # "工号:角色" -> messages 列表（有序：最近用过的排最后，便于淘汰）


def remember(conv_key, messages):
    """收尾：就地裁剪会话 + LRU 淘汰。就地裁剪是关键——换新列表会让调用方的引用失效。"""
    keep = CHAT_KEEP_ROUNDS * 2
    if len(messages) > keep + 1:                 # +1 是第一条 system（角色设定+记忆）
        del messages[1:len(messages) - keep]     # 只留 system + 最近 keep 条
    conversations.move_to_end(conv_key)
    while len(conversations) > CHAT_MAX_CONV:
        conversations.popitem(last=False)        # 淘汰最久未用的会话


def acquire_slot():
    """取一个并发名额：满了就排队等 CHAT_WAIT_S 秒（有界），等不到返回 False。"""
    if CHAT_SLOTS.acquire(timeout=CHAT_WAIT_S):
        CHAT_STAT["inflight"] += 1
        CHAT_STAT["peak"] = max(CHAT_STAT["peak"], CHAT_STAT["inflight"])
        return True
    CHAT_STAT["shed"] += 1
    return False


def release_slot():
    CHAT_STAT["inflight"] = max(0, CHAT_STAT["inflight"] - 1)
    CHAT_SLOTS.release()


def busy_message():
    return ("现在提问题的人有点多（前面还有 %d 个问题正在处理），请过一会儿再问一次。"
            % CHAT_STAT["inflight"])


def busy_response(emp_id, role, q, t0):
    """满员时的立刻回复：不走检索、不调大模型，用户马上看到原因。"""
    CHAT_STAT["shed"] += 1
    msg = busy_message()
    def _gen():
        audit_qa(emp_id, role, q, {}, [], msg, 1, int((time.time() - t0) * 1000), guarded="busy-shed")
        print("[busy] %s 被闸门挡下（当前在跑 %d 个）" % (emp_id, CHAT_STAT["inflight"]), flush=True)
        yield f"data: {json.dumps({'delta': msg, 'busy': True})}\n\n"
        yield f"data: {json.dumps({'done': True, 'role': role})}\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream")


def guarded_stream(gen_factory, emp_id, role, q, info, t0):
    """把回答包在名额生命周期里：名额在进函数时取好，流结束（或异常）时释放。"""
    def _gen():
        try:
            yield from gen_factory()
        finally:
            release_slot()
    return StreamingResponse(_gen(), media_type="text/event-stream")



def save_token(token, emp_id):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT OR REPLACE INTO tokens(token, emp_id, ts) VALUES(?,?,?)", (token, emp_id, time.time()))
    conn.commit()
    conn.close()


def drop_token(token):
    sessions.pop(token, None)
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM tokens WHERE token=?", (token,))
    conn.commit()
    conn.close()


def load_sessions():
    """启动时把库里未过期的登录态读回内存，避免重启后旧 token 全失效"""
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM tokens WHERE ts < ?", (time.time() - TOKEN_TTL_DAYS * 86400,))
    for tok, emp in conn.execute("SELECT token, emp_id FROM tokens"):
        sessions[tok] = emp
    conn.commit()
    conn.close()
    return len(sessions)

# ================= FastAPI =================
app = FastAPI()

@app.post("/api/register")
async def register(req: Request):
    d = await req.json()
    emp_id = d.get("emp_id", "").strip()
    password = d.get("password", "")
    if not emp_id or not password:
        return JSONResponse({"ok": False, "msg": "工号和密码不能为空"})
    if len(password) < 8:
        return JSONResponse({"ok": False, "msg": "密码至少 8 位"})
    conn = sqlite3.connect(DB)
    name = (d.get("name") or "").strip()
    in_roster = conn.execute("SELECT name FROM roster WHERE emp_id=?", (emp_id,)).fetchone()
    if not in_roster:
        conn.close()
        return JSONResponse({"ok": False, "msg": "工号不在花名表，无法注册（请联系管理员录入）"})
    # 姓名核对：注册只校验工号的话，谁都能用别人的工号抢先注册，从而拿到那个工号的权限
    if not acl_register.name_matches(in_roster[0], name):
        conn.close()
        return JSONResponse({"ok": False, "msg": "姓名与花名表不一致，请核对后重试；确属本人请联系管理员"})
    exists = conn.execute("SELECT 1 FROM users WHERE emp_id=?", (emp_id,)).fetchone()
    if exists:
        conn.close()
        return JSONResponse({"ok": False, "msg": "该工号已开户，请用管理员发放的初始密码直接登录；忘记密码请联系管理员重置"})
    salt = secrets.token_hex(16)
    conn.execute("INSERT INTO users(emp_id, pwd_hash, salt) VALUES(?,?,?)", (emp_id, hash_pwd(password, salt), salt))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "msg": "注册成功"})

@app.post("/api/login")
async def login(req: Request):
    d = await req.json()
    emp_id = d.get("emp_id", "").strip()
    password = d.get("password", "")
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT pwd_hash, salt, is_admin, disabled FROM users WHERE emp_id=?", (emp_id,)).fetchone()
    conn.close()
    if row and len(row) > 3 and row[3]:      # 已停用：先挡住，不透露密码对错
        return JSONResponse({"ok": False, "msg": "该账号已停用，请联系管理员"})
    if row and row[0] == hash_pwd(password, row[1]):
        token = secrets.token_hex(16)
        sessions[token] = emp_id
        save_token(token, emp_id)   # 落库，重启服务后依然有效
        return JSONResponse({"ok": True, "token": token, "is_admin": bool(row[2] if len(row) > 2 else 0)})
    return JSONResponse({"ok": False, "msg": "工号或密码错误"})

@app.get("/api/me")
async def me(req: Request):
    """校验登录态是否仍有效。登录页进站前先问一次，凭这个决定去聊天页还是留在登录页，
    避免拿着失效 token 直接跳 /chat 又被 401 弹回来，造成两页来回闪。"""
    token = req.headers.get("Authorization", "").replace("Bearer ", "")
    emp_id = sessions.get(token)
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    return JSONResponse({"ok": True, "emp_id": emp_id, "is_admin": is_admin(emp_id)})

@app.post("/api/logout")
async def logout_api(req: Request):
    """退出登录：把 token 从内存和库里一起删掉"""
    drop_token(req.headers.get("Authorization", "").replace("Bearer ", ""))
    return JSONResponse({"ok": True})

@app.post("/api/password")
async def change_password(req: Request):
    """修改密码：已登录用 token 认人；未登录（登录页）则需提供 工号 + 原密码。"""
    d = await req.json()
    token = req.headers.get("Authorization", "").replace("Bearer ", "")
    emp_id = sessions.get(token) or (d.get("emp_id") or "").strip()
    old = d.get("old_password", "")
    new = d.get("new_password", "")
    if not emp_id or not old or not new:
        return JSONResponse({"ok": False, "msg": "工号、原密码、新密码都不能为空"})
    if len(new) < 8:
        return JSONResponse({"ok": False, "msg": "新密码至少 8 位"})
    if new == old:
        return JSONResponse({"ok": False, "msg": "新密码不能与原密码相同"})
    con = sqlite3.connect(DB)
    row = con.execute("SELECT pwd_hash, salt FROM users WHERE emp_id=?", (emp_id,)).fetchone()
    if not row or row[0] != hash_pwd(old, row[1]):
        con.close()
        return JSONResponse({"ok": False, "msg": "工号或原密码不正确"})
    salt = secrets.token_hex(16)
    con.execute("UPDATE users SET pwd_hash=?, salt=? WHERE emp_id=?",
                (hash_pwd(new, salt), salt, emp_id))
    con.commit()
    con.close()
    return JSONResponse({"ok": True, "msg": "密码已修改，下次登录请用新密码"})

@app.post("/api/chat")
async def chat(req: Request):
    token = req.headers.get("Authorization", "").replace("Bearer ", "")
    emp_id = sessions.get(token)
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    d = await req.json()
    q = d.get("message", "").strip()
    role = d.get("role", "lucky")
    # 自然语言召唤：用户说"lucky在吗"这类，自动切换到对应角色
    summoned = detect_role(q)
    if summoned:
        role = summoned
    if not q:
        return JSONResponse({"ok": False, "msg": "消息为空"})
    # 并发名额：满了先排队等几秒（有界），仍拿不到就明确回复，不做检索也不调模型
    if not await run_in_threadpool(acquire_slot):     # 排队等待放到线程里，别堵事件循环
        return busy_response(emp_id, role, q, time.time())
    try:
        return await _answer_with_slot(emp_id, role, q, time.time())
    except Exception:
        release_slot()
        raise


async def _answer_with_slot(emp_id, role, q, _t0):
    """真正干活的部分：检索 → 工具 → 流式回答（名额已由调用方取好）。"""
    _t0 = time.time()          # 记账起点（管家分支的生成器也要用到，必须放在最前面）
    _info = {}
    role_prompt = system_for(ROLES.get(role, "你是数字员工"))   # 每个角色都带「知识库作答协议」
    conv_key = f"{emp_id}:{role}"
    if conv_key not in conversations:
        mems = load_memories(emp_id)
        mt = "\n".join(f"- {m}" for m in mems) if mems else "（暂无）"
        history = load_history(emp_id)
        conversations[conv_key] = [{"role": "system", "content": f"{role_prompt}\n你记得关于这个用户（工号 {emp_id}）的这些事：\n{mt}"}] + history
    messages = conversations[conv_key]
    # 工具类问题提前判定：算数/问时间这类不依赖知识库，别让"没资料"把它拦成拒答
    import re as _re
    _force_tool = None
    if _re.search(r"\d", q) and _re.search(r"[+\-*/×÷]|等于|计算|算一下|结果是多少", q) and "结算" not in q:
        _force_tool = "calculate"
    elif _re.search(r"今天|现在|当前|日期|时间|星期|几号", q):
        _force_tool = "get_current_time"

    # 管家：走多 agent 编排流程
    if role == "管家":
        def generate():
            yield f"data: {json.dumps({'orchestrator': '正在拆解任务...'})}\n\n"
            plan = orchestrate_plan(q)
            if not plan:
                yield f"data: {json.dumps({'delta': '抱歉，我没能拆解这个任务，请换个说法。'})}\n\n"
                yield f"data: {json.dumps({'done': True, 'role': role})}\n\n"
                return
            results = []
            for expert, sub in plan:
                yield f"data: {json.dumps({'expert': expert, 'sub': sub})}\n\n"
                r = expert_run(system_for(ROLES[expert]), sub, emp_id=emp_id)
                results.append((expert, r))
            yield f"data: {json.dumps({'orchestrator': '正在汇总结果...'})}\n\n"
            final = summarize(q, results)
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": final})
            save_message(emp_id, "user", q)
            save_message(emp_id, "assistant", final)
            audit_qa(emp_id, role, q, _info, [], final, 0,
                     int((time.time() - _t0) * 1000), guarded="orchestrator")
            fact = extract_memory(q, final)
            if fact and fact != "无":
                for line in fact.split("\n"):
                    if "|" in line:
                        topic, val = line.split("|", 1)
                        upsert_memory(emp_id, topic.strip(), val.strip())
            k = extract_knowledge(q, final)
            if k and k != "无" and "|" in k:
                kq, ka = k.split("|", 1)
                save_knowledge(kq.strip(), ka.strip(), emp_id)
            remember(conv_key, messages)          # 管家分支：裁剪 + LRU
            yield f"data: {json.dumps({'delta': final})}\n\n"
            yield f"data: {json.dumps({'done': True, 'role': role})}\n\n"
        return guarded_stream(generate, emp_id, role, q, _info, _t0)
    _t0 = time.time()
    _info = {}
    top = await run_in_threadpool(retrieve_for_user, q, emp_id, TOP_K, _info)   # 检索是 CPU 密集，别占着事件循环
    if not top:
        # A5：他的话里没有任何可访问依据 → 直接按统一话术拒答，不调模型（也就无从编造）
        def _refuse():
            yield f"data: {json.dumps({'delta': _pr.REFUSAL})}\n\n"
            yield f"data: {json.dumps({'done': True, 'role': role})}\n\n"
        # 先看是不是"工具类问题"（问时间、算数等不依赖知识库）：这类问题没有资料也该答
        _call_msgs = [{"role": "system", "content":
            "你只能处理两类请求："
            "① 问当前日期/时间 → 必须调用 get_current_time；"
            "② 数学计算 → 必须调用 calculate（参数是纯算式）。"
            "拿到工具结果后，用一句中文把结果说清楚。"
            "任何其它请求：只回复「没有依据」，不要尝试回答。"},
                      {"role": "user", "content": q}]
        try:
            _made = await run_in_threadpool(run_tool_loop, _call_msgs, 2, _force_tool)
        except Exception as _e:
            _made = []
            print("[tool] 工具循环失败：%s" % _e, flush=True)
        if _made:
            def _gen_tool():
                for tc in _made:
                    yield f"data: {json.dumps({'tool': tc['name'], 'result': tc['result']})}\n\n"
                full = ""
                try:
                    stream = client.chat.completions.create(model="deepseek-chat", messages=_call_msgs,
                                                            temperature=0.3, max_tokens=1024, stream=True)
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            full += delta
                            yield f"data: {json.dumps({'delta': delta})}\n\n"
                except Exception as _e:
                    full = "（工具结果：%s）" % _made[0]["result"]
                    yield f"data: {json.dumps({'delta': full})}\n\n"
                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": full})
                save_message(emp_id, "user", q)
                save_message(emp_id, "assistant", full)
                audit_qa(emp_id, role, q, _info, _made, full, 0,
                         int((time.time() - _t0) * 1000), guarded="tool-only")
                remember(conv_key, messages)      # 工具分支：裁剪 + LRU
                yield f"data: {json.dumps({'done': True, 'sources': [], 'role': role})}\n\n"
            print("[tool] %s 无资料但走了工具：%s" % (emp_id, [t["name"] for t in _made]), flush=True)
            return guarded_stream(_gen_tool, emp_id, role, q, _info, _t0)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": _pr.REFUSAL})
        save_message(emp_id, "user", q)
        save_message(emp_id, "assistant", _pr.REFUSAL)
        audit_qa(emp_id, role, q, _info, [], _pr.REFUSAL, 1,
                 int((time.time() - _t0) * 1000), guarded="refusal-fastpath")
        remember(conv_key, messages)          # 拒答分支：裁剪 + LRU
        print("[acl] %s 无可用依据 → 直接拒答" % emp_id, flush=True)
        return guarded_stream(_refuse, emp_id, role, q, _info, _t0)
    context = fmt_refs(top)        # 【资料N｜项目/文件名】+ 正文（编号便于引用，短来源避免同名文件混淆）
    call_messages = messages + [{"role": "user", "content": build_rag_prompt(q, top)}]
    _ = context                      # 保留变量便于调试打印

    # 工具循环（抽成函数，与"无检索结果但可能是工具类问题"共用）
    tool_calls_made = await run_in_threadpool(run_tool_loop, call_messages, 3, _force_tool)
    if tool_calls_made:
        call_messages.append({"role": "system", "content":
            "本次已用工具得到权威结果，请直接用中文把结果说清楚，不要以资料没有依据为由拒答。"})

    def generate():
        for tc in tool_calls_made:
            yield f"data: {json.dumps({'tool': tc['name'], 'result': tc['result']})}\n\n"
        full = ""
        try:
            stream = client.chat.completions.create(model="deepseek-chat", messages=call_messages, temperature=0.3, max_tokens=4096, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full += delta
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
        except Exception as _e:                    # 超时/断流：明确告知，别让用户对着转圈等
            CHAT_STAT["llm_failed"] += 1
            print("[llm] %s 生成失败：%s: %s" % (emp_id, type(_e).__name__, _e), flush=True)
            _tip = ("抱歉，这次回答超时或中断了（%s）。可以把问题问得更具体一些，或者稍后再点一次。"
                    % type(_e).__name__) if not full else (chr(10) + "（上一条回答中断于此处：%s）" % type(_e).__name__)
            full += _tip
            yield f"data: {json.dumps({'delta': _tip, 'error': 'llm-failed'})}\n\n"
        # A5 后置兜底：检查回答里有没有"引用受限内容"或"元信息泄露"（流式下无法撤回，追加更正）
        _ok, _flags = acl_guard.judge(full, [t for t, _s in top], acl_forbid_terms(),
                                    extra_ok=[str(x.get("result", "")) for x in tool_calls_made])
        if _flags:
            print("[acl-guard] %s 回答命中：%s" % (emp_id, _flags), flush=True)
        if any(f.startswith(("引用受限", "元信息")) for f in _flags):
            yield f"data: {json.dumps({'delta': chr(10) + '（更正：上一条回答超出你可访问的资料范围，已撤回。）' + _pr.REFUSAL})}\n\n"

        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": full})
        save_message(emp_id, "user", q)
        save_message(emp_id, "assistant", full)
        fact = extract_memory(q, full)
        if fact and fact != "无":
            for line in fact.split("\n"):
                if "|" in line:
                    topic, val = line.split("|", 1)
                    upsert_memory(emp_id, topic.strip(), val.strip())
        k = extract_knowledge(q, full)
        if k and k != "无" and "|" in k:
            kq, ka = k.split("|", 1)
            save_knowledge(kq.strip(), ka.strip(), emp_id)
        doc_content = full
        if "【成品】" in full:
            doc_content = full.split("【成品】", 1)[1].strip()
        doc_name = None
        if len(doc_content) > 200:
            try:
                doc_name = generate_docx(emp_id, q[:30] or "文档", doc_content)
            except Exception:
                doc_name = None
        audit_qa(emp_id, role, q, _info, tool_calls_made, full, 0,
                 int((time.time() - _t0) * 1000), guarded="、".join(_flags))
        remember(conv_key, messages)          # 主分支：裁剪 + LRU
        yield f"data: {json.dumps({'done': True, 'sources': [s for t, s in top], 'doc': doc_name, 'role': role})}\n\n"

    return guarded_stream(generate, emp_id, role, q, _info, _t0)

@app.get("/api/history")
async def history(req: Request):
    token = req.headers.get("Authorization", "").replace("Bearer ", "")
    emp_id = sessions.get(token)
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    return JSONResponse({"ok": True, "history": load_history(emp_id)})

# ================= 文件上传/下载（按用户文件夹隔离） =================
FILES_DIR = DATA / "files"

def user_files_dir(emp_id):
    d = FILES_DIR / emp_id
    d.mkdir(parents=True, exist_ok=True)
    return d

@app.post("/api/upload")
async def upload(req: Request, file: UploadFile = File(...)):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    safe_name = Path(file.filename).name   # 防路径穿越
    with open(user_files_dir(emp_id) / safe_name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # 如果是文档，切块+向量化加入知识库，这样 RAG 能检索到
    fpath = str(user_files_dir(emp_id) / safe_name)
    if is_admin(emp_id):
        kb_added = add_to_knowledge_base(fpath, safe_name)   # 管理员→全局共享
    else:
        kb_added = add_to_user_kb(emp_id, fpath, safe_name)  # 普通用户→个人私有
    return JSONResponse({"ok": True, "filename": safe_name, "kb_added": kb_added})

@app.get("/api/files")
async def list_files(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    d = user_files_dir(emp_id)
    files = [{"name": p.name, "size": p.stat().st_size} for p in sorted(d.iterdir()) if p.is_file()]
    return JSONResponse({"ok": True, "files": files})

def generate_docx(emp_id, title, content):
    import docx as docx_mod
    doc = docx_mod.Document()
    doc.add_heading(title, 0)
    for para in content.split("\n"):
        if para.strip():
            doc.add_paragraph(para)
    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
    filename = f"{safe_title}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.docx"
    doc.save(str(user_files_dir(emp_id) / filename))
    return filename

@app.post("/api/save_doc")
async def save_doc(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    d = await req.json()
    title = (d.get("title", "文档") or "文档").strip()
    content = d.get("content", "")
    filename = generate_docx(emp_id, title, content)
    return JSONResponse({"ok": True, "filename": filename})

@app.get("/api/download/{filename}")
async def download(filename: str, token: str = ""):
    emp_id = sessions.get(token)
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    safe_name = Path(filename).name
    filepath = user_files_dir(emp_id) / safe_name
    if not filepath.exists():
        return JSONResponse({"ok": False, "msg": "文件不存在"}, status_code=404)
    return FileResponse(str(filepath), filename=safe_name)

# ================= 花名表管理（仅管理员） =================
@app.post("/api/roster/add")
async def roster_add(req: Request):
    """把员工加入花名表（在册名单）。部门/职位决定他能看到哪些资料，务必填对。"""
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    rid = (d.get("emp_id") or "").strip()
    nm = (d.get("name") or "").strip()
    dept = (d.get("dept") or "").strip()
    title = (d.get("title") or "").strip()
    leader = 1 if d.get("is_leader") else 0
    if not rid or not nm:
        return JSONResponse({"ok": False, "msg": "工号和姓名不能为空"})
    conn = sqlite3.connect(DB)
    conn.execute("INSERT OR REPLACE INTO roster(emp_id, name, dept, title, is_leader)"
                 " VALUES(?,?,?,?,?)", (rid, nm, dept, title, leader))
    conn.commit()
    conn.close()
    print("[admin] %s 加入名册 %s %s（%s/%s）" % (emp_id, rid, nm, dept, title), flush=True)
    return JSONResponse({"ok": True, "msg": "已加入名册。别忘了点「重算权限」"})

@app.post("/api/roster/update")
async def roster_update(req: Request):
    """修改员工的组织属性（转部门 / 升职 / 改职位）。权限按岗位算，重算后生效。"""
    me = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not me or not is_admin(me):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    rid = (d.get("emp_id") or "").strip()
    if not rid:
        return JSONResponse({"ok": False, "msg": "请指定工号"})
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT name, dept, title, is_leader FROM roster WHERE emp_id=?", (rid,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"ok": False, "msg": "花名表里没有这个工号"})
    dept = (d.get("dept") if d.get("dept") is not None else row[1]) or ""
    title = (d.get("title") if d.get("title") is not None else row[2]) or ""
    leader = (1 if d.get("is_leader") else 0) if d.get("is_leader") is not None else int(row[3] or 0)
    conn.execute("UPDATE roster SET dept=?, title=?, is_leader=? WHERE emp_id=?",
                 (dept.strip(), title.strip(), leader, rid))
    conn.commit()
    conn.close()
    print("[admin] %s 改了 %s 的组织属性 → %s/%s/%s" % (me, rid, dept, title, leader), flush=True)
    return JSONResponse({"ok": True, "msg": "已保存。别忘了点「重算权限」"})

@app.post("/api/roster/remove")
async def roster_remove(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    rid = d.get("emp_id", "").strip()
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM roster WHERE emp_id=?", (rid,))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True})

@app.post("/api/admin/recalc_acl")
async def admin_recalc_acl(req: Request):
    """按当前花名表重算权限（生成 acl_roster.json）。权限加载带 mtime 缓存，重算后立即生效。"""
    me = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not me or not is_admin(me):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    script = BASE_ROOT / "acl_roster.py"
    try:
        p = subprocess.run([sys.executable, str(script)], cwd=str(BASE_ROOT),
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        return JSONResponse({"ok": False, "msg": "重算失败：%s" % e})
    if p.returncode != 0:
        return JSONResponse({"ok": False, "msg": "重算失败：%s" % (p.stderr or p.stdout)[-400:]})
    lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
    head = lines[:1] + lines[2:8]
    print("[admin] %s 重算了权限：%s" % (me, lines[0] if lines else ""), flush=True)
    return JSONResponse({"ok": True, "msg": "权限已重算并生效",
                         "summary": chr(10).join(head)})

@app.post("/api/admin/toggle_account")
async def admin_toggle_account(req: Request):
    """管理员停用/启用账号。停用是标记+踢下线，可逆；管理员账号不允许被停用。"""
    me = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not me or not is_admin(me):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    eid = (d.get("emp_id") or "").strip()
    dis = 1 if d.get("disabled") else 0
    if not eid:
        return JSONResponse({"ok": False, "msg": "请指定工号"})
    con = sqlite3.connect(DB)
    row = con.execute("SELECT is_admin FROM users WHERE emp_id=?", (eid,)).fetchone()
    if not row:
        con.close()
        return JSONResponse({"ok": False, "msg": "该工号还没有账号"})
    if row[0]:
        con.close()
        return JSONResponse({"ok": False, "msg": "管理员账号不允许停用"})
    con.execute("UPDATE users SET disabled=? WHERE emp_id=?", (dis, eid))
    if dis:
        con.execute("DELETE FROM tokens WHERE emp_id=?", (eid,))     # 立即踢下线
    con.commit()
    con.close()
    if dis:
        for t in [k for k, v in list(sessions.items()) if v == eid]:
            sessions.pop(t, None)
    print("[admin] %s %s 了 %s 的账号" % (me, "停用" if dis else "启用", eid), flush=True)
    return JSONResponse({"ok": True, "emp_id": eid, "disabled": bool(dis),
                         "msg": "已停用" if dis else "已启用"})

@app.post("/api/admin/reset_password")
async def admin_reset_password(req: Request):
    """管理员重置员工密码：随机 10 位新密码（也可指定），并踢掉该员工已登录的会话。"""
    me = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not me or not is_admin(me):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    eid = (d.get("emp_id") or "").strip()
    new = (d.get("new_password") or "").strip()
    if not eid:
        return JSONResponse({"ok": False, "msg": "请指定工号"})
    if new and len(new) < 8:
        return JSONResponse({"ok": False, "msg": "新密码至少 8 位"})
    con = sqlite3.connect(DB)
    if not con.execute("SELECT 1 FROM users WHERE emp_id=?", (eid,)).fetchone():
        con.close()
        return JSONResponse({"ok": False, "msg": "该工号还没有账号（尚未注册/开户）"})
    if not new:
        CHARSET = "ACDEFGHJKLMNPQRTUVWXY34679acdefghjkmnpqrtuvwxy"   # 去掉易混字符
        new = "".join(secrets.choice(CHARSET) for _ in range(10))
    salt = secrets.token_hex(16)
    con.execute("UPDATE users SET pwd_hash=?, salt=? WHERE emp_id=?",
                (hash_pwd(new, salt), salt, eid))
    con.execute("DELETE FROM tokens WHERE emp_id=?", (eid,))        # 库里的登录态失效
    con.commit()
    con.close()
    for t in [k for k, v in list(sessions.items()) if v == eid]:    # 内存里的会话也清掉
        sessions.pop(t, None)
    print("[admin] %s 重置了 %s 的密码" % (me, eid), flush=True)
    return JSONResponse({"ok": True, "emp_id": eid, "new_password": new,
                         "msg": "已重置，请把新密码转交本人（只显示这一次）"})

@app.get("/api/admin/audit")
async def admin_audit(req: Request):
    """问答审计：按时间倒序返回记录（可按工号过滤、只看拒答）。"""
    me = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not me or not is_admin(me):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    emp = (req.query_params.get("emp_id") or "").strip()
    only_refused = req.query_params.get("refused") == "1"
    limit = min(int(req.query_params.get("limit", 50) or 50), 500)
    sql = ("SELECT id, ts, emp_id, role, question, n_cand, n_blocked, blocked_cats, sources,"
           " tools, answer, refused, guarded, latency_ms FROM qa_audit WHERE 1=1")
    args = []
    if emp:
        sql += " AND emp_id=?"; args.append(emp)
    if only_refused:
        sql += " AND refused=1"
    if req.query_params.get("days"):
        dnum = int(req.query_params.get("days"))
        sql += " AND ts>=?"
        args.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - dnum * 86400)))
    sql += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    con = sqlite3.connect(DB)
    rows = con.execute(sql, args).fetchall()
    con.close()
    keys = ["id", "ts", "emp_id", "role", "question", "n_cand", "n_blocked", "blocked_cats",
            "sources", "tools", "answer", "refused", "guarded", "latency_ms"]
    return JSONResponse({"ok": True, "rows": [dict(zip(keys, r)) for r in rows]})

_BOOT_TS = time.time()

@app.get("/api/admin/runtime")
def api_admin_runtime(request: Request):
    """运行时体检：会话缓存、线程、进程 RSS、在线登录态。压测和上线运维都用它。"""
    me = sessions.get(request.headers.get("Authorization", "").replace("Bearer ", ""))
    if not me or not is_admin(me):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    try:
        import resource, threading
        rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)   # macOS: 字节
        return {
            "conversations": len(conversations),        # 进程内对话缓存条目（设计上按工号+角色隔离）
            "messages_in_memory": sum(len(v) for v in conversations.values()),
            "threads": threading.active_count(),
            "rss_mb": rss_mb,
            "uptime_s": int(time.time() - _BOOT_TS),
            "sessions": len(sessions),      # token -> emp_id（在线登录态）
            "chat_inflight": CHAT_STAT["inflight"],        # 正在处理的问答数
            "chat_peak": CHAT_STAT["peak"],                # 历史峰值并发
            "chat_shed": CHAT_STAT["shed"],                # 被闸门挡下的请求数（给了明确提示）
            "chat_llm_failed": CHAT_STAT["llm_failed"],    # 大模型超时/断流次数
            "chat_max_inflight": CHAT_MAX_INFLIGHT,         # 当前闸门上限
            "chat_keep_rounds": CHAT_KEEP_ROUNDS,           # 每会话保留轮数
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/admin/audit_stats")
async def admin_audit_stats(req: Request):
    """审计概览：总量、拒答率、延迟分位、被挡最多的类别、工具调用次数、活跃工号。"""
    me = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not me or not is_admin(me):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    days = int(req.query_params.get("days", 7) or 7)
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - days * 86400))
    con = sqlite3.connect(DB)
    tot = con.execute("SELECT COUNT(*), SUM(refused) FROM qa_audit WHERE ts>=?", (since,)).fetchone()
    lats = [r[0] for r in con.execute("SELECT latency_ms FROM qa_audit WHERE ts>=? AND latency_ms IS NOT NULL", (since,)).fetchall()]
    cats = con.execute("SELECT blocked_cats FROM qa_audit WHERE ts>=? AND blocked_cats<>''", (since,)).fetchall()
    tools = con.execute("SELECT tools FROM qa_audit WHERE ts>=? AND tools<>'[]'", (since,)).fetchall()
    users = con.execute("SELECT emp_id, COUNT(*) c FROM qa_audit WHERE ts>=? GROUP BY emp_id ORDER BY c DESC LIMIT 10", (since,)).fetchall()
    qs = con.execute("SELECT question, COUNT(*) c FROM qa_audit WHERE ts>=? GROUP BY question ORDER BY c DESC LIMIT 10", (since,)).fetchall()
    con.close()
    lats = sorted(x for x in lats if x is not None)
    def pct(p):
        return lats[min(len(lats) - 1, int(len(lats) * p))] if lats else None
    from collections import Counter
    cc = Counter()
    for (bc,) in cats:
        for c in (bc or "").split("、"):
            if c: cc[c] += 1
    tc = Counter()
    for (tj,) in tools:
        try:
            for t in json.loads(tj or "[]"): tc[t.get("name", "?")] += 1
        except Exception: pass
    total = tot[0] or 0
    ref = tot[1] or 0
    return JSONResponse({"ok": True, "days": days, "total": total, "refused": ref,
        "refused_rate": round(ref * 100.0 / total, 1) if total else 0,
        "latency_p50": pct(0.5), "latency_p95": pct(0.95), "latency_avg": round(sum(lats) / len(lats)) if lats else None,
        "blocked_cats": cc.most_common(10), "tools": tc.most_common(10),
        "top_users": [{"emp_id": e, "n": c} for e, c in users],
        "top_questions": [{"q": qq, "n": c} for qq, c in qs]})

@app.get("/api/roster/list")
async def roster_list(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT r.emp_id, r.name, r.dept, r.title, r.is_leader,"
                        " COALESCE(u.disabled,0), CASE WHEN u.emp_id IS NULL THEN 0 ELSE 1 END"
                        " FROM roster r LEFT JOIN users u ON u.emp_id = r.emp_id"
                        " ORDER BY r.emp_id").fetchall()
    conn.close()
    return JSONResponse({"ok": True, "roster": [
        {"emp_id": e, "name": n, "dept": dp or "", "title": tt or "",
         "is_leader": int(ld or 0), "disabled": int(dis), "registered": int(reg)}
        for e, n, dp, tt, ld, dis, reg in rows]})

# ================= 知识审核（仅管理员） =================
@app.get("/api/knowledge/pending")
async def knowledge_pending(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT id, question, answer, created_by, ts FROM knowledge WHERE status='pending' ORDER BY id").fetchall()
    conn.close()
    return JSONResponse({"ok": True, "items": [{"id": r[0], "question": r[1], "answer": r[2], "created_by": r[3], "ts": r[4]} for r in rows]})

@app.post("/api/knowledge/approve")
async def knowledge_approve(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    kid = d.get("id")
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT question, answer FROM knowledge WHERE id=?", (kid,)).fetchone()
    if row:
        conn.execute("UPDATE knowledge SET status='approved' WHERE id=?", (kid,))
        conn.commit()
        conn.close()
        content = f"问：{row[0]}\n答：{row[1]}"
        add_text_to_global(content, f"知识库-{row[0][:20]}")
        return JSONResponse({"ok": True})
    conn.close()
    return JSONResponse({"ok": False, "msg": "条目不存在"})

@app.post("/api/knowledge/reject")
async def knowledge_reject(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    kid = d.get("id")
    conn = sqlite3.connect(DB)
    conn.execute("UPDATE knowledge SET status='rejected' WHERE id=?", (kid,))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True})

# ================= Skill 平台 =================
@app.post("/api/skill/create")
async def skill_create(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    d = await req.json()
    name = d.get("name", "").strip()
    description = d.get("description", "").strip()
    prompt = d.get("prompt", "").strip()
    if not name or not prompt:
        return JSONResponse({"ok": False, "msg": "名字和提示词不能为空"})
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO skills(name, description, prompt, author, ts) VALUES(?,?,?,?,datetime('now'))",
                 (name, description, prompt, emp_id))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True})

@app.get("/api/skill/list")
async def skill_list(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT id, name, description, author, ts FROM skills ORDER BY id DESC").fetchall()
    conn.close()
    return JSONResponse({"ok": True, "skills": [{"id": r[0], "name": r[1], "description": r[2], "author": r[3], "ts": r[4]} for r in rows]})

@app.get("/api/skill/{sid}")
async def skill_detail(sid: int, req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT id, name, description, prompt, author, ts FROM skills WHERE id=?", (sid,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"ok": False, "msg": "技能不存在"}, status_code=404)
    return JSONResponse({"ok": True, "skill": {"id": row[0], "name": row[1], "description": row[2], "prompt": row[3], "author": row[4], "ts": row[5]}})

@app.get("/api/skill/{sid}/download")
async def skill_download(sid: int, token: str = ""):
    emp_id = sessions.get(token)
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT name, description, prompt, author FROM skills WHERE id=?", (sid,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"ok": False, "msg": "技能不存在"}, status_code=404)
    md = f"# {row[0]}\n\n**作者**: {row[3]}\n**描述**: {row[1]}\n\n## 提示词\n\n{row[2]}\n"
    safe = re.sub(r'[\\/:*?"<>|]', '_', row[0])
    fname = f"{safe}.md"
    fpath = user_files_dir(emp_id) / fname
    fpath.write_text(md, encoding="utf-8")
    return FileResponse(str(fpath), filename=fname)

BUILD_ID = time.strftime("%Y%m%d%H%M%S")   # 服务启动时间戳，用作页面版本号


def _page(name):
    """返回页面：注入版本号 + 彻底禁用缓存。

    - `no-store`：浏览器不许缓存页面，改完界面刷新即生效，不用教用户按 Ctrl+F5；
    - `__BUILD__` → BUILD_ID：页面内所有跳转都带 ?v=<启动时间戳>
      （2026-09-11 踩过大坑：服务端已经把页面改好了，浏览器却还在用缓存里的旧页面互相跳，
       页面请求不产生（走缓存）但 API 请求一直发（history 401），表面看就是"修了没用、
       登录页和聊天页一直切换"。给 URL 打版本号后，跳转目标永远是没缓存过的新 URL。）
    """
    html = (BASE / "static" / name).read_text(encoding="utf-8").replace("__BUILD__", BUILD_ID)
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    })

@app.get("/skills")
async def skills_page():
    return _page("skills.html")

@app.get("/")
async def root():
    return _page("login.html")

@app.get("/chat")
async def chat_page():
    return _page("chat.html")

# ================= 索引状态 + 自动重建（丢文件进 docs/ 即可，不用跑命令） =================
index_state = {"rebuilding": False, "reason": "", "last_error": None, "finished_at": None}
_rebuild_lock = threading.Lock()


def reload_index():
    """重建完成后把磁盘上的 faiss.index / chunks.json 重新读进内存（热加载，不用重启）"""
    global index, chunks, sources, bm25
    with open(INDEX_FILE, "rb") as f:
        index = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
    data = json.load(open(CHUNKS_FILE, encoding="utf-8"))
    chunks = [d["text"] for d in data]
    sources = [d["source"] for d in data]
    bm25 = BM25Okapi([list(jieba.cut(t)) for t in chunks])
    print(f"🔄 已热加载新索引：{len(chunks)} 块", flush=True)


def _rebuild_worker(reason):
    """子进程跑全量重建（独立进程，不占服务线程；重建期间服务照常用旧索引答问）"""
    log_path = BASE_ROOT / "auto_rebuild.log"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} 自动重建：{reason} =====\n")
        log.flush()
        r = subprocess.run([sys.executable, str(BASE_ROOT / "rebuild_index.py")],
                           cwd=str(BASE_ROOT), stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        return f"重建失败(exit {r.returncode})，详见 auto_rebuild.log"
    reload_index()
    return None


def start_rebuild(reason):
    """后台线程触发一次重建；已在重建中则忽略（返回 False）"""
    with _rebuild_lock:
        if index_state["rebuilding"]:
            return False
        index_state.update(rebuilding=True, reason=reason, last_error=None)

    def _run():
        try:
            err = _rebuild_worker(reason)
            index_state["last_error"] = err
            index_state["finished_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{'⚠' if err else '✅'} 自动重建结束：{err or '成功，已热加载新索引'}", flush=True)
        except Exception as e:
            index_state["last_error"] = f"{type(e).__name__}: {e}"
            print(f"⚠ 自动重建异常：{e}", flush=True)
        finally:
            with _rebuild_lock:
                index_state["rebuilding"] = False

    threading.Thread(target=_run, daemon=True, name="auto-rebuild").start()
    return True


def watch_docs():
    """常驻线程：每次间隔比一次 docs 指纹，变了就重建（无需重启服务）"""
    while True:
        time.sleep(AUTO_REBUILD_INTERVAL)
        try:
            stale, why = index_meta.is_stale()
            if stale and not index_state["rebuilding"]:
                print(f"👀 检测到 docs 变化：{why} → 自动重建", flush=True)
                start_rebuild(why)
        except Exception as e:
            print(f"⚠ 指纹检查异常：{e}", flush=True)


@app.get("/api/index_status")
async def api_index_status():
    """索引状态：块数、上次构建时间、是否正在重建（前端头部轮询显示）"""
    meta = index_meta.load() or {}
    return JSONResponse({
        "ok": True,
        "chunks": len(chunks),
        "files": meta.get("files"),
        "built_at": meta.get("built_at"),
        "auto_rebuild": KB_AUTO_REBUILD,
        "rebuilding": index_state["rebuilding"],
        "reason": index_state["reason"],
        "last_error": index_state["last_error"],
        "finished_at": index_state["finished_at"],
    })


@app.post("/api/index/rebuild")
async def api_index_rebuild(req: Request):
    """管理员手动触发一次重建（等不及自动触发时用）"""
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id:
        return JSONResponse({"ok": False, "msg": "未登录"}, status_code=401)
    if not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "仅管理员可触发重建"}, status_code=403)
    return JSONResponse({"ok": True, "started": start_rebuild("管理员手动触发")})


# ================= 启动 =================
init_db()
print(f"🔑 已恢复 {load_sessions()} 个未过期登录态（{TOKEN_TTL_DAYS} 天内重启服务不用重新登录）", flush=True)
index, chunks, sources = load_or_build()
bm25 = BM25Okapi([list(jieba.cut(t)) for t in chunks])

if KB_AUTO_REBUILD:
    _stale, _why = index_meta.is_stale()
    if _stale:
        print(f"📚 索引需要更新：{_why} → 后台自动重建（先用现有 {len(chunks)} 块服务）", flush=True)
        start_rebuild(_why)
    else:
        print(f"📚 索引已是最新：{_why}", flush=True)
    threading.Thread(target=watch_docs, daemon=True, name="docs-watch").start()
    print(f"👀 已开启 docs 目录监听：每 {AUTO_REBUILD_INTERVAL}s 检查一次（KB_AUTO_REBUILD=0 可关）", flush=True)

print(f"✅ 知识库就绪：{len(chunks)} 块，花名表由管理员在界面管理")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8644")))
