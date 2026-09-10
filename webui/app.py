# -*- coding: utf-8 -*-
"""数字员工 WebUI 后端：花名表 + 工号登录 + 记忆隔离 + RAG"""
import os, json, re, hashlib, secrets, sqlite3, shutil, threading, time, subprocess
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

os.environ.setdefault("HF_HOME", "E:/hf_cache")
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

# ================= 配置 =================
BASE = Path("E:/数字员工项目/webui")
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

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit(
        "❌ 未找到 DEEPSEEK_API_KEY。请在 " + str(BASE_ROOT / ".env") +
        " 里加一行：DEEPSEEK_API_KEY=sk-xxxxx（或设为系统环境变量后重启）"
    )

# 自动重建开关：KB_AUTO_REBUILD=0 关闭；KB_WATCH_INTERVAL 为指纹检查间隔（秒）
KB_AUTO_REBUILD = os.getenv("KB_AUTO_REBUILD", "1").lower() not in ("0", "false", "no")
AUTO_REBUILD_INTERVAL = int(os.getenv("KB_WATCH_INTERVAL", "60"))
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

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
def retrieve_top_ids(query, faiss_idx, bm25, chunks, k=3):
    cand = retrieval.candidate_ids(query, bm25, faiss_idx, embed_model, n=20)
    scores = retrieval.rerank_scores(query, [chunks[i] for i in cand], reranker)
    ranked = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)
    return [i for i, s in ranked[:k]]

def retrieve_for_user(query, emp_id, k=3):
    """合并全局知识库 + 个人知识库检索，返回 [(text, source)]。检索策略见 retrieval.RETRIEVAL_STRATEGY。"""
    candidates = []
    g_ids = retrieval.candidate_ids(query, bm25, index, embed_model, n=15)
    for i in g_ids:
        if i < len(chunks):
            candidates.append((chunks[i], sources[i]))
    uidx, uchunks, usources, ubm25 = load_user_kb(emp_id)
    if len(uchunks) > 0:
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
        return []
    texts = [t for t, s in dedup[:20]]
    scores = retrieval.rerank_scores(query, texts, reranker)
    ranked = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    return [(dedup[i][0], dedup[i][1]) for i in ranked[:k]]

# ================= 数据库：花名表 + 用户 + 记忆 =================
def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS roster(emp_id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS users(emp_id TEXT PRIMARY KEY, pwd_hash TEXT, salt TEXT)")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
    if "is_admin" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    conn.execute("CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT, topic TEXT, fact TEXT, ts TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT, role TEXT, content TEXT, ts TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS knowledge(id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, status TEXT, created_by TEXT, ts TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS skills(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, prompt TEXT, author TEXT, ts TEXT)")
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
ROLES = {
    "妙妙": "你是妙妙，公司的文档文案专家，负责文档编写、PPT、翻译、文案、投标文件、会议纪要等。你文笔专业、清晰、有条理。当用户要求你修改、优化、改写、润色一份文档时，请分两步输出：第一步，简要列出修改建议（改了哪些地方、为什么这样改）；第二步，单独一行写【成品】作为分隔，然后输出修改后的完整文档内容（这部分会被自动保存成可下载的Word文件）。",
    "准准": "你是准准，公司的数据分析与财务专家，负责数据统计、可视化、报表生成、财务核算、成本分析。你严谨、细致、准确，擅长从表格数据中找出规律和异常，输出清晰的报表和分析结论。",
    "码码": "你是码码，公司的代码研发助手，负责代码检视、测试用例编写、自动化测试、版本发布。你熟悉编程规范和代码质量，能发现代码中的潜在问题、逻辑漏洞和隐患。",
    "图图": "你是图图，公司的机械设计助手，负责图纸审阅、工艺文档编写、BOM整理、选型计算、设计报告、专利底书。你熟悉机械设计规范和图纸标准，能发现图纸中的低级错误和设计问题。",
    "采采": "你是采采，公司的采购助手，负责物料交期确认、成本核实、供应商审查、合同评审、采购谈价。你熟悉采购流程和供应链管理，擅长分析价格和风险。",
    "安安": "你是安安，公司的人事助手，负责考勤初核、档案录入、入职流程、制度答疑、简历初筛。你细心、贴心，熟悉人事行政制度和流程。",
    "顺顺": "你是顺顺，公司的生产助手，负责生产计划、物料需求、工时统计、调试分析。你熟悉生产流程和排产调度，擅长分析生产数据和质量问题。",
    "管家": "你是管家，数字员工团队的总调度员。你负责接收用户的复杂任务，把它拆解成子任务，分发给合适的专家（妙妙/准准/码码/图图/采采/安安/顺顺），最后汇总成完整结果交给用户。",
}

def detect_role(message):
    """判断用户消息是否在召唤某个角色，是则返回角色名，否则 None"""
    names = ["妙妙", "准准", "码码", "图图", "采采", "安安", "顺顺"]
    if not any(n in message for n in names):
        return None
    prompt = (
        "判断用户这句话是否在召唤某个数字员工。员工：妙妙（文档文案）、准准（数据财务）、码码（代码）、图图（机械设计）、采采（采购）、安安（人事）、顺顺（生产）。\n"
        "如果用户在召唤某个员工（例如'妙妙在吗'、'叫码码来'、'安安帮我看看考勤'），只回复对应的名字（妙妙/准准/码码/图图/采采/安安/顺顺）。\n"
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

def expert_run(role_prompt, sub_task):
    """专家 agent 执行子任务，返回结果"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": role_prompt},
                  {"role": "user", "content": sub_task}],
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

def execute_tool(name, args):
    if name == "get_current_time":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif name == "calculate":
        try:
            # 学习阶段用 eval，产品级要换成安全计算（如 ast 解析限制）
            return str(eval(args.get("expression", "")))
        except Exception as e:
            return f"计算错误: {e}"
    return "未知工具"

# ================= 会话（每工号独立） =================
sessions = {}       # token -> emp_id
conversations = {}  # emp_id -> messages 列表

# ================= FastAPI =================
app = FastAPI()

@app.post("/api/register")
async def register(req: Request):
    d = await req.json()
    emp_id = d.get("emp_id", "").strip()
    password = d.get("password", "")
    if not emp_id or not password:
        return JSONResponse({"ok": False, "msg": "工号和密码不能为空"})
    conn = sqlite3.connect(DB)
    in_roster = conn.execute("SELECT 1 FROM roster WHERE emp_id=?", (emp_id,)).fetchone()
    if not in_roster:
        conn.close()
        return JSONResponse({"ok": False, "msg": "工号不在花名表，无法注册"})
    exists = conn.execute("SELECT 1 FROM users WHERE emp_id=?", (emp_id,)).fetchone()
    if exists:
        conn.close()
        return JSONResponse({"ok": False, "msg": "该工号已注册，请直接登录"})
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
    row = conn.execute("SELECT pwd_hash, salt, is_admin FROM users WHERE emp_id=?", (emp_id,)).fetchone()
    conn.close()
    if row and row[0] == hash_pwd(password, row[1]):
        token = secrets.token_hex(16)
        sessions[token] = emp_id
        return JSONResponse({"ok": True, "token": token, "is_admin": bool(row[2] if len(row) > 2 else 0)})
    return JSONResponse({"ok": False, "msg": "工号或密码错误"})

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
    role_prompt = ROLES.get(role, "你是数字员工")
    conv_key = f"{emp_id}:{role}"
    if conv_key not in conversations:
        mems = load_memories(emp_id)
        mt = "\n".join(f"- {m}" for m in mems) if mems else "（暂无）"
        history = load_history(emp_id)
        conversations[conv_key] = [{"role": "system", "content": f"{role_prompt}\n你记得关于这个用户（工号 {emp_id}）的这些事：\n{mt}"}] + history
    messages = conversations[conv_key]
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
                r = expert_run(ROLES[expert], sub)
                results.append((expert, r))
            yield f"data: {json.dumps({'orchestrator': '正在汇总结果...'})}\n\n"
            final = summarize(q, results)
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": final})
            save_message(emp_id, "user", q)
            save_message(emp_id, "assistant", final)
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
            yield f"data: {json.dumps({'delta': final})}\n\n"
            yield f"data: {json.dumps({'done': True, 'role': role})}\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    top = retrieve_for_user(q, emp_id, k=3)
    context = "\n\n".join(f"【来自 {s}】\n{t}" for t, s in top)
    call_messages = messages + [{"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】\n{q}"}]

    # 工具循环：非流式判断要不要调工具，最多 3 轮
    tool_calls_made = []
    for _ in range(3):
        resp = client.chat.completions.create(model="deepseek-chat", messages=call_messages, tools=TOOLS, max_tokens=200)
        msg = resp.choices[0].message
        if msg.tool_calls:
            call_messages.append({"role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = execute_tool(tc.function.name, args)
                tool_calls_made.append({"name": tc.function.name, "result": result})
                call_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            break

    def generate():
        for tc in tool_calls_made:
            yield f"data: {json.dumps({'tool': tc['name'], 'result': tc['result']})}\n\n"
        full = ""
        stream = client.chat.completions.create(model="deepseek-chat", messages=call_messages, temperature=0.3, max_tokens=4096, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full += delta
                yield f"data: {json.dumps({'delta': delta})}\n\n"
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
        yield f"data: {json.dumps({'done': True, 'sources': [s for t, s in top], 'doc': doc_name, 'role': role})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

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
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    d = await req.json()
    new_id = d.get("emp_id", "").strip()
    name = d.get("name", "").strip()
    if not new_id or not name:
        return JSONResponse({"ok": False, "msg": "工号和姓名不能为空"})
    conn = sqlite3.connect(DB)
    conn.execute("INSERT OR REPLACE INTO roster VALUES(?,?)", (new_id, name))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True})

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

@app.get("/api/roster/list")
async def roster_list(req: Request):
    emp_id = sessions.get(req.headers.get("Authorization", "").replace("Bearer ", ""))
    if not emp_id or not is_admin(emp_id):
        return JSONResponse({"ok": False, "msg": "无权限"}, status_code=403)
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT emp_id, name FROM roster ORDER BY emp_id").fetchall()
    conn.close()
    return JSONResponse({"ok": True, "roster": [{"emp_id": r[0], "name": r[1]} for r in rows]})

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

@app.get("/skills")
async def skills_page():
    return FileResponse(str(BASE / "static" / "skills.html"))

@app.get("/")
async def root():
    return FileResponse(str(BASE / "static" / "login.html"))

@app.get("/chat")
async def chat_page():
    return FileResponse(str(BASE / "static" / "chat.html"))

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
