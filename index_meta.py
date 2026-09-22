# -*- coding: utf-8 -*-
"""知识库索引元信息：docs/ 目录指纹 + 索引构建记录。

为什么要它：让"往 docs/ 丢文件"不用再手动跑命令。
- rebuild_index.py 建完索引后写 index_meta.json（记录 docs 指纹 + 块数 + 时间）
- webui/app.py 启动时与运行中定时比对指纹，发现变化就自动后台重建并热加载

指纹 = 所有文件 (相对路径, 大小, mtime) 排序后取 sha1。
新增 / 删除 / 改名 / 改内容（mtime 变）都会让指纹变化。

环境变量 KB_DOCS_DIR 可覆盖 docs 目录（测试用，默认项目根/docs）。
"""
import hashlib, json, os, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = BASE / "docs"


def meta_file():
    """index_meta.json 路径（KB_META_FILE 可覆盖，测试/多库用）"""
    v = os.environ.get("KB_META_FILE")
    return Path(v) if v else BASE / "index_meta.json"


def index_file():
    """faiss.index 路径（KB_INDEX_FILE 可覆盖）"""
    v = os.environ.get("KB_INDEX_FILE")
    return Path(v) if v else BASE / "faiss.index"


def chunks_file():
    """chunks.json 路径（KB_CHUNKS_FILE 可覆盖）"""
    v = os.environ.get("KB_CHUNKS_FILE")
    return Path(v) if v else BASE / "chunks.json"

# 向量库支持的格式 / 暂不支持的格式（与 rebuild_index.py、app.py 保持一致）
SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls"}
UNSUPPORTED_EXTS = {".doc", ".wps", ".ppt", ".pptx", ".csv"}


def docs_dir():
    """docs 目录（KB_DOCS_DIR 可覆盖，测试用）"""
    d = os.environ.get("KB_DOCS_DIR")
    return Path(d) if d else DEFAULT_DOCS_DIR


def scan(docs=None):
    """扫描 docs 目录，返回排序后的 [(相对路径, 字节数, mtime_ns), ...]。
    跳过 ~$ 开头的 Word/WPS 临时锁文件。不区分格式（含不支持的），
    这样"丢进来一个 .doc"也会让指纹变化，进而触发重建并打印跳过提示。"""
    d = Path(docs) if docs else docs_dir()
    out = []
    if not d.exists():
        return out
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.name.startswith("~$"):
            continue
        st = f.stat()
        out.append((str(f.relative_to(d)), st.st_size, st.st_mtime_ns))
    return out


def fingerprint(docs=None):
    """docs 目录指纹（sha1 十六进制）。空目录返回固定值，不报错。"""
    h = hashlib.sha1()
    for rel, size, mtime in scan(docs):
        h.update(f"{rel}\x00{size}\x00{mtime}\n".encode("utf-8"))
    return h.hexdigest()


def supported_files(docs=None):
    """只返回受支持格式的文件（供建索引用）"""
    d = Path(docs) if docs else docs_dir()
    files, unsupported = [], []
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.name.startswith("~$"):
            continue
        ext = f.suffix.lower()
        if ext in SUPPORTED_EXTS:
            files.append(f)
        elif ext in UNSUPPORTED_EXTS:
            unsupported.append(f)
    return files, unsupported


def load():
    """读 index_meta.json；不存在或损坏返回 None"""
    try:
        return json.loads(meta_file().read_text(encoding="utf-8"))
    except Exception:
        return None


def save(chunks, files, docs=None, extra=None):
    """索引建完后写入元信息"""
    meta = {
        "fingerprint": fingerprint(docs),
        "chunks": chunks,
        "files": files,
        "docs_dir": str(docs_dir()),
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    meta_file().write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def is_stale():
    """索引是否需要重建：(1) 没有 meta（老索引，未记录指纹）(2) 指纹不一致"""
    meta = load()
    if not meta or not meta.get("fingerprint"):
        return True, "无索引元信息（首次启用自动重建）"
    now = fingerprint()
    if meta["fingerprint"] != now:
        old_n = meta.get("files")
        new_n = len(scan())
        return True, f"docs 目录已变化（文件数 {old_n} → {new_n}）"
    return False, "索引是最新的"
