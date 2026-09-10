# -*- coding: utf-8 -*-
"""隔离沙箱测试：验证「往 docs/ 丢文件 → 自动重建索引 → 热加载」整条链路。

不动真实索引：一切路径指到 _sandbox_kb/。
"""
import os, sys, json, shutil, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SB = ROOT / "_sandbox_kb"
shutil.rmtree(SB, ignore_errors=True)
(SB / "docs").mkdir(parents=True)

os.environ["KB_DOCS_DIR"] = str(SB / "docs")
os.environ["KB_INDEX_FILE"] = str(SB / "faiss.index")
os.environ["KB_CHUNKS_FILE"] = str(SB / "chunks.json")
os.environ["KB_META_FILE"] = str(SB / "index_meta.json")
os.environ["KB_WATCH_INTERVAL"] = "5"          # 指纹检查间隔缩短，便于测试
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")

# 造两份小文档
(SB / "docs" / "考勤制度.md").write_text(
    "# 考勤制度\n公司上班时间为上午8:30至12:00，下午14:00至18:00。\n"
    "员工迟到超过30分钟按旷工半天处理。\n年休假：工作满1年不满10年的，年休假5天。\n", encoding="utf-8")
(SB / "docs" / "安全制度.txt").write_text(
    "燃气管道巡检每两周一次。\n发现泄漏立即关阀通风，禁止明火。\n", encoding="utf-8")

sys.path.insert(0, str(ROOT / "webui"))
t0 = time.time()
print(f"[{time.time()-t0:6.1f}s] 导入 app（会触发启动流程：建索引 + 指纹检查 + 自动重建）...", flush=True)
import app as A
print(f"[{time.time()-t0:6.1f}s] 导入完成，当前块数={len(A.chunks)}", flush=True)
print(f"[{time.time()-t0:6.1f}s] 状态: rebuilding={A.index_state['rebuilding']} reason={A.index_state['reason']}", flush=True)

from fastapi.testclient import TestClient
c = TestClient(A.app)

# ---- 1) 等首次自动重建完成 ----
deadline = time.time() + 300
while time.time() < deadline:
    st = c.get("/api/index_status").json()
    if not st["rebuilding"]:
        break
    time.sleep(3)
print(f"[{time.time()-t0:6.1f}s] 首次自动重建结束: chunks={st['chunks']} files={st['files']} "
      f"built_at={st['built_at']} err={st['last_error']}", flush=True)
meta = json.loads((SB / "index_meta.json").read_text(encoding="utf-8"))
print(f"           index_meta.json: chunks={meta['chunks']} files={meta['files']} fp={meta['fingerprint'][:10]}", flush=True)
assert not st["rebuilding"] and st["chunks"] > 0 and not st["last_error"], "首次重建未成功"
assert st["chunks"] == len(A.chunks), f"热加载未生效：status={st['chunks']} 内存={len(A.chunks)}"
first_chunks = st["chunks"]

# ---- 2) 模拟「丢一个新文件进 docs/」，不重启服务，看是否自动重建 ----
(SB / "docs" / "薪酬制度.md").write_text(
    "# 薪酬制度\n基本工资按月发放。\n高温补贴每人每月200元，发放6至8月。\n"
    "全勤奖每月300元。\n", encoding="utf-8")
print(f"[{time.time()-t0:6.1f}s] 已丢入新文件 薪酬制度.md，等服务自动发现（间隔5s）...", flush=True)

found = False
while time.time() < deadline:
    st2 = c.get("/api/index_status").json()
    if (not st2["rebuilding"]) and st2["chunks"] > first_chunks:
        found = True
        break
    time.sleep(3)
print(f"[{time.time()-t0:6.1f}s] 第二次自动重建结果: chunks={st2['chunks']}（原 {first_chunks}） "
      f"reason={st2['reason']} err={st2['last_error']}", flush=True)
assert found, "新文件未被自动检测/重建"
assert st2["chunks"] > first_chunks, "块数没增长"
assert st2["chunks"] == len(A.chunks), "第二次热加载未生效"

# ---- 3) 检索验证：新文件内容能被检索到（走生产的 retrieval + rerank）----
import retrieval
ids = retrieval.candidate_ids("高温补贴多少钱", A.bm25, A.index, A.embed_model, n=10)
hit = any("高温补贴" in A.chunks[i] for i in ids)
print(f"[{time.time()-t0:6.1f}s] 检索'高温补贴多少钱' → top10 命中新文件: {hit}", flush=True)
assert hit, "新文件内容检索不到"

# ---- 4) 删掉一个文件，验证删除也触发重建 ----
(SB / "docs" / "安全制度.txt").unlink()
print(f"[{time.time()-t0:6.1f}s] 已删除 安全制度.txt，等待自动重建...", flush=True)
while time.time() < deadline:
    st3 = c.get("/api/index_status").json()
    if (not st3["rebuilding"]) and st3["chunks"] < st2["chunks"]:
        break
    time.sleep(3)
print(f"[{time.time()-t0:6.1f}s] 删除后: chunks={st3['chunks']}（原 {st2['chunks']}）files={st3['files']} err={st3['last_error']}", flush=True)
assert st3["chunks"] < st2["chunks"], "删除文件后索引没更新"

print("\n✅ 沙箱测试全部通过：首次自动重建 / 新增自动重建 / 删除自动重建 / 热加载 / 检索可见", flush=True)
