# -*- coding: utf-8 -*-
import os, sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME","E:/hf_cache"); os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET","1"); os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1"); os.environ.setdefault("NO_PROXY","*")
sys.path.insert(0, str(ROOT/"webui"))
t0 = time.time()
import app as A
from fastapi.testclient import TestClient
c = TestClient(A.app)
st = c.get("/api/index_status").json()
print(f"[{time.time()-t0:5.1f}s] 启动成功，未崩溃", flush=True)
print("启动状态:", st, flush=True)
print("首页 200:", c.get("/").status_code, "| /chat 200:", c.get("/chat").status_code, flush=True)
print("真实检索抽查:", [s for _, s in A.retrieve_for_user("示例路工程造价", "CHECK", k=2)], flush=True)
