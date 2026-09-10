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
for path, marker in [("/", "登录工作台"), ("/chat", "数字员工"), ("/skills", "技能库")]:
    r = c.get(path)
    txt = r.text
    ok = (r.status_code == 200)
    print(f"  {path:9s} HTTP {r.status_code}  {len(txt)} 字符  含关键内容={marker in txt}", flush=True)
# 关键能力标记
chat = c.get("/chat").text
for k in ("renderMD", "kbStatus", 'id="emptyState"', "toggleTheme", "syncSendBtn", "重建索引"):
    print(f"  /chat 含 {k}: {k in chat}", flush=True)
print(f"  启动耗时 {time.time()-t0:.1f}s，知识库 {len(A.chunks)} 块", flush=True)
