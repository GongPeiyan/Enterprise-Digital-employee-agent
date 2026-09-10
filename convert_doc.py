# -*- coding: utf-8 -*-
"""批量 .doc -> .docx（v2）：Word 崩溃后自动重启继续，跳过已存在的。"""
import win32com.client, time
from pathlib import Path

DOCS = Path("E:/数字员工项目/docs")
doc_files = [f for f in DOCS.rglob("*.doc") if f.is_file() and not f.name.startswith("~$")]
print(f"待转换 {len(doc_files)} 个 .doc（已存在的 .docx 会跳过）", flush=True)

def start_word():
    w = win32com.client.Dispatch("Word.Application")
    w.Visible = False
    try:
        w.DisplayAlerts = 0
    except Exception:
        pass
    return w

word = start_word()
ok = skip = 0
fail = []
for i, f in enumerate(doc_files):
    docx_path = f.with_suffix(".docx")
    if docx_path.exists():
        skip += 1
        continue
    try:
        doc = word.Documents.Open(str(f), ReadOnly=True, AddToRecentFiles=False)
        doc.SaveAs(str(docx_path), FileFormat=16)  # docx
        doc.Close(False)
        ok += 1
        print(f"[{i+1}/{len(doc_files)}] OK: {f.name}", flush=True)
    except Exception as e:
        fail.append((str(f.relative_to(DOCS)), str(e)[:70]))
        print(f"[{i+1}/{len(doc_files)}] 失败: {f.name} ({type(e).__name__})", flush=True)
        # Word 可能崩了，重启
        try:
            word.Quit()
        except Exception:
            pass
        time.sleep(1)
        word = start_word()

try:
    word.Quit()
except Exception:
    pass
print(f"DONE 成功 {ok}, 跳过 {skip}, 失败 {len(fail)}", flush=True)
for p, e in fail:
    print(f"  失败: {p}", flush=True)
