# -*- coding: utf-8 -*-
"""数字员工 WebUI 启动器（双击「启动数字员工.bat」调用本文件）。

做四件事：
1. 已经在跑 → 直接打开浏览器，不重复启动
2. 端口被别的程序占着 → 明确告诉你，不闷头失败
3. 没在跑 → 新开一个窗口启动服务，轮询等它就绪（最多 120 秒）再打开浏览器
4. 打印本机/局域网地址、账号、加资料位置

环境变量：PORT 改端口（默认 8644）；KB_NO_BROWSER=1 不自动开浏览器（无界面/测试用）。
"""
import os, sys, time, socket, subprocess, pathlib, urllib.request, webbrowser

ROOT = pathlib.Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PORT = int(os.environ.get("PORT", "8644"))
URL = f"http://127.0.0.1:{PORT}"
NO_BROWSER = os.environ.get("KB_NO_BROWSER") == "1"


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def alive(timeout=3):
    try:
        with urllib.request.urlopen(URL + "/", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "（查不到，可运行 ipconfig 查看）"


def open_browser():
    if NO_BROWSER:
        print("    （KB_NO_BROWSER=1，跳过打开浏览器）")
        return
    webbrowser.open(URL)


print("=" * 62)
print("                 数字员工 WebUI   启动器")
print("=" * 62)
print(f"  项目目录 : {ROOT}")
print(f"  本机访问 : {URL}")
print(f"  局域网   : http://{lan_ip()}:{PORT}   （同一内网/手机可访问）")
print("-" * 62)

if not PY.exists():
    print(f"❌ 没找到虚拟环境：{PY}")
    print("   说明项目被移动过，或 .venv 被删了。")
    input("按回车退出...")
    sys.exit(1)

if alive():
    print("✅ 服务已经在运行，直接打开浏览器（不用重复启动）。")
    open_browser()
    time.sleep(3)
    sys.exit(0)

if port_busy(PORT):
    print(f"⚠ 端口 {PORT} 被别的程序占着（多半是上次没关干净的数字员工服务）。")
    print("   先关掉那个「数字员工服务」黑窗口，再双击本文件。")
    input("按回车退出...")
    sys.exit(1)

print("[1/3] 启动服务（会新开一个黑窗口。加载模型约 20-30 秒，属正常，别关它）")
cmd = [str(PY), "-m", "uvicorn", "app:app", "--app-dir", "webui",
       "--host", "0.0.0.0", "--port", str(PORT)]
proc = subprocess.Popen(cmd, cwd=str(ROOT),
                        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))

print("[2/3] 等待服务就绪（最多 120 秒）...")
t0 = time.time()
while time.time() - t0 < 120:
    if alive():
        break
    if proc.poll() is not None:
        print("❌ 服务进程已退出。请看那个黑窗口里的报错信息。")
        input("按回车退出...")
        sys.exit(1)
    time.sleep(2)
else:
    print("⚠ 等待超时。请看那个「数字员工服务」黑窗口里的报错。")
    input("按回车退出...")
    sys.exit(1)

print(f"[3/3] 已就绪（用时 {time.time() - t0:.0f} 秒），打开浏览器")
open_browser()
print()
print("-" * 62)
print("  登录账号 : 用工号登录（账号由管理员在「花名表」中管理）")
print("  忘记密码 : 双击「重置密码.bat」，或联系管理员")
print("  加资料   : 文档丢进 " + str(ROOT / "docs") + " 里，自动重建索引")
print("  停止服务 : 关掉那个「数字员工服务」黑窗口")
print("-" * 62)
time.sleep(4)
