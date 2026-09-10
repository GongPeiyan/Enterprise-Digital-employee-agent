# -*- coding: utf-8 -*-
"""重置 / 新建 WebUI 登录密码（双击「重置密码.bat」调用本文件）。

用法一：双击 重置密码.bat，按提示输入工号和新密码
用法二：.venv\Scripts\python.exe reset_pwd.py 001 新密码

哈希算法与 webui/app.py 完全一致（sha256(salt + 密码)），改完立即生效、无需重启服务。
"""
import sys, sqlite3, secrets, hashlib, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DB = ROOT / "webui" / "data" / "app.db"


def hash_pwd(password, salt):
    return hashlib.sha256((salt + password).encode()).hexdigest()


def main():
    if not DB.exists():
        print(f"❌ 找不到数据库：{DB}")
        return 1
    conn = sqlite3.connect(DB)
    accounts = conn.execute("SELECT emp_id, is_admin FROM users ORDER BY emp_id").fetchall()
    roster = conn.execute("SELECT emp_id, name FROM roster ORDER BY emp_id").fetchall()
    print("现有账号 :", "、".join(f"{e}{'(管理员)' if a else ''}" for e, a in accounts) or "（无）")
    print("花名表   :", "、".join(f"{e} {n}" for e, n in roster) or "（无）")
    print()

    if len(sys.argv) >= 3:
        emp_id, pwd = sys.argv[1].strip(), sys.argv[2]
    else:
        emp_id = input("要重置哪个工号（如 001）：").strip()
        pwd = input("新密码：").strip()
    if not emp_id or not pwd:
        print("❌ 工号和新密码都不能为空")
        conn.close()
        return 1

    salt = secrets.token_hex(16)
    exists = conn.execute("SELECT 1 FROM users WHERE emp_id=?", (emp_id,)).fetchone()
    if exists:
        conn.execute("UPDATE users SET pwd_hash=?, salt=? WHERE emp_id=?",
                     (hash_pwd(pwd, salt), salt, emp_id))
        act = "密码已重置"
    else:
        is_admin = 1 if emp_id.lower() == "admin" else 0
        conn.execute("INSERT INTO users(emp_id, pwd_hash, salt, is_admin) VALUES(?,?,?,?)",
                     (emp_id, hash_pwd(pwd, salt), salt, is_admin))
        act = "已新建账号（" + ("管理员" if is_admin else "普通用户") + "）"
        if not is_admin:
            print("   注：正常流程应由管理员在页面「花名表」里加工号、员工自己注册。")
    conn.commit()
    conn.close()
    print(f"✅ {emp_id} {act}。现在可用新密码登录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
