# -*- coding: utf-8 -*-
"""命令行重置密码（服务器上用；界面走不通或要批量重置时用）。

用法：
  python reset_pwd.py --list                        看有哪些账号
  python reset_pwd.py --emp_id HR001               重置为随机新密码（打印出来）
  python reset_pwd.py --emp_id HR001 --pwd Abcd1234 指定新密码（≥8 位）
  python reset_pwd.py --emp_id HR001 --handout     同时写入桌面《密码重置记录.xlsx》
  python reset_pwd.py --db <路径>                   指定数据库（默认 webui/data/app.db）
"""
import argparse
import hashlib
import secrets
import sqlite3
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_DB = BASE / "webui" / "data" / "app.db"
CHARSET = "ACDEFGHJKLMNPQRTUVWXY34679acdefghjkmnpqrtuvwxy"


def shash(pwd, salt):
    return hashlib.sha256((salt + pwd).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--emp_id")
    ap.add_argument("--pwd")
    ap.add_argument("--handout", action="store_true")
    a = ap.parse_args()

    db = Path(a.db)
    if not db.exists():
        raise SystemExit("找不到数据库：%s" % db)
    con = sqlite3.connect(str(db))
    if a.list:
        rows = con.execute("SELECT emp_id, is_admin FROM users ORDER BY emp_id").fetchall()
        print("共 %d 个账号：" % len(rows))
        for e, ad in rows:
            print("   %-10s %s" % (e, "管理员" if ad else ""))
        return
    if not a.emp_id:
        raise SystemExit("请加 --emp_id，或 --list 查看账号")
    if not con.execute("SELECT 1 FROM users WHERE emp_id=?", (a.emp_id,)).fetchone():
        raise SystemExit("工号 %s 还没有账号（未注册/未开户）" % a.emp_id)
    pwd = (a.pwd or "").strip() or "".join(secrets.choice(CHARSET) for _ in range(10))
    if len(pwd) < 8:
        raise SystemExit("新密码至少 8 位")
    salt = secrets.token_hex(16)
    con.execute("UPDATE users SET pwd_hash=?, salt=? WHERE emp_id=?", (shash(pwd, salt), salt, a.emp_id))
    con.execute("DELETE FROM tokens WHERE emp_id=?", (a.emp_id,))
    con.commit()
    con.close()
    print("已重置 %s 的密码：%s" % (a.emp_id, pwd))
    print("（该员工此前的登录已失效，需用新密码重新登录）")
    if a.handout:
        from openpyxl import Workbook
        out = Path.home() / "Desktop" / "密码重置记录.xlsx"
        if out.exists():
            wb = __import__("openpyxl").load_workbook(out)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.title = "密码重置记录"
            ws.append(["时间", "工号", "新密码", "操作人"])
        ws.append([time.strftime("%Y-%m-%d %H:%M:%S"), a.emp_id, pwd, "命令行"])
        wb.save(out)
        print("已写入", out)


if __name__ == "__main__":
    main()
