# -*- coding: utf-8 -*-
"""A6：权限维护机制 —— 每天/每次人事变动跑一次，保证"谁该看什么"始终是对的。

它做四件事：
  ① 同步：把员工表 → acl_roster.json（谁有哪些角色、能看哪些类别），并**打印变更清单**
     （新入职 / 离职 / 转部门 / 升职·降职 / 改名）—— 人事变动后不用改代码，跑一下就行
  ② 校验：五类异常一次性查出来，输出「权限异常清单」
     a. 离职未禁用：名册里已经没有这个工号，但账号还在（现在系统没有这条保护，最危险）
     b. 有账号无名册 / 有名册无账号（前者要禁用，后者登不进来）
     c. 授权台账：过期的授权要失效、引用了不存在的工号或类别要报错
     d. 新语料未打标签：这些文件在生产里是"锁闭"状态，员工查不到 → 提醒重跑 acl_tagging.py
     e. 组织数据异常：职位为空、同一职位名有多人（B2 个人薪酬是按职位名授权的，多人会放大范围）
  ③ 台账：桌面《授权台账.xlsx》是唯一入口（临时授权/代理/总经理全库），本脚本读它 → 生成 acl_grants.json
  ④ 审计：每次权限变更追加一行到 acl_audit.log（谁、什么时候、从什么变成什么、依据哪份名册）

用法：
  python acl_sync.py --check          只检查，不动任何文件（建议每天跑）
  python acl_sync.py --apply          同步 + 更新快照 + 写审计（人事变动后跑）
  python acl_sync.py --init-template  在桌面生成《授权台账.xlsx》模板
"""
import argparse
import json
import shutil
import sqlite3
import time
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent
import acl_roster                                  # noqa: E402  （复用同一套判定，绝不另写一份）

SNAP = BASE / "acl_roster_snapshot.json"           # 上一次的名册快照（用来算变更）
AUDIT = BASE / "acl_audit.log"                     # 权限变更审计
GRANTS_XLSX = Path("/Users/a1/Desktop/授权台账.xlsx")
REPORT = BASE / "acl_sync_report.txt"
TAGS_BY_SOURCE = BASE / "acl_tags_by_source.json"


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def snapshot_of(roster_rows):
    """名册 → {工号: {部门/职位/负责人}}，用于比对变更。"""
    return {r["emp_id"]: {"name": r["name"], "dept": r["dept"],
                          "title": r["title"], "is_leader": r["is_leader"]} for r in roster_rows}


def diff_roster(old, new):
    """算人事变更清单。"""
    out = []
    for eid, n in new.items():
        o = old.get(eid)
        if not o:
            out.append(("新入职", eid, "%s %s/%s%s" % (n["name"], n["dept"], n["title"],
                                                     "（负责人）" if n["is_leader"] else "")))
            continue
        if o["dept"] != n["dept"]:
            out.append(("转部门", eid, "%s → %s" % (o["dept"], n["dept"])))
        if o["is_leader"] != n["is_leader"]:
            out.append(("升职/免职", eid, "%s：负责人 %s → %s" % (n["name"], o["is_leader"], n["is_leader"])))
        if o["title"] != n["title"]:
            out.append(("职位变更", eid, "%s → %s" % (o["title"], n["title"])))
        if o["name"] != n["name"]:
            out.append(("改名", eid, "%s → %s" % (o["name"], n["name"])))
    for eid, o in old.items():
        if eid not in new:
            out.append(("离职", eid, "%s %s/%s" % (o["name"], o["dept"], o["title"])))
    return out


def check_accounts(roster_ids):
    """名册 vs 生产账号库：离职未禁用 / 有账号无名册。找不到库就跳过（本机没有生产库）。"""
    dbs = sorted(list((BASE / "webui").rglob("*.db")) + list((BASE / "webui").rglob("*.sqlite")))
    if not dbs:
        return ["未找到生产账号库（webui/ 下无 .db）→ 离职禁用检查需在服务器上运行；建议纳入每日任务"], []
    msgs, stale = [], []
    for db in dbs:
        try:
            con = sqlite3.connect(str(db))
            cols = [r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()]
            if not cols:
                msgs.append("%s 里没有 users 表" % db.name)
                continue
            idcol = "emp_id" if "emp_id" in cols else cols[0]
            rows = con.execute("SELECT %s FROM users" % idcol).fetchall()
            ids = [str(r[0]) for r in rows]
            stale = [i for i in ids if i not in roster_ids]
            never = [i for i in roster_ids if i not in ids]
            if stale:
                msgs.append("【离职未禁用】%s（%s 里还有账号，但名册里已经没有这些人）" % ("、".join(stale), db.name))
            if never:
                msgs.append("【有名册无账号】%s（这些人登不进系统，需补账号）" % "、".join(never[:20]))
            if not stale and not never:
                msgs.append("账号与名册一致（%d 个账号）" % len(ids))
        except Exception as e:
            msgs.append("%s 读取失败：%s" % (db.name, e))
    return msgs, stale


def load_grants_from_xlsx():
    """桌面《授权台账.xlsx》→ 授权字典。字段：工号/姓名/授权范围/事由/批准人/生效/失效。"""
    if not GRANTS_XLSX.exists():
        return None, "桌面还没有《授权台账.xlsx》（可用 --init-template 生成）"
    from openpyxl import load_workbook
    ws = load_workbook(GRANTS_XLSX, data_only=True)["授权台账"]
    rows = list(ws.iter_rows(values_only=True))
    head = rows[0]
    idx = {str(h).strip(): i for i, h in enumerate(head) if h}
    out, msgs = {}, []
    for r in rows[1:]:
        eid = str(r[idx["工号"]] or "").strip()
        if not eid or eid.startswith("示例"):
            continue
        scopes = str(r[idx["授权范围"]] or "").replace("，", ",").split(",")
        scopes = [s.strip() for s in scopes if s.strip()]
        out[eid] = {"scopes": scopes,
                    "note": str(r[idx["事由"]] or ""),
                    "approved_by": str(r[idx["批准人"]] or ""),
                    "expires": str(r[idx["失效日期"]] or "")}
    return out, None


def check_grants(grants, roster_ids, matrix):
    msgs = []
    today = date.today().isoformat()
    for eid, g in (grants or {}).items():
        if eid not in roster_ids:
            msgs.append("【台账错】工号 %s 不在员工名册里" % eid)
        for s in g.get("scopes", []):
            if s != "全库" and s not in matrix:
                msgs.append("【台账错】%s 的授权范围「%s」不是有效类别名" % (eid, s))
        exp = (g.get("expires") or "").strip()
        if exp and exp < today:
            msgs.append("【授权过期】%s 的授权已于 %s 到期，应失效（从台账删掉或改日期）" % (eid, exp))
        if not g.get("approved_by"):
            msgs.append("【台账缺批准人】%s 的授权没有批准人" % eid)
    return msgs


def check_tags(chunks_srcs):
    tags = load_json(TAGS_BY_SOURCE, {})
    missing = sorted({s for s in chunks_srcs if s not in tags})
    msgs = []
    if missing:
        msgs.append("【新语料未打标签】%d 个文件（生产里是锁闭状态，员工查不到）→ 跑 acl_tagging.py 后重跑 acl_roster.py："
                    % len(missing))
        for x in missing[:5]:
            msgs.append("      %s" % x)
    return msgs


def check_org(rows):
    """组织数据异常：职位为空 / 同一职位名多人（按职位名授权的类别会被放大）。"""
    msgs = []
    no_title = [r["emp_id"] for r in rows if not r["title"]]
    if no_title:
        msgs.append("【组织数据】%d 人没有职位：%s" % (len(no_title), "、".join(no_title[:10])))
    from collections import Counter
    c = Counter(r["title"] for r in rows if r["title"])
    dup = [t for t, n in c.items() if n > 1 and t in ("人力资源主管", "总经理")]
    if dup:
        msgs.append("【组织数据】职位名重复：%s（这类职位是按名字授权的，多人会放大可见范围）" % "、".join(dup))
    leaders = [r for r in rows if r["is_leader"]]
    c2 = Counter(r["dept"] for r in leaders)
    multi = [d for d, n in c2.items() if n > 1]
    if multi:
        msgs.append("【组织数据】一个部门有多个负责人：%s（确认是否符合预期）" % "、".join(multi))
    return msgs


def make_template():
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "授权台账"
    head = ["工号", "姓名", "授权范围（类别名或 全库，多个用逗号）", "事由", "批准人", "生效日期", "失效日期"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.append(["示例（正式使用时删掉本行）", "示例", "全库", "总经理查看全库", "总经理", "2026-09-22", "2027-12-31"])
    ws2 = wb.create_sheet("填写说明")
    for line in [
        "1. 只登记「不随岗位自动获得」的权限：总经理全库、出差代理、专项授权。",
        "2. 部门负责人/人力资源主管这类权限由员工表自动推导，不要写进本台账。",
        "3. 授权范围写类别名（见 acl_matrix.json，如 C 结算采购）或写全库。",
        "4. 必须填批准人和失效日期；过了失效日期，acl_sync.py 会报「授权过期」并要求失效。",
        "5. 每次改动都要在「事由」里写清原因，供审计追溯。",
        "6. 保存为桌面《授权台账.xlsx》，然后跑：python acl_sync.py --apply",
    ]:
        ws2.append([line])
    ws2.column_dimensions["A"].width = 90
    for col, w in zip("ABCDEFG", (18, 10, 40, 30, 12, 14, 14)):
        ws.column_dimensions[col].width = w
    wb.save(GRANTS_XLSX)
    print("已生成模板 →", GRANTS_XLSX)



# ==================== 花名表同步 与 批量开户 ====================
DB_DEFAULT = BASE / "webui" / "data" / "app.db"
ROSTER_COLS = [("dept", "TEXT"), ("title", "TEXT"), ("is_leader", "INTEGER")]   # 权限判定需要的组织字段


def db_path(arg=None):
    """数据库位置：--db 优先，否则自动找 webui 下的 .db。"""
    if arg:
        return Path(arg)
    if DB_DEFAULT.exists():
        return DB_DEFAULT
    found = sorted(list((BASE / "webui").rglob("*.db")))
    return found[0] if found else DB_DEFAULT


def sync_roster_to_db(db, roster_rows):
    """员工表 → 数据库花名表 roster（谁可以注册）；顺便补齐 dept/title/is_leader 三列。"""
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE IF NOT EXISTS roster(emp_id TEXT PRIMARY KEY, name TEXT)")
    cols = [r[1] for r in con.execute("PRAGMA table_info(roster)").fetchall()]
    for name, typ in ROSTER_COLS:
        if name not in cols:
            con.execute("ALTER TABLE roster ADD COLUMN %s %s" % (name, typ))
    for r in roster_rows:
        con.execute("INSERT OR REPLACE INTO roster(emp_id, name, dept, title, is_leader) VALUES(?,?,?,?,?)",
                    (r["emp_id"], r["name"], r["dept"], r["title"], 1 if r["is_leader"] else 0))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM roster").fetchone()[0]
    con.close()
    return n


def open_accounts(db, roster_rows, handout=None, only_new=True):
    """批量开户：名册里没有账号的人 → 建号 + 随机初始密码。

    为什么必须批量开户：注册只校验"工号在花名表里"、密码由本人自设，
    所以谁先拿 HR001 去注册，谁就拿到了人力资源主管的权限。
    先把账号占住（初始密码由管理员发放），冒名注册这条路就堵上了。
    返回 (新建账号列表, 已存在列表)。
    """
    import secrets as _secrets
    CHARSET = "ACDEFGHJKLMNPQRTUVWXY34679acdefghjkmnpqrtuvwxy"   # 去掉 0/O/1/l/I 等易混字符
    con = sqlite3.connect(str(db))
    cols = [r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()]
    if "is_admin" not in cols:
        con.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    have = {r[0] for r in con.execute("SELECT emp_id FROM users").fetchall()}
    created, existing = [], []
    for r in roster_rows:
        if r["emp_id"] in have:
            existing.append(r["emp_id"])
            continue
        if only_new:
            pwd = "".join(_secrets.choice(CHARSET) for _ in range(10))
            salt = _secrets.token_hex(16)
            con.execute("INSERT INTO users(emp_id, pwd_hash, salt, is_admin) VALUES(?,?,?,0)",
                        (r["emp_id"], acl_roster_shash(pwd, salt), salt))
            created.append({"emp_id": r["emp_id"], "name": r["name"], "dept": r["dept"],
                            "title": r["title"], "pwd": pwd})
    con.commit()
    con.close()
    return created, existing


def acl_roster_shash(pwd, salt):
    """与 webui/app.py 的 hash_pwd 完全一致：sha256(盐 + 密码)。"""
    import hashlib
    return hashlib.sha256((salt + pwd).encode()).hexdigest()


def disable_stale(db, roster_ids, dry=True):
    """禁用"名册里没有但账号还在"的人（离职）。管理员账号只报告、绝不动。"""
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT emp_id, is_admin FROM users").fetchall()
    stale = [(e, a) for e, a in rows if e not in roster_ids]
    victims = [e for e, a in stale if not a]          # 管理员不动
    protected = [e for e, a in stale if a]
    if not dry and victims:
        # 置为"停用"而不是删除：保留账号与密码，随时可启用（删号会导致误删后要重新开户）
        cols = [r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()]
        if "disabled" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN disabled INTEGER DEFAULT 0")
        con.execute("UPDATE users SET disabled=1 WHERE emp_id IN (%s)" % ",".join("?" * len(victims)), victims)
        for e in victims:
            con.execute("DELETE FROM tokens WHERE emp_id=?", (e,))
        con.commit()
    con.close()
    return victims, protected


def write_handout(created, path: Path):
    """《初始密码发放表》——含明文初始密码，发完即删。"""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "初始密码发放表"
    ws.append(["工号", "姓名", "部门", "职位", "初始密码"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in created:
        ws.append([r["emp_id"], r["name"], r["dept"], r["title"], r["pwd"]])
    for col, w in zip("ABCDE", (14, 12, 16, 18, 16)):
        ws.column_dimensions[col].width = w
    ws2 = wb.create_sheet("使用说明")
    for line in [
        "1. 本表含明文初始密码，只发给本人，发放完成后请删除本文件（不要留在共享盘/微信群里）。",
        "2. 账号已预先建好：员工打开 http://<服务器IP>:8644 用「工号 + 初始密码」直接登录。",
        "3. 员工若自己去注册页注册，会提示「该工号已注册，请直接登录」——这就是批量开户的目的：",
        "   防止有人用别人的工号抢先注册，从而拿到该工号的权限（例如人力资源主管可见的个人薪酬）。",
        "4. 建议发密码时要求员工登录后尽快改密码（改密码功能上线前，可由管理员用 reset_pwd 脚本重置）。",
    ]:
        ws2.append([line])
    ws2.column_dimensions["A"].width = 100
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--init-template", action="store_true")
    ap.add_argument("--db", default=None, help="数据库路径（默认 webui/data/app.db）")
    ap.add_argument("--sync-db", action="store_true", help="把员工表同步进数据库花名表 roster")
    ap.add_argument("--open-accounts", action="store_true", help="给名册里没有账号的人批量开户并发初始密码")
    ap.add_argument("--disable-stale", action="store_true", help="禁用名册外的账号（管理员账号不动）")
    ap.add_argument("--handout", default=str(Path.home() / "Desktop" / "初始密码发放表（请勿外传）.xlsx"))
    a = ap.parse_args()
    if a.init_template:
        return make_template()

    matrix = load_json(BASE / "acl_matrix.json", {})
    rows = acl_roster.read_roster(acl_roster.ROSTER_SRC)
    new = snapshot_of(rows)
    old = load_json(SNAP, {})
    changes = diff_roster(old, new) if old else []

    # ① 同步
    roster, _ = acl_roster.build()                 # 重建 acl_roster.json
    ids = set(r["emp_id"] for r in rows)

    # ② 各类校验
    acct, stale = check_accounts(ids)
    grants, gmsg = load_grants_from_xlsx()
    gchk = check_grants(grants, ids, matrix) if grants is not None else [gmsg] if gmsg else []
    chunks = load_json(BASE / "chunks.json", [])
    tagchk = check_tags([c["source"] for c in chunks])
    orgchk = check_org(rows)

    # ③ 台账 → acl_grants.json（代码只读 JSON）
    if grants is not None:
        payload = {"_说明": "由 桌面《授权台账.xlsx》自动生成，勿手改；改台账后跑 acl_sync.py --apply",
                   "_生成时间": time.strftime("%Y-%m-%d %H:%M:%S")}
        payload.update(grants)
        (BASE / "acl_grants.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ④ 汇总
    L = ["权限同步与异常清单（acl_sync.py）", "=" * 46,
         "运行时间：%s ｜ 名册 %d 人 ｜ 类别 %d 个" % (time.strftime("%Y-%m-%d %H:%M:%S"), len(rows),
                                                len([k for k in matrix if not k.startswith("_")])),
         "", "① 人事变更（与上次快照比）"]
    if not old:
        L.append("   首次运行，无历史快照可比对（已建立基线）")
    elif not changes:
        L.append("   无变更")
    for kind, eid, detail in changes:
        L.append("   %-8s %s  %s" % (kind, eid, detail))
    for title, items in (("② 账号与名册", acct), ("③ 授权台账", gchk),
                         ("④ 语料标签", tagchk), ("⑤ 组织数据", orgchk)):
        L += ["", title + ("（无异常）" if not items else "")]
        for m in (items or ["   无异常"]):
            L.append("   " + m)
    L += ["", "处置建议：",
          "   · 离职未禁用 → 在账号库里禁用/删除该工号（这是当前系统缺的一环，务必每天跑）",
          "   · 新语料未打标签 → 跑 acl_tagging.py、acl_roster.py，再重跑 acl_visibility_test.py（A4）",
          "   · 授权过期 → 从台账删除或延长日期",
          "   · 组织数据异常 → 修正员工表（职位/负责人字段）"]
    # ── A6 延伸：数据库侧动作（花名表同步 / 批量开户 / 离职禁用）──
    db = db_path(a.db)
    dbmsg = []
    if a.sync_db or a.open_accounts or a.disable_stale:
        if not db.exists():
            dbmsg.append("找不到数据库 %s（在服务器上运行才会生效）" % db)
        else:
            if a.sync_db:
                n = sync_roster_to_db(db, rows)
                dbmsg.append("花名表已同步：roster 共 %d 人（含部门/职位/负责人三列）" % n)
            if a.open_accounts:
                created, existing = open_accounts(db, rows)
                if created:
                    write_handout(created, Path(a.handout))
                    dbmsg.append("批量开户：新建 %d 个账号，已存在 %d 个 → 初始密码发放表：%s"
                                 % (len(created), len(existing), a.handout))
                else:
                    dbmsg.append("批量开户：无需新建（名册 %d 人全部已有账号）" % len(existing))
            if a.disable_stale:
                victims, protected = disable_stale(db, ids, dry=not a.apply)
                dbmsg.append("离职禁用：待处理 %d 个%s%s"
                             % (len(victims), "（本次为预演，未动手；加 --apply 才真的禁用）" if not a.apply else "（已禁用）",
                                ("；管理员账号受保护不动：%s" % "、".join(protected)) if protected else ""))
            print("\n".join("   " + m for m in dbmsg))

    if dbmsg:
        L += ["", "⑥ 数据库侧动作（花名表 / 开户 / 离职禁用）"] + ["   " + m for m in dbmsg]
    REPORT.write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L))
    print("\n已写出", REPORT.name)


    if a.apply:
        SNAP.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        errs = [x for x in (gchk + tagchk + orgchk) if "【" in x]
        with AUDIT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": "权限同步",
                                "roster": len(rows), "changes": len(changes),
                                "anomalies": len([x for x in (acct + gchk + tagchk + orgchk) if "【" in x]),
                                "source": acl_roster.ROSTER_SRC.name}, ensure_ascii=False) + "\n")
            for kind, eid, detail in changes:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": "权限变更",
                                    "type": kind, "emp_id": eid, "detail": detail,
                                    "source": acl_roster.ROSTER_SRC.name}, ensure_ascii=False) + "\n")
            if errs:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": "权限异常",
                                    "items": errs, "stale_accounts": stale}, ensure_ascii=False) + "\n")
        print("已更新快照 %s，并写入审计 %s（本次 %d 条变更）"
              % (SNAP.name, AUDIT.name, len(changes)))
    else:
        print("（--check 模式：未改动任何文件；确认无误后用 --apply 落地）")


if __name__ == "__main__":
    main()
