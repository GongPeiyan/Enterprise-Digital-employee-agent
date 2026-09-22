# -*- coding: utf-8 -*-
"""A3 / A6 的地基：把「员工表」翻译成「每个人能看哪些类别」。

核心思路（A6 可维护性的关键）：
    权限不写死在个人身上，而是每次从组织数据算出来：
        员工表（工号 / 部门 / 职位 / 是否负责人）
          ×  acl_matrix.json（类别 → 哪些角色可见）
          =  该员工可见的类别集合
    所以人事变动只改员工表（或授权台账），权限自动跟着变，代码一行不用动。

三类标签的匹配方式：
    全员级：标签 "全员"                        → 例如 A 公共制度 / E 通用技术 / G 安全应急
    部门级：标签 "工程部" / "工程部全体"        → 例  D 项目技术 = 工程部全体 + 品技部
    岗位级：标签 "工程部负责人" / "人力资源主管" → 例  B2 个人薪酬明细 只给 人力资源主管
    授权级：acl_grants.json 里的临时授权/全库授权（A6 的授权台账，带批注与有效期）

用法：
    python acl_roster.py                生成 acl_roster.json + 打印所有角色的可见类别（用来核对矩阵）
    python acl_roster.py --who ENG001   查一个工号能看到什么
    python acl_roster.py --who C 结算采购  解释"某类别谁能看"（反向查）
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
# ── 花名表数据源 ────────────────────────────────────────────────
# A 方案：以管理界面的花名表为准（数据库），管理员在界面上加人/改部门后点「重算权限」即可。
# 也保留走 CSV 的能力（导入/核对用）：--csv <路径> 或环境变量 ACL_ROSTER_SRC。
DEFAULT_DB = BASE / "webui" / "data" / "app.db"
CSV_FALLBACK = Path(os.environ.get("ACL_ROSTER_SRC",
                                   "/Users/a1/Desktop/员工花名册.csv"))
_SRC_NOTE = [""]        # 记录本次用的数据源，打印用
MATRIX = BASE / "acl_matrix.json"
TAGS_BY_SOURCE = BASE / "acl_tags_by_source.json"
GRANTS = BASE / "acl_grants.json"          # 授权台账（临时授权 / 总经理全库），A6 用
OUT = BASE / "acl_roster.json"

# 部门名 → 矩阵里可能出现的写法（员工表写"人力资源"，矩阵写"人力资源部"，都要认）
DEPT_ALIAS = {
    "人力资源": ["人力资源部", "人力资源"],
    "工程部": ["工程部"],
    "研发部": ["研发部"],
    "市场部": ["市场部"],
    "财务部": ["财务部"],
    "总经办": ["总经办"],
    "品技部": ["品技部"],
    "供应链采购部": ["供应链采购部"],
}


def role_labels(dept: str, title: str, is_leader: bool) -> list:
    """把一个人的组织属性翻译成"角色标签"，用于和矩阵里的可见范围做交集。"""
    labels = {"全员"}
    for a in DEPT_ALIAS.get(dept, [dept]):
        labels.add(a)                      # 部门级：如「品技部」
    labels.add(dept + "全体")               # 部门全体：如「工程部全体」
    if is_leader:
        labels.add(dept + "负责人")          # 负责人级：如「工程部负责人」
        if dept == "供应链采购部":
            labels.add("采购负责人")          # 矩阵里用的是这个写法
    if title:
        labels.add(title)                   # 岗位级：如「人力资源主管」
    return sorted(labels)


def read_roster_from_db(db_path: Path) -> list:
    """从数据库花名表读员工（管理界面上维护的那张表）。

    数据库里没有的人 = 不在名册 → 权限表里也没有他 → 登录后什么都看不到（安全侧默认）。
    """
    import sqlite3
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT emp_id, name, dept, title, is_leader FROM roster"
                           " ORDER BY emp_id").fetchall()
    finally:
        con.close()
    out = []
    for e, n, dp, tt, ld in rows:
        out.append({"emp_id": (e or "").strip(), "name": (n or "").strip(),
                    "dept": (dp or "").strip(), "title": (tt or "").strip(),
                    "is_leader": bool(ld)})
    return [p for p in out if p["emp_id"]]


def resolve_roster(csv_path: Path = None, db_path: Path = None) -> list:
    """决定用哪份名册：显式参数 > 数据库 > CSV（数据库存在就优先，A 方案）。"""
    if csv_path:
        _SRC_NOTE[0] = "CSV：%s" % csv_path
        return read_roster(csv_path)
    dbf = db_path or DEFAULT_DB
    if dbf.exists():
        rs = read_roster_from_db(dbf)
        if rs:
            _SRC_NOTE[0] = "数据库花名表：%s（%d 人）" % (dbf, len(rs))
            return rs
    _SRC_NOTE[0] = "CSV：%s（数据库花名表为空，回退到 CSV）" % CSV_FALLBACK
    return read_roster(CSV_FALLBACK)


def read_roster(path: Path) -> list:
    """读员工表 → [{emp_id, name, dept, title, is_leader}]。支持 csv；xlsx 用 openpyxl。"""
    rows = []
    if path.suffix.lower() == ".csv":
        data = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    else:
        from openpyxl import load_workbook
        ws = load_workbook(path, data_only=True).worksheets[0]
        data = [[("" if c is None else str(c)) for c in row] for row in ws.iter_rows(values_only=True)]
    head = next(i for i, r in enumerate(data) if r and r[0].strip() == "工号")
    for r in data[head + 1:]:
        r = (r + [""] * 6)[:6]
        eid = r[0].strip()
        if not eid or not eid[:1].isalpha():          # 跳过「合计」等汇总行
            continue
        rows.append({"emp_id": eid, "name": r[1].strip(), "dept": r[2].strip(),
                     "title": r[3].strip(), "is_leader": r[4].strip() == "是"})
    return rows


def load_grants() -> dict:
    """授权台账：{"GM001": {"scopes": ["全库"], "note": "...", "expires": "2027-12-31"}}"""
    if GRANTS.exists():
        return {k: v for k, v in json.loads(GRANTS.read_text(encoding="utf-8")).items()
                if not k.startswith("_")}
    return {}


def build(csv_path: Path = None, db_path: Path = None):
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    roster = resolve_roster(csv_path=csv_path, db_path=db_path)
    grants = load_grants()
    out = {}
    for p in roster:
        labels = role_labels(p["dept"], p["title"], p["is_leader"])
        labels = labels + list((grants.get(p["emp_id"], {}) or {}).get("scopes", []))
        # 判定：某类别的可见范围 与 这个人的角色标签 有交集 → 可见
        visible = sorted(cat for cat, scopes in matrix.items()
                         if not cat.startswith("_") and set(scopes) & set(labels))
        out[p["emp_id"]] = {**p, "labels": labels, "visible": visible}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, matrix


def all_categories() -> set:
    """全部类别（管理员 / 全库授权用）。"""
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    return {c for c in matrix if not c.startswith("_")}


def visible_categories(emp_id: str) -> set:
    """给 app.py 调用的入口：工号 → 可见类别集合（带 mtime 缓存，改文件即生效）。"""
    global _CACHE
    try:
        mtime = os.stat(OUT).st_mtime if OUT.exists() else 0
    except OSError:
        mtime = 0
    if not _CACHE or _CACHE[0] != mtime:
        data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        _CACHE = (mtime, data)
    rec = _CACHE[1].get(emp_id)
    if not rec:
        return set()                                  # 不在名册里 → 什么都看不到（安全侧默认）
    return all_categories() if "全库" in (rec.get("labels") or []) else set(rec["visible"])


_CACHE = None


def _norm_path(s: str) -> str:
    """路径归一化：反斜杠→正斜杠、去掉所有空白、转小写。

    为什么需要：语料重建后同一份文件可能写成「工程资料/…/55交工证书.docx」，
    而标签表里可能是「天然气\\…\\55交工证书.docx」——只按原样查表会全部落空，
    结果就是所有资料都被判「未分类」锁死，员工什么都查不到。
    """
    return re.sub(r"\s+", "", str(s).replace("\\", "/")).lower()


BLOCK_OVERRIDES = BASE / "acl_block_overrides.json"
_BLOCK_CACHE = []


def _block_rules():
    """读块级覆盖规则（带 mtime 缓存，改文件即生效）。"""
    try:
        mtime = os.stat(BLOCK_OVERRIDES).st_mtime if BLOCK_OVERRIDES.exists() else 0
    except OSError:
        mtime = 0
    if not _BLOCK_CACHE or _BLOCK_CACHE[0] != mtime:
        rules = []
        if BLOCK_OVERRIDES.exists():
            try:
                rules = (json.loads(BLOCK_OVERRIDES.read_text(encoding="utf-8")) or {}).get("rules", [])
            except Exception as e:
                print("[acl] 块级覆盖规则读取失败：%s" % e, flush=True)
        _BLOCK_CACHE[:] = [mtime, rules]
    return _BLOCK_CACHE[1]


def cat_of_block(source: str, text: str = "") -> str:
    """一个语料块属于哪一类：先看块级覆盖（混载文件的例外），再回退到文件级标签。

    为什么要块级：员工手册这类文件里"全员制度"和"薪酬章节"混在一起，
    只按文件打一个类别，会把只该给 HR 的内容一起放开（2026-09-22 B1 审计发现的真实漏洞）。
    """
    src = source or ""
    for r in _block_rules():
        fc = r.get("file_contains") or ""
        if fc and fc in src and any(k and k in (text or "") for k in (r.get("block_any") or [])):
            return r.get("cat") or cat_of_source(src)
    return cat_of_source(src)


def cat_of_source(source: str) -> str:
    """文件路径 → 类别。表里没有的文件默认「未分类」（锁闭），等打标签后自动放开。"""
    if not _TAGS[1]:
        _TAGS[1] = json.loads(TAGS_BY_SOURCE.read_text(encoding="utf-8")) if TAGS_BY_SOURCE.exists() else {}
        _TAGS[2] = {_norm_path(k): v for k, v in _TAGS[1].items()}      # 归一化索引
    hit = _TAGS[2].get(_norm_path(source))
    if hit is not None:
        return hit
    alt = str(source).replace("\\", "/")
    return _TAGS[1].get(source) or _TAGS[1].get(alt) or "未分类"


_TAGS = [0, {}, {}]      # [mtime, 原始表, 归一化索引]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", nargs="+", help="查工号，或 --who 类别名 反查")
    ap.add_argument("--csv", help="改用 CSV 名册（默认读数据库花名表）")
    ap.add_argument("--db", help="指定数据库路径（默认 webui/data/app.db）")
    a = ap.parse_args()

    out, matrix = build(csv_path=Path(a.csv) if a.csv else None,
                        db_path=Path(a.db) if a.db else None)
    print("数据源：%s\n" % _SRC_NOTE[0])
    cats = [c for c in matrix if not c.startswith("_")]
    print("已生成 %s（%d 人）\n" % (OUT.name, len(out)))

    # 按"角色"归并打印（同部门+同身份的人权限一样，所以只列代表）
    seen = set()
    print("%-26s %-12s %s" % ("角色（代表）", "人数", "可见类别"))
    for eid, rec in out.items():
        key = (rec["dept"], rec["is_leader"], rec["title"])
        if key in seen:
            continue
        seen.add(key)
        n = sum(1 for r in out.values() if (r["dept"], r["is_leader"], r["title"]) == key)
        print("%-26s %-12s %s" % ("%s %s" % (rec["name"], rec["title"]), "%d 人" % n,
                                  "／".join(rec["visible"]) or "（无）"))

    if a.who:
        q = a.who[0]
        if q in out:
            r = out[q]
            print("\n%s %s（%s / %s / %s）" % (q, r["name"], r["dept"], r["title"],
                                             "部门负责人" if r["is_leader"] else "员工"))
            print("  角色标签：", "、".join(r["labels"]))
            print("  可见类别：", "／".join(r["visible"]) or "（无）")
        else:
            print("\n%s 能看 %s 的有：%s" % (q, q, "、".join(
                f"{e}({r['name']})" for e, r in out.items() if q in r["visible"]) or "无"))


if __name__ == "__main__":
    main()
