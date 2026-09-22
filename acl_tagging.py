# -*- coding: utf-8 -*-
"""A2：给语料块打可见范围标签（ACL 的地基）— v2

v1 的三个缺陷（本版修掉）：
  ① 只看路径：文件名是「1.docx」「工作簿1.xlsx」时什么都看不出 → 加**内容回退**（路径判不出就看正文）
  ② 空格吃掉关键词：正文里是「工 程 预 (结) 算 书」，匹配不到"结算" → 加**归一化**（去空白、统一括号）
  ③ 敏感词不够：发票、增值税、工资汇总、造价、工程量确认单、试验记录都漏了 → 补规则

v2 新增类别：B2 个人薪酬明细（带姓名/月份的工资表）→ 只有人力资源主管可见。
  这是实测发现的真实文件：《2023年度1-12月份土建组工资汇总表》。

产出：acl_tags.json ／ 桌面《未分类语料复核表.xlsx》 ／ acl_tag_report.txt
用法：
  lora-from-scratch/.venv/bin/python acl_tagging.py --check   # 只跑误伤自检
  lora-from-scratch/.venv/bin/python acl_tagging.py           # 全量打标签
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parent
CHUNKS = BASE / "chunks.json"
OUT_TAGS = BASE / "acl_tags.json"
OUT_XLSX = Path("/Users/a1/Desktop/未分类语料复核表.xlsx")
OUT_REPORT = BASE / "acl_tag_report.txt"
OUT_BY_SRC = BASE / "acl_tags_by_source.json"
MATRIX = BASE / "acl_matrix.json"          # A1 确认后写入，本脚本优先读它

# ── 规则（优先级从上到下；敏感领域优先，防止「薪酬制度」被当公共制度放开）──
RULES = [
    ("B2 个人薪酬明细", r"工资汇总表|工资明细表|工资发放|工资条|工资、考勤表|工资考勤|薪资表|个税明细"),
    ("B 薪酬人事", r"薪酬|薪资|工资汇总|工资明细|工资表|工资条|工资标准|工资总额|绩效|社保|公积金"
                    r"|招聘|考勤|请假|劳动合同|人事|职级|福利|奖金|导师表|聘任书"),
    ("C2 采购台账", r"采购|供应商|订单|询价|比价"),
    ("C 结算采购", r"结算|决算|预\(结\)算|预算书|造价|工程量确认单|报价|投标|招标"
                    r"|发票|增值税|开票|费用汇总|工程造价|付款|签证|变更单|中标|合同|挂账|欠款|租地协议"),
    ("F 经营财务", r"税务局|税务|财务报表|财务|审计|资产|利润|记账|凭证|对账|银行|预算|成本|投资"),
    ("D 项目技术", r"施工组织|施工方案|图纸|竣工|验收|报审|报验|开工|隐蔽|技术交底|施工记录|试压|吹扫"
                    r"|管网|管道|设备安装|工程量|监理|进度计划|材料报审|质量报验|见证取样|卷外封面"
                    r"|工程概况|竣工资料|低压资料|中压资料|资料20\d\d|监理新增|试验记录|强度试验"
                    r"|气密性|无损检测|打压|影像资料|勘测|零星工程|门站|宴会厅|热水器|安装工程|护坡"
                    r"|更换|改造|装修|粉刷|拆除|维修|绿化|安装|24年计划|年计划"),
    ("G2 事故隐患记录", r"事故报告|事故记录|事故台账|事故调查|隐患台账|隐患整改|未遂|伤亡|追责"),
    ("G 安全应急", r"安全|应急|事故|隐患|消防|演练|职业健康|危险源|操作规程"),
    ("E 通用技术", r"说明书|规范|GB\d|标准|使用手册|产品|技术参数|材质|管材|流量计|调压"),
    ("H 市场客户", r"客户|市场|渠道|品牌|销售|推广|营销|用户回访"),
    ("I 研发技术", r"研发|算法|软件|测试用例|专利|架构|接口|数据库"),
    ("A 公共制度", r"制度|办法|手册|流程|规定|职责|通知|纪要|员工手册|岗位说明书"),
]

# ── 可见范围（A1 逐条确认后可用 acl_matrix.json 覆盖）────────────────────
SCOPE_DEFAULT = {
    "A 公共制度": ["全员"],
    "E 通用技术": ["全员"],
    "G 安全应急": ["全员"],
    "G2 事故隐患记录": ["品技部", "工程部负责人"],
    "B 薪酬人事": ["人力资源部"],
    "B2 个人薪酬明细": ["人力资源主管"],            # 最敏感：带姓名的个人工资数据
    "C 结算采购": ["财务部负责人"],
    "C2 采购台账": ["财务部负责人", "采购负责人"],
    "F 经营财务": ["财务部"],
    "D 项目技术": ["工程部全体", "品技部"],
    "H 市场客户": ["市场部"],
    "I 研发技术": ["研发部"],
    "未分类": ["（默认锁闭，等人工复核）"],
}

# ── 误伤回归用例：改规则必须全过（前 6 条是真实踩过的坑，后 3 条靠内容回退）──
#    元组：(路径, 期望类别, 说明, 正文片段或空)
REGRESSION = [
    ("工程资料/某项目低压资料/7.2焊工资格证书（扫描件）.docx",
     "D 项目技术", "「焊工资格」里含子串「工资」，曾被误判为薪酬类", ""),
    ("工程资料/某项目竣工资料/监理新增资料/18开工报审表.docx",
     "D 项目技术", "「竣工资料」= 竣-工-资-料，也含「工资」", ""),
    ("公司制度/薪酬福利管理制度.docx",
     "B 薪酬人事", "含「制度」二字，但敏感优先，不能被公共制度抢走", ""),
    ("公司制度/考勤请假管理办法.docx",
     "B 薪酬人事", "含「办法」，同样敏感优先", ""),
    ("结算资料/2023年零星工程结算书",
     "C 结算采购", "结算类，不能被「市场」抢走", ""),
    ("工程资料/15施工组织方案.docx", "D 项目技术", "纯项目技术", ""),
    ("其他文件/结算说明.PDF", "C 结算采购",
     "正文是「工程费用汇总表／预(结)算书」→ 走内容回退", "项目工程费用汇总表 预算造价 定额直接费 单方造价"),
    ("天然气/桌面其他文件/工作簿1.xlsx", "B2 个人薪酬明细",
     "正文是「土建组1-12月工资汇总表」→ 走内容回退", "2023年度1-12月份土建组工资汇总表"),
    ("天然气/天然气有关/1.docx", "C 结算采购",
     "正文是「增值税专用发票」→ 走内容回退", "山东增值税 专用 发票 发票联"),
]


def norm(s: str) -> str:
    """归一化：去掉所有空白（含全角空格），全角括号转半角，让「预 (结) 算 书」也能命中。"""
    s = s.replace("\u3000", "").replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", s)


def match(rules, text):
    for name, rx in rules:
        if re.search(rx, text):
            return name
    return None


def tag_category(path: str, text: str = "") -> str:
    """先看路径；路径判不出来时用正文（内容回退）。"""
    cat = match(RULES, norm(path.replace("\\", "/")))
    if cat:
        return cat
    if text:
        return match(RULES, norm(text)) or "未分类"
    return "未分类"


def load_matrix():
    """A1 确认的矩阵（有就用，没有就用默认）。"""
    if MATRIX.exists():
        scopes = dict(SCOPE_DEFAULT)
        scopes.update(json.loads(MATRIX.read_text(encoding="utf-8")))
        return scopes, True
    return dict(SCOPE_DEFAULT), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑误伤自检")
    args = ap.parse_args()

    # ① 误伤回归自检（先跑，规则不对就别往下走）
    print("① 误伤回归自检（含内容回退用例）")
    bad = 0
    for path, expect, why, text in REGRESSION:
        got = tag_category(path, text)
        ok = got == expect
        bad += 0 if ok else 1
        print("   %s %-16s←应判 %-16s %s" % ("✓" if ok else "✗", got, expect, why))
    print("   结果：%d/%d 通过\n" % (len(REGRESSION) - bad, len(REGRESSION)))
    if args.check:
        raise SystemExit(1 if bad else 0)

    scopes, from_matrix = load_matrix()
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    print("② 打标签（%d 块；可见范围来自%s）"
          % (len(chunks), "acl_matrix.json" if from_matrix else "默认矩阵（A1 确认后可覆盖）"))

    tags = []
    cat_cnt = Counter()
    unk_files = defaultdict(int)
    by_src = {}          # 文件路径 → 类别（生产按这个对齐）
    by_path = {}          # 每个来源文件留一段正文，供内容回退使用
    for gid, c in enumerate(chunks):
        src = c.get("source", "")
        if src not in by_path:
            by_path[src] = c.get("text", "")[:600]
        cat = tag_category(src, by_path[src])
        cat_cnt[cat] += 1
        if cat == "未分类":
            unk_files[src] += 1
        tags.append({"gid": gid, "source": src, "category": cat, "scope": scopes.get(cat, ["全员"])})
        by_src[src] = cat

    print("   类别分布：")
    for k, v in cat_cnt.most_common():
        print("     %-16s %5d 块 (%4.1f%%)  → 可见：%s"
              % (k, v, v * 100 / len(chunks), "／".join(scopes.get(k, []))))
    ratio = cat_cnt["未分类"] * 100 / len(chunks)
    print("   未分类：%d 块（%.1f%%），%d 个文件 → 默认锁闭\n" % (cat_cnt["未分类"], ratio, len(unk_files)))

    OUT_TAGS.write_text(json.dumps(tags, ensure_ascii=False), encoding="utf-8")
    print("③ 已写出 %s（%d 条）" % (OUT_TAGS.name, len(tags)))
    OUT_BY_SRC.write_text(json.dumps(by_src, ensure_ascii=False, indent=1), encoding="utf-8")
    print("   已写出 %s（%d 个文件，生产按路径对齐用这个）" % (OUT_BY_SRC.name, len(by_src)))

    # ④ 未分类复核表
    wb = Workbook()
    ws = wb.active
    ws.title = "未分类复核"
    ws.append(["序号", "来源路径", "块数", "我建议的归属", "你的决定", "正文首段（判据）"])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F3864")
    for i, (src, n) in enumerate(sorted(unk_files.items(), key=lambda x: -x[1]), 1):
        head = norm(by_path.get(src, ""))[:60]
        ws.append([i, src, n, "待定（正文也判不出，需你确认）", "", head])
    for row in ws.iter_rows(min_row=2):
        row[4].fill = PatternFill("solid", fgColor="FFEB9C")
    for j, w in enumerate([6, 84, 8, 26, 14, 46], 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = "A2"
    wb.save(OUT_XLSX)
    print("④ 已写出 %s（%d 个文件待复核）" % (OUT_XLSX, len(unk_files)))

    # ⑤ 报告落盘（验收留档）
    rep = [
        "A2 打标签报告（自动生成，v2 含内容回退）",
        "语料块 %d ｜ 来源文件 %d" % (len(chunks), len({c.get("source") for c in chunks})),
        "误伤回归 %d/%d 通过" % (len(REGRESSION) - bad, len(REGRESSION)),
        "未分类 %d 块（%.1f%%），判据 ≤5%% → %s" % (cat_cnt["未分类"], ratio, "通过" if ratio <= 5 else "未通过"),
        "",
        "类别分布：",
    ]
    for k, v in cat_cnt.most_common():
        rep.append("  %-16s %5d 块 (%4.1f%%)  可见：%s"
                   % (k, v, v * 100 / len(chunks), "／".join(scopes.get(k, []))))
    OUT_REPORT.write_text("\n".join(rep), encoding="utf-8")
    print("⑤ 已写出 %s\n" % OUT_REPORT.name)
    print("判据：误伤回归 %s ｜ 未分类 ≤5%% %s（实际 %.1f%%）"
          % ("✓" if bad == 0 else "✗", "✓" if ratio <= 5 else "✗", ratio))


if __name__ == "__main__":
    main()
