# -*- coding: utf-8 -*-
"""A5 兜底校验：判断一个回答有没有"推断泄露"或"元信息泄露"。

生产用法（app.py 里）：
    from acl_guard import judge
    refused, flags = judge(answer, ctx_texts, forbidden_terms)
    if any(f.startswith(("引用受限", "元信息")) for f in flags):   # 严重的才拦
        answer = prompting.REFUSAL                                 # 兜底换成统一话术

这里的原则：宁可换掉一个回答，也不能让员工从回答里反推出受限资料的存在或内容。
"""
from prompting import REFUSAL

# 元信息泄露用语：出现这些，说明模型在暗示"有一份你看不到的资料"
META_PHRASES = ["无权", "没有权限", "权限不足", "权限不够", "存在但是", "存在但", "受限于权限",
                "受限资料", "被隐藏", "已隐藏", "被拦截", "该资料存在", "这份资料存在",
                "无法访问该资料", "你不在可见范围内"]

REFUSAL_MARK = "找不到依据"          # 统一话术的特征词，用来判断"是否拒答"

# 敏感类别：这些类别的内容一旦出现在回答里，就是要拦的（兜底词表只从这里面抽）
SENSITIVE_CATS = ("B 薪酬人事", "B2 个人薪酬明细", "C 结算采购", "C2 采购台账", "F 经营财务")          # 统一话术的特征词，用来判断"是否拒答"


def digits(s: str) -> set:
    """抽出文本里的数字串（≥2 位）。用于判断"回答里的数字有没有出处"。"""
    out, cur = set(), ""
    for ch in str(s):
        if ch.isdigit():
            cur += ch
        else:
            if len(cur) >= 2:
                out.add(cur)
            cur = ""
    if len(cur) >= 2:
        out.add(cur)
    return out


def refused(answer: str) -> bool:
    """是否按统一话术拒答了。"""
    if REFUSAL_MARK in answer:
        return True
    return ("无法回答" in answer or "没有依据" in answer) and len(answer) < 200


def judge(answer: str, ctx_texts, forbidden_terms, extra_ok=()):
    """返回 (是否拒答, 问题标记列表)。

    ctx_texts       : 这次真正给模型的资料正文（用来判断数字有没有出处 → 编造）
    forbidden_terms : 不该出现在回答里的词（从受限类别里自动抽取：金额数字、文件名、专有名词）
    extra_ok        : 额外的"有出处"文本（如工具调用结果）—— 工具算出来的数字不该被判成编造
    """
    flags = []
    for p in META_PHRASES:
        if p in answer:
            flags.append("元信息泄露(%s)" % p)
    hit = [t for t in forbidden_terms if t and t in answer]
    if hit:
        flags.append("引用受限内容(%s)" % "，".join(hit[:3]))

    ctx_digits = set()
    for t in (ctx_texts or []):
        ctx_digits |= digits(t)
    for t in (extra_ok or []):        # 工具结果里的数字视为有出处
        ctx_digits |= digits(t)
    invented = sorted(d for d in digits(answer) if len(d) >= 3 and d not in ctx_digits)
    if invented:
        flags.append("无依据数字(%s)" % "，".join(invented[:3]))
    return refused(answer), flags


def extract_forbidden(texts, sources, max_terms=12):
    """从一个受限类别的语料里抽出"独有特征词"，用来当越权判据。

    抽什么：
      · ≥4 位的数字串（金额、编号；工资/造价这类最敏感的就是数字）
      · 文件名短名（如 工作簿1.xlsx、2023年度工资、考勤表.xls）
    不抽单个汉字（会误判），宁少不滥。
    """
    terms = set()
    for t in texts:
        for d in digits(t):
            if len(d) >= 4:
                terms.add(d)
    for s in sources:
        name = str(s).replace("\\", "/").split("/")[-1]
        if len(name) >= 6:
            terms.add(name)
    return sorted(terms)[:max_terms]
