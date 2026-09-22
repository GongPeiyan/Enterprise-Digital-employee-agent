# -*- coding: utf-8 -*-
"""注册时的姓名核对（独立成模块，便于单测；webui/app.py 只调用）。

为什么需要：注册只校验"工号在花名表里"，密码又是本人自设的——
所以谁先拿 HR001 去注册，谁就拿到了人力资源主管的权限（含个人薪酬可见）。
批量开户（工号先被占用）是主防线，姓名核对是第二道。

比对规则（宽严的取舍）：
  · 去掉所有空白、中点、全角/半角差异后比较 —— 员工输入「张 三」「张三」都算对
  · 花名表里姓名为空时放行（兼容老数据，不至于把所有人挡在门外）
  · 其余情况一律不放行，报「姓名与花名表不一致」
"""
import re
import unicodedata


def norm_name(s: str) -> str:
    """归一化姓名：全角转半角、去掉空白与中点、统一大小写。"""
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = re.sub(r"[\s·・.．\-—_]", "", s)
    return s.strip().lower()


def name_matches(stored: str, given: str) -> bool:
    """stored = 花名表里的姓名；given = 员工注册时填的姓名。"""
    a, b = norm_name(stored), norm_name(given)
    if not a:                       # 花名表没登记姓名 → 放行（并把填写的姓名存进去）
        return True
    if not b:                       # 员工没填 → 不放行
        return False
    return a == b
