#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Universal Correction Guard 2.0 (全通用无硬编码大模型字幕纠错守门员)
=============================================================================
特性：
1. 长度屏障 (Layer 1)：非标点正文字符长度比例约束，防止跨行搬运与整句替换；
2. 口吃叠词极速通道 (Layer 2)：自然消除即兴口语结巴/叠词 (如 "自自己" -> "自己")；
3. 块级声学校验 (Layer 3)：
   - 严格数值一致性：口述数字不可篡改为不同数值 (拦截 200.1.1.1 -> 192.168.1.1)；
   - 专业学术单位/点分地址转换 (100千赫兹 -> 100 kHz, 零点零点零点零 -> 0.0.0.0)；
   - 普通话声母混淆组声学比对 (b/p, d/t, g/k, z/c/s, zh/ch/sh, f/h, l/n)；
   - 彻底拦截非同音篡改、语义脑补 (如 "骚图" -> "表图", "泛红" -> "之外", "针" -> "地址")。
"""

from __future__ import annotations

import difflib
import re
from typing import Any, List, Optional, Tuple

try:
    from pypinyin import lazy_pinyin
except Exception:
    lazy_pinyin = None

CJK_PATTERN = re.compile(r'[\u4e00-\u9fff]')
LATIN_PATTERN = re.compile(r'[a-zA-Z]')
PUNCT_CLEAN_RE = re.compile(r'[\s,，.。、；;：:!！?？"\'“”‘’()（）\[\]【】《》<>—\-…~～/\\_#@*&^%+=|`]+')

CN_NUM_AND_UNIT_CHARS = set('零〇一幺二两三四五六七八九十百千万亿点.分之兆千百十微毫纳皮秒分时天米克伏安欧瓦赫赫兹摩尔')
LATIN_NUM_AND_UNIT_CHARS = set('0123456789. %/kMGTmupnsdhvkAVWHzmolL')

CN_DIGIT_MAP = {
    '零': '0', '〇': '0', '一': '1', '幺': '1', '二': '2', '两': '2',
    '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'
}

CONSONANT_ACOUSTIC_GROUPS = [
    {"b", "p"}, {"d", "t"}, {"g", "k"}, {"j", "q", "x"},
    {"z", "c", "s"}, {"zh", "ch", "sh", "r"}, {"f", "h"}, {"l", "n", "r"}
]


def is_cjk(s: str) -> bool:
    return bool(CJK_PATTERN.search(s))


def extract_digits(s: str) -> str:
    """从中文数字或 ASCII 数字中提取纯数字序列"""
    res = []
    for c in s:
        if c in CN_DIGIT_MAP:
            res.append(CN_DIGIT_MAP[c])
        elif c.isdigit():
            res.append(c)
    return ''.join(res)


def get_initial_consonant(py: str) -> str:
    for c in ["zh", "ch", "sh"]:
        if py.startswith(c):
            return c
    if py and py[0] in "bpmfdtnlgkhjqxzcsryw":
        return py[0]
    return ""


def are_confusable_initials(c1: str, c2: str) -> bool:
    if not lazy_pinyin:
        return False
    p1 = lazy_pinyin(c1)
    p2 = lazy_pinyin(c2)
    if not p1 or not p2:
        return False
    i1 = get_initial_consonant(p1[0])
    i2 = get_initial_consonant(p2[0])
    if i1 == i2 and i1:
        return True
    for g in CONSONANT_ACOUSTIC_GROUPS:
        if i1 in g and i2 in g:
            return True
    return False


def is_stutter_reduction(orig: str, corr: str) -> bool:
    """检测是否属于自然消除口吃/结巴叠词 (如 自自己 -> 自己, 在在一个 -> 在一个)"""
    o = PUNCT_CLEAN_RE.sub('', orig).strip()
    c = PUNCT_CLEAN_RE.sub('', corr).strip()
    if not o or not c:
        return False
    if len(o) <= len(c) or len(o) - len(c) > 6:
        return False

    # 单字口吃叠词: AA -> A
    for i in range(len(o) - 1):
        if o[i] == o[i + 1]:
            reduced = o[:i] + o[i + 1:]
            if reduced == c:
                return True
    # 2~4 字词级叠词: ABAB -> AB
    for w_len in (2, 3, 4):
        for i in range(len(o) - 2 * w_len + 1):
            w1 = o[i:i + w_len]
            w2 = o[i + w_len:i + 2 * w_len]
            if w1 == w2:
                reduced = o[:i] + w1 + o[i + 2 * w_len:]
                if reduced == c:
                    return True
    return False


def is_numeral_or_unit_conversion(sub_orig: str, sub_corr: str) -> bool:
    """判断是否属于口述数字、地址或计量单位的标准化规范"""
    so = sub_orig.strip()
    sc = sub_corr.strip()
    if not so or not sc:
        return False

    # 变量标识符: 如 h零 -> H0, c二 -> C2
    ident_match = re.match(r'^([a-zA-Z])([零〇一幺二两三四五六七八九十0-9]+)$', so)
    if ident_match:
        if re.match(r'^[a-zA-Z][0-9]+$', sc):
            return so[0].lower() == sc[0].lower()
        return False

    if all(c in CN_NUM_AND_UNIT_CHARS or c in LATIN_NUM_AND_UNIT_CHARS for c in so):
        if all(c in LATIN_NUM_AND_UNIT_CHARS or c in CN_NUM_AND_UNIT_CHARS for c in sc):
            d_so = extract_digits(so)
            d_sc = extract_digits(sc)
            if d_so and d_sc and d_so != d_sc:
                return False
            return True
    return False


def normalize_fuzzy_pinyin(syllables: List[str]) -> List[str]:
    """声学模糊拼音映射归一化"""
    res = []
    for s in syllables:
        s = s.lower()
        if s.startswith('p'): s = 'b' + s[1:]
        elif s.startswith('t'): s = 'd' + s[1:]
        elif s.startswith('k'): s = 'g' + s[1:]
        elif s.startswith('l'): s = 'n' + s[1:]
        elif s.startswith('zh'): s = 'z' + s[2:]
        elif s.startswith('ch'): s = 'c' + s[2:]
        elif s.startswith('sh'): s = 's' + s[2:]
        if s.endswith('eng'): s = s[:-3] + 'en'
        elif s.endswith('ing'): s = s[:-3] + 'in'
        res.append(s)
    return res


def is_phonetic_match(a: str, b: str) -> bool:
    """检测两个词是否发音相同或高度近似 (支持声母混淆、音译外来词)"""
    if not a or not b:
        return False
    if a == b:
        return True
    if not lazy_pinyin:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.75

    pa = lazy_pinyin(a)
    pb = lazy_pinyin(b)
    if pa == pb:
        return True

    sa = "".join(pa).lower()
    sb = "".join(pb).lower()
    if sa == sb:
        return True

    has_latin_a = bool(LATIN_PATTERN.search(a))
    has_latin_b = bool(LATIN_PATTERN.search(b))

    # Case A: 纯中文近音/同音校验
    if not has_latin_a and not has_latin_b:
        if abs(len(a) - len(b)) > 1:
            return False
        if difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.70:
            return True
        fa = "".join(normalize_fuzzy_pinyin(pa))
        fb = "".join(normalize_fuzzy_pinyin(pb))
        return difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.70

    # Case B: 包含英文字母/音译词 (如 麦克针 <-> MAC帧)
    if is_cjk(a[-1]) and is_cjk(b[-1]):
        tail_a_py = lazy_pinyin(a[-1])[0]
        tail_b_py = lazy_pinyin(b[-1])[0]
        if difflib.SequenceMatcher(None, tail_a_py, tail_b_py).ratio() < 0.70:
            return False

    if is_cjk(a[0]) and is_cjk(b[0]):
        head_a_py = lazy_pinyin(a[0])[0]
        head_b_py = lazy_pinyin(b[0])[0]
        if difflib.SequenceMatcher(None, head_a_py, head_b_py).ratio() < 0.70:
            return False

    sm = difflib.SequenceMatcher(None, sa, sb)
    ratio = sm.ratio()
    if sa and sb and sa[0] == sb[0] and ratio >= 0.48:
        return True

    return ratio >= 0.70


def verify_correction(original: str, corrected: str, profile: Optional[Any] = None) -> Tuple[bool, str]:
    """
    通用纠错守门员校验核心入口：
    返回: (是否通过: bool, 原因描述: str)
    """
    if not corrected or not corrected.strip():
        return False, "empty_output"

    orig_clean = (original or "").strip()
    corr_clean = (corrected or "").strip()

    if orig_clean == corr_clean:
        return True, "identical"

    # Layer 1: 非标点正文字符长度突变检查
    c_orig_raw = PUNCT_CLEAN_RE.sub('', orig_clean)
    c_corr_raw = PUNCT_CLEAN_RE.sub('', corr_clean)
    if c_orig_raw:
        ratio = len(c_corr_raw) / max(1, len(c_orig_raw))
        max_ratio = 2.0 if len(c_orig_raw) <= 5 else 1.65
        min_ratio = 0.40 if len(c_orig_raw) <= 5 else 0.55
        if ratio < min_ratio or ratio > max_ratio:
            return False, f"length_ratio_{ratio:.2f}"

    # Layer 2: 口吃/叠词消除极速放行
    if is_stutter_reduction(orig_clean, corr_clean):
        return True, "accepted_stutter_reduction"

    # 去标点比较
    if not c_orig_raw and not c_corr_raw:
        return True, "accepted_punctuation_only"

    if c_orig_raw == c_corr_raw:
        return True, "accepted_formatting_only"

    # Layer 3: 块级逐片段细致检验
    sm = difflib.SequenceMatcher(None, c_orig_raw, c_corr_raw)
    blocks = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in blocks:
        if tag == 'equal':
            continue
        sub_orig = c_orig_raw[i1:i2]
        sub_corr = c_corr_raw[j1:j2]

        if not sub_orig and not sub_corr:
            continue

        # 检查数值/单位转化
        if is_numeral_or_unit_conversion(sub_orig, sub_corr):
            continue

        # 严格数值一致性：原数字不能变成其他无关数字
        d_so = extract_digits(sub_orig)
        d_sc = extract_digits(sub_corr)
        if d_so and d_sc and d_so != d_sc:
            return False, f"digit_mismatch_{d_so}->{d_sc}"

        # 大小写拉丁匹配 (如 mac -> MAC)
        if sub_orig.lower() == sub_corr.lower():
            continue

        # 领域画像混淆词验证
        if profile and hasattr(profile, "confusion_rules"):
            clean_so = PUNCT_CLEAN_RE.sub('', sub_orig).lower()
            clean_sc = PUNCT_CLEAN_RE.sub('', sub_corr).lower()
            if clean_so in profile.confusion_rules and PUNCT_CLEAN_RE.sub('', profile.confusion_rules[clean_so]).lower() == clean_sc:
                if is_phonetic_match(sub_orig, sub_corr) or is_numeral_or_unit_conversion(sub_orig, sub_corr):
                    continue

        # 专业术语尾部补齐 (如 SIF -> SIFS)
        if profile and getattr(profile, "jargon_list", None):
            so_u = sub_orig.upper()
            sc_u = sub_corr.upper()
            if so_u and any(j.upper() == sc_u and j.upper().startswith(so_u) and len(j) - len(so_u) <= 2 for j in profile.jargon_list):
                continue

        # 发音相近匹配
        if is_phonetic_match(sub_orig, sub_corr):
            continue

        # 单字声学混淆扩展匹配 (声母同组且非数字)
        if len(sub_orig) <= 1 and len(sub_corr) <= 1:
            if not d_so and not d_sc and are_confusable_initials(sub_orig, sub_corr):
                ctx_o = c_orig_raw[max(0, i1 - 1):min(len(c_orig_raw), i2 + 1)]
                ctx_c = c_corr_raw[max(0, j1 - 1):min(len(c_corr_raw), j2 + 1)]
                if is_phonetic_match(ctx_o, ctx_c):
                    continue

        return False, f"non_phonetic_change_{sub_orig}->{sub_corr}"

    return True, "accepted"
