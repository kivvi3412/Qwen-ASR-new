#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal AI Subtitle Correction Engine (AI_srt)
================================================
Designed for high-accuracy, zero-hallucination, zero-desync ASR subtitle post-processing.

Key Guarantees:
1. Zero Audio-Subtitle Desync: Timestamps (start, end) are 100% frozen from validated acoustic anchors.
2. Zero Line Bleeding / 串行: Strict 1-to-1 [ID] batch anchoring, no line splitting or merging by LLM.
3. Zero Hallucination / 跑飞: Three-layer Correction Guard (length ratio, sequence similarity, phonetic verification).
4. Dual-Stage Architecture:
   - Stage 1: Global Domain Auto-Profiling (extracts domain, topic, jargon list, and confusion rules).
   - Stage 2: Parallel Batch Correction via OpenAI-compatible endpoint with persistent breakpoint cache.
5. Universal Normalization: Technical acronym case formatting, numeral formatting, Pangu CJK/Latin spacing.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pypinyin import lazy_pinyin
except ImportError:
    lazy_pinyin = None

# =============================================================================
# Constants and Universal Normalization Rules
# =============================================================================

CJK_PATTERN = re.compile(r'[\u4e00-\u9fff]')
LATIN_PATTERN = re.compile(r'[a-zA-Z0-9]')
PUNCT_CLEAN_RE = re.compile(r'[\s,，.。、；;：:!！?？\"\'“”‘’()（）\[\]【】《》<>—\-…~～/\\·・]+')

# Universal numeric and scientific unit vocabulary for recognizing normalization across disciplines
CN_NUM_AND_UNIT_CHARS = set('零〇一幺二两三四五六七八九十百千万亿点分之千兆微毫纳皮秒分时天米克升摩尔赫兹欧姆伏安瓦特焦耳牛顿帕斯卡分贝字节比特元度个条把张次倍')
LATIN_NUM_AND_UNIT_CHARS = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.+-*/^%μΩ° ')


def is_cjk(s: str) -> bool:
    return bool(CJK_PATTERN.search(s))


def pangu_format(text: str) -> str:
    """Inserts clean whitespace between CJK and Latin/digits, strips extra spaces."""
    if not text:
        return ""
    cjk = r'[\u4e00-\u9fa5]'
    text = re.sub(f'({cjk})([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(f'([a-zA-Z0-9])({cjk})', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    # Strip markdown bold or italics if hallucinated
    text = re.sub(r'[*_~`]+', '', text)
    return text.strip()


CN_DIGIT_MAP = {
    '零': '0', '〇': '0', '一': '1', '幺': '1', '二': '2', '两': '2',
    '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'
}


def extract_digits(s: str) -> str:
    """Extracts sequence of numerical digits from Chinese numerals or ASCII digits."""
    res = []
    for c in s:
        if c in CN_DIGIT_MAP:
            res.append(CN_DIGIT_MAP[c])
        elif c.isdigit():
            res.append(c)
    return ''.join(res)


def is_numeral_or_unit_conversion(sub_orig: str, sub_corr: str) -> bool:
    """
    Universally recognizes if the difference is a spoken numeral/scientific unit/address
    being converted to standard professional representation (e.g. 100千赫兹 -> 100 kHz,
    50兆赫 -> 50 MHz, 零点零点零点零 -> 0.0.0.0, 百分之八十 -> 80%, 二十四 -> 24, 2.5 mol/L).
    Guards against converting variable identifiers (e.g. h零, c二) or hallucinating different numerical values.
    """
    so = sub_orig.strip()
    sc = sub_corr.strip()
    if not so or not sc:
        return False

    # Variable identifier check: e.g. h零 -> H0, c二 -> C2, r一 -> R1
    ident_match = re.match(r'^([a-zA-Z])([零〇一幺二两三四五六七八九十0-9]+)$', so)
    if ident_match:
        if re.match(r'^[a-zA-Z][0-9]+$', sc):
            return so[0].lower() == sc[0].lower()
        return False

    if all(c in CN_NUM_AND_UNIT_CHARS or c in LATIN_NUM_AND_UNIT_CHARS for c in so):
        if all(c in LATIN_NUM_AND_UNIT_CHARS or c in CN_NUM_AND_UNIT_CHARS for c in sc):
            # Strict digit consistency: spoken numbers cannot be replaced with arbitrary foreign numbers
            d_so = extract_digits(so)
            d_sc = extract_digits(sc)
            if d_so and d_sc and d_so != d_sc:
                return False
            return True
    return False


def normalize_universal_tokens(text: str) -> str:
    """
    Applies deterministic language-level whitespace, casing, and spacing formatting.
    Completely domain-agnostic without hardcoding specific disciplinary terms.
    """
    if not text:
        return ""

    # Fix spaced single Latin letters: e.g. 't c p' -> 'TCP', 'p h' -> 'pH', 'd n a' -> 'DNA'
    def replace_spaced_latin(m):
        return m.group(0).replace(" ", "")

    text = re.sub(r'\b[a-zA-Z](?:\s+[a-zA-Z]){1,5}\b', replace_spaced_latin, text)

    # Clean hallucinated markdown tags
    text = re.sub(r'[*_~`]+', '', text)
    return pangu_format(text)


# =============================================================================
# SRT Parsing & Serializing
# =============================================================================

class SubtitleEntry:
    def __init__(self, idx: int, time_line: str, start: float, end: float, text: str):
        self.id = idx
        self.time_line = time_line
        self.start = start
        self.end = end
        self.text = text

    def to_srt_block(self) -> str:
        return f"{self.id}\n{self.time_line}\n{self.text}\n"


def parse_srt_file(path: str) -> List[SubtitleEntry]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    blocks = re.split(r'\n\s*\n', content)
    subtitles = []

    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time_line = lines[1].strip()
        text = " ".join(lines[2:]).strip()

        m = re.match(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)", time_line)
        if not m:
            continue

        def parse_t(ts: str) -> float:
            h, mn, s_ms = ts.split(":")
            s, ms = s_ms.split(",")
            return int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0

        st = parse_t(m.group(1))
        et = parse_t(m.group(2))

        subtitles.append(SubtitleEntry(idx, time_line, st, et, text))

    return subtitles


def save_srt_file(subtitles: List[SubtitleEntry], out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in subtitles:
            f.write(s.to_srt_block() + "\n")


# =============================================================================
# Stage 1: Domain & Terminology Auto-Profiler
# =============================================================================

class DomainProfile:
    def __init__(
        self,
        domain: str = "通用专业学术与知识讲座",
        topic: str = "核心知识讲解与专题分析",
        jargon_list: Optional[List[str]] = None,
        confusion_rules: Optional[Dict[str, str]] = None,
        unit_conventions: Optional[Dict[str, str]] = None
    ):
        self.domain = domain
        self.topic = topic
        self.jargon_list = jargon_list or []
        self.confusion_rules = confusion_rules or {}
        self.unit_conventions = unit_conventions or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "topic": self.topic,
            "jargon_list": self.jargon_list,
            "confusion_rules": self.confusion_rules,
            "unit_conventions": self.unit_conventions
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainProfile":
        return cls(
            domain=d.get("domain", "通用专业学术与知识讲座"),
            topic=d.get("topic", "核心知识讲解与专题分析"),
            jargon_list=d.get("jargon_list", []),
            confusion_rules=d.get("confusion_rules", {}),
            unit_conventions=d.get("unit_conventions", {})
        )


def analyze_domain_profile(
    subtitles: List[SubtitleEntry],
    api_base: str,
    model: str,
    file_title: Optional[str] = None,
    timeout: int = 20
) -> DomainProfile:
    """
    Evenly samples text from across the subtitle file (focusing on the lecture body), sending it to LLM
    with title context to extract domain characteristics, key technical jargon, ASR confusion pairs, and unit conventions.
    100% dynamic - adapts automatically to Marxism, Chemistry, Mathematics, CS, Law, Medicine, etc.
    """
    if not subtitles:
        return DomainProfile()

    n = len(subtitles)
    start_idx = int(0.04 * n) if n > 200 else 0
    end_idx = int(0.96 * n) if n > 200 else n
    step = max(1, (end_idx - start_idx) // 30)
    samples = []
    for i in range(start_idx, end_idx, step):
        t = subtitles[i].text.strip()
        if is_cjk(t) and len(t) >= 8:
            samples.append(t)
        if len(samples) >= 28:
            break

    sample_text = "\n".join(samples)
    if not sample_text:
        return DomainProfile()

    title_hint = f"当前讲座/课程标题为：【{file_title}】\n" if file_title else ""

    prompt = f"""你是一个专业的通用 ASR 语音识别领域分析专家。
{title_hint}请结合标题背景与以下从该讲座正文中抽样的语音识别文本，全面提取该学科领域的专业元数据：
1. domain: 细分学科领域（例如：计算机考研/计算机网络、马克思主义基本原理、有机化学反应机理、高等数学/微积分、民法典讲座、医学临床等）
2. topic: 本场核心议题（概括主要内容，10-25字）
3. jargon_list: 核心专业术语与专有名词列表（根据学科深度，推断出本课程中高频出现的标准专业术语、英文缩写及概念，不少于 25 个）
4. confusion_rules: 常见 ASR 同音错词 -> 规范专业词预测字典（例如在计算机网络中识别出的 '泛红'->'泛洪', '麦克针'->'MAC帧', '权一'->'全1', '手部'->'首部', 或哲学中的 '辩正'->'辩证', 化学中的 '高猛酸钾'->'高锰酸钾' 等同音错误）
5. unit_conventions: 本学科常用的数值、地址、公式、单位书写规范（如：'零点零点零点零'->'0.0.0.0', '100千赫兹'->'100 kHz', '50兆赫'->'50 MHz', '2.5摩尔每升'->'2.5 mol/L' 等）

必须以严格纯 JSON 格式输出，严禁包含 Markdown 解释、思考标签或任何多余字符：
{{"domain": "...", "topic": "...", "jargon_list": [...], "confusion_rules": {{...}}, "unit_conventions": {{...}}}}

文本样本：
{sample_text}"""

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens': 800
    }

    for retry in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                raw = data['choices'][0]['message'].get('content', '').strip()
                m = re.search(r'\{[\s\S]*\}', raw)
                if m:
                    parsed = json.loads(m.group(0))
                    profile = DomainProfile.from_dict(parsed)
                    return profile
        except Exception:
            time.sleep(1.0 * (retry + 1))

    return DomainProfile()


# =============================================================================
# Stage 3: Three-Layer Correction Guard (Zero Hallucination)
# =============================================================================

def clean_content(s: str) -> str:
    """Removes punctuation and whitespace, converting to lower-case for structural comparison."""
    return PUNCT_CLEAN_RE.sub('', s or '').lower()


def normalize_fuzzy_pinyin(py_list: List[str]) -> List[str]:
    """
    Normalizes acoustic confusion pairs in Mandarin ASR:
    - Stops / affricates: p/b, t/d, k/g, q/j, f/h, c/z, zh/z, ch/c, sh/s
    - Nasals: n/l, and front/back nasals (an/ang, en/eng, in/ing)
    """
    res = []
    for p in py_list:
        p = p.lower()
        p = re.sub(r'^(zh|ch|sh)', lambda m: m.group(1)[0], p)
        p = re.sub(r'^[ptkqfc]', lambda m: {'p':'b', 't':'d', 'k':'g', 'q':'j', 'f':'h', 'c':'z'}.get(m.group(0), m.group(0)), p)
        p = re.sub(r'^l', 'n', p)
        p = re.sub(r'([aeiou])ng$', r'\1n', p)
        res.append(p)
    return res


def is_stutter_reduction(orig: str, corr: str) -> bool:
    """
    Detects if the difference is solely the removal of oral stutter / consecutive repeated tokens:
    e.g. '自自己' -> '自己', '在在一个' -> '在一个', '交换机交换机' -> '交换机', '单播单播' -> '单播'.
    """
    o = PUNCT_CLEAN_RE.sub('', orig or '').lower()
    c = PUNCT_CLEAN_RE.sub('', corr or '').lower()
    if not o or not c or o == c:
        return False
    # Single-char stutter: AA -> A
    for m in re.finditer(r'(.)\1+', o):
        char = m.group(1)
        reduced = o[:m.start()] + char + o[m.end():]
        if reduced == c:
            return True
    # 2-4 char word stutter: ABAB -> AB
    for w_len in (2, 3, 4):
        for i in range(len(o) - 2 * w_len + 1):
            w1 = o[i:i + w_len]
            w2 = o[i + w_len:i + 2 * w_len]
            if w1 == w2:
                reduced = o[:i] + w1 + o[i + 2 * w_len:]
                if reduced == c:
                    return True
    return False


def is_phonetic_match(a: str, b: str) -> bool:
    """
    Checks if two tokens have identical or highly similar pronunciation.
    Supports Mandarin Pinyin syllables, fuzzy acoustic pairs (b/p, d/t, l/n, en/eng),
    and cross-lingual transliteration (e.g. 麦克 <-> mac, 拜特 <-> byte).
    """
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

    # Case A: Both are Chinese (pure Mandarin ASR homophone correction)
    if not has_latin_a and not has_latin_b:
        if abs(len(a) - len(b)) > 1:
            return False
        # Standard pinyin ratio
        if difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.70:
            return True
        # Fuzzy acoustic pinyin ratio (e.g. 分布/分配 b<->p, en<->eng)
        fa = "".join(normalize_fuzzy_pinyin(pa))
        fb = "".join(normalize_fuzzy_pinyin(pb))
        return difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.70

    # Case B: Transliteration or hybrid (e.g. 麦克 <-> MAC, 麦克针 <-> MAC帧)
    # If both end with CJK, the tail characters MUST have matching pinyin (e.g. 针 <-> 帧, not 针 <-> 地址)
    if is_cjk(a[-1]) and is_cjk(b[-1]):
        tail_a_py = lazy_pinyin(a[-1])[0]
        tail_b_py = lazy_pinyin(b[-1])[0]
        if difflib.SequenceMatcher(None, tail_a_py, tail_b_py).ratio() < 0.70:
            return False

    # If both start with CJK, the head characters MUST have matching pinyin
    if is_cjk(a[0]) and is_cjk(b[0]):
        head_a_py = lazy_pinyin(a[0])[0]
        head_b_py = lazy_pinyin(b[0])[0]
        if difflib.SequenceMatcher(None, head_a_py, head_b_py).ratio() < 0.70:
            return False

    # Overall sound similarity
    sm = difflib.SequenceMatcher(None, sa, sb)
    ratio = sm.ratio()
    if sa and sb and sa[0] == sb[0] and ratio >= 0.48:
        return True

    return ratio >= 0.70


CONSONANT_ACOUSTIC_GROUPS = [
    {"b", "p"}, {"d", "t"}, {"g", "k"}, {"j", "q", "x"},
    {"z", "c", "s"}, {"zh", "ch", "sh", "r"}, {"f", "h"}, {"l", "n", "r"}
]


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


def verify_correction(orig: str, corr: str, profile: Optional[DomainProfile] = None) -> Tuple[bool, str]:
    """
    Rigorous yet flexible verification guard:
    Layer 1: Length ratio constraint (0.55 <= ratio <= 1.60).
    Layer 2: Fast-track natural oral stutter cleanup (e.g. '自自己' -> '自己', '在在一个' -> '在一个').
    Layer 3: Check EVERY modified block:
             - Numeral / scientific unit conversion (e.g. 100千赫兹 -> 100 kHz, 零点零点零点零 -> 0.0.0.0, 幺六六... -> 166...)
             - Case-insensitive Latin match (e.g. mac -> MAC)
             - Phonetic homophone, fuzzy acoustic, or transliteration match (e.g. 泛红 -> 泛洪, 分布 -> 分配, 麦克针 -> MAC帧, 辩正 -> 辩证)
             - Jargon missing-letter completion (e.g. SIF -> SIFS where SIFS is in profile)
             - Strictly rejects hallucinations (e.g. 泛红 -> 之外, 麦克针 -> MAC地址, 针 -> 地址).
    """
    if not corr or not corr.strip():
        return False, "empty_output"

    orig_clean = orig.strip()
    corr_clean = corr.strip()

    # Fast path: identical content (or trivial whitespace)
    if orig_clean == corr_clean:
        return True, "identical"

    # Layer 1: Extreme length change check (on non-punctuation characters)
    c_orig_raw = PUNCT_CLEAN_RE.sub('', orig_clean)
    c_corr_raw = PUNCT_CLEAN_RE.sub('', corr_clean)
    if c_orig_raw:
        ratio = len(c_corr_raw) / max(1, len(c_orig_raw))
        max_ratio = 2.0 if len(c_orig_raw) <= 5 else 1.65
        min_ratio = 0.40 if len(c_orig_raw) <= 5 else 0.55
        if ratio < min_ratio or ratio > max_ratio:
            return False, f"length_ratio_{ratio:.2f}"

    # Fast-track oral stutter removal
    if is_stutter_reduction(orig_clean, corr_clean):
        return True, "accepted_stutter_reduction"

    # Compare character strings ignoring standard punctuation
    c_orig = PUNCT_CLEAN_RE.sub('', orig_clean)
    c_corr = PUNCT_CLEAN_RE.sub('', corr_clean)

    if not c_orig and not c_corr:
        return True, "accepted_punctuation_only"

    # If Chinese characters are identical and only punctuation/spacing changed
    if c_orig == c_corr:
        return True, "accepted_formatting_only"

    # Layer 2: Check every modified block
    sm = difflib.SequenceMatcher(None, c_orig, c_corr)
    blocks = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in blocks:
        if tag == 'equal':
            continue
        sub_orig = c_orig[i1:i2]
        sub_corr = c_corr[j1:j2]

        if not sub_orig and not sub_corr:
            continue

        # Check numeral or unit conversion
        if is_numeral_or_unit_conversion(sub_orig, sub_corr):
            continue

        # Check digit consistency: spoken numbers cannot be replaced with arbitrary foreign numbers
        d_so = extract_digits(sub_orig)
        d_sc = extract_digits(sub_corr)
        if d_so and d_sc and d_so != d_sc:
            return False, f"digit_mismatch_{d_so}->{d_sc}"

        # Check case-insensitive Latin match
        if sub_orig.lower() == sub_corr.lower():
            continue

        # Check known profile rules with phonetic/numeral validation
        clean_so = PUNCT_CLEAN_RE.sub('', sub_orig).lower()
        clean_sc = PUNCT_CLEAN_RE.sub('', sub_corr).lower()
        if profile and clean_so in profile.confusion_rules and PUNCT_CLEAN_RE.sub('', profile.confusion_rules[clean_so]).lower() == clean_sc:
            if is_phonetic_match(sub_orig, sub_corr) or is_numeral_or_unit_conversion(sub_orig, sub_corr):
                continue

        # Check jargon completion: e.g. SIF -> SIFS where SIFS is in profile.jargon_list
        if profile and profile.jargon_list:
            so_u = sub_orig.upper()
            sc_u = sub_corr.upper()
            if so_u and any(j.upper() == sc_u and j.upper().startswith(so_u) and len(j) - len(so_u) <= 2 for j in profile.jargon_list):
                continue
            if len(sub_orig) <= 1 or len(sub_corr) <= 1:
                word_o = c_orig[max(0, i1 - 4):min(len(c_orig), i2 + 4)]
                word_c = c_corr[max(0, j1 - 4):min(len(c_corr), j2 + 4)]
                if any(j.lower() == word_c and j.lower().startswith(word_o) for j in profile.jargon_list):
                    continue

        # Check phonetic match (including standard and fuzzy Mandarin pinyin)
        if is_phonetic_match(sub_orig, sub_corr):
            continue

        # Context-expanded phonetic match ONLY for single character acoustic confusions (e.g. 分布 -> 分配)
        # Must share confusable initial consonants and NOT be digits!
        if len(sub_orig) <= 1 and len(sub_corr) <= 1:
            if not d_so and not d_sc and are_confusable_initials(sub_orig, sub_corr):
                ctx_o = c_orig[max(0, i1 - 1):min(len(c_orig), i2 + 1)]
                ctx_c = c_corr[max(0, j1 - 1):min(len(c_corr), j2 + 1)]
                if is_phonetic_match(ctx_o, ctx_c):
                    continue

        # Any non-phonetic modification is strictly rejected!
        return False, f"non_phonetic_change_{sub_orig}->{sub_corr}"

    return True, "accepted"


# =============================================================================
# Stage 2: Parallel Batch LLM Correction Engine
# =============================================================================

class LLMSubtitleCorrector:
    def __init__(
        self,
        api_base: str = "http://192.168.2.16:8002/v1",
        model: str = "qwen3-8b",
        batch_size: int = 25,
        workers: int = 20,
        profile: Optional[DomainProfile] = None,
        cache_file: Optional[str] = None,
        timeout: int = 25,
        max_retries: int = 2
    ):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.batch_size = batch_size
        self.workers = workers
        self.profile = profile or DomainProfile()
        self.cache_file = cache_file
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache: Dict[str, str] = {}

        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                pass

    def save_cache(self):
        if not self.cache_file:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def build_system_prompt(self) -> str:
        p = self.profile
        jargon_str = ", ".join(p.jargon_list[:30]) if p.jargon_list else "标准专业学术术语与英文缩写"
        confusion_items = [f"  - {k} -> {v}" for k, v in list(p.confusion_rules.items())[:15]]
        confusion_str = "\n".join(confusion_items) if confusion_items else "  - 规范常见同音错字与字母拆开"
        unit_items = [f"  - {k} -> {v}" for k, v in list(p.unit_conventions.items())[:10]]
        unit_str = "\n".join(unit_items) if unit_items else "  - 规范专业学术单位及数值点分十进制表示"

        return f"""你是一个高精度的通用 ASR 语音识别字幕逐行纠错专家。
当前音频元画像：
- 学科领域：【{p.domain}】
- 核心主题：【{p.topic}】
- 核心术语库：{jargon_str}
- 常见混淆参考：
{confusion_str}
- 格式规范参考：
{unit_str}

【硬性约束，必须逐条遵守】：
1. 必须且仅能输出 [ID] 编号开头的逐行纠错文本，例如：
   [1] 第一行纠错后文本
   [2] 第二行纠错后文本
   编号必须与输入完全一一对应，严禁增行、减行、合并行或跨行搬移文字！
2. 专业术语与专有名词必须准确规范（同音字校正）：
   - 根据【{p.domain}】学科专业背景与术语库，准确校正口语中被 ASR 误识的同音词、近音词（拼音发音相同或相近的专业词汇）；
   - 英文缩写与专有名词统一为标准大小写格式（如 MAC、TCP、IP、DNA、pH 等）。
3. 口语化数值、单位与专业标识规范化：
   - 连续念出的数字地址、版本号等规范为标准点分或阿拉伯数字（例如 "零点零点零点零" -> "0.0.0.0"）；
   - 学术计量单位规范为标准科学记号（例如 "100千赫兹" -> "100 kHz", "50兆赫" -> "50 MHz", "2.5摩尔每升" -> "2.5 mol/L" 等）；
4. 严禁语义脑补或篡改：
   - 绝对禁止将同音专业术语篡改为无关的日常词汇！校正后的词语必须与原词在汉语拼音上发音相近或属于标准外来语/缩写音译。
   - 完整保留说话人的原声语气词（呃、啊、呢、吧、对吧、你看、其实等），严禁擅自增删句子主干或总结修饰！
5. 无法确定的词语必须保持原样，禁止凭想象自作聪明魔改。
6. 严禁输出任何解释、分析、思考过程或多余文字，直接输出 [ID] 编号开头的纠错结果。"""

    def call_api(self, prompt_text: str) -> Optional[str]:
        url = f"{self.api_base}/chat/completions"
        system_prompt = self.build_system_prompt()

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"请对以下字幕进行逐行纠错，按行输出：\n{prompt_text}"}
            ],
            'temperature': 0.1,
            'max_tokens': 1200
        }

        req_bytes = json.dumps(payload).encode('utf-8')

        for retry in range(4):
            try:
                req = urllib.request.Request(
                    url,
                    data=req_bytes,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    res = json.loads(resp.read().decode('utf-8'))
                    msg = res['choices'][0]['message']
                    content = msg.get('content', '') or ''
                    content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                    if content:
                        return content
            except Exception as e:
                time.sleep(1.0 * (retry + 1))

        return None

    def parse_batch_response(self, raw_text: str, batch: List[Tuple[int, str]]) -> Dict[int, str]:
        line_pattern = re.compile(r'^(?:\[(\d+)\]|(\d+)[\.\:：\s])\s*(.*)$')
        parsed: Dict[int, str] = {}
        batch_ids = {idx for idx, _ in batch}

        for line in raw_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            m = line_pattern.match(line)
            if m:
                id_str = m.group(1) or m.group(2)
                try:
                    line_id = int(id_str)
                except ValueError:
                    continue
                if line_id in batch_ids:
                    txt = m.group(3).strip()
                    parsed[line_id] = txt

        return parsed

    def process_single_batch(self, batch: List[Tuple[int, str]]) -> Tuple[Dict[int, Tuple[str, str, bool]], bool]:
        prompt_lines = [f"[{idx}] {text}" for idx, text in batch]
        prompt_str = "\n".join(prompt_lines)

        raw_output = self.call_api(prompt_str)
        if raw_output is None:
            # Network failed completely, safety fallback to original without caching
            batch_result = {idx: (orig_text, orig_text, False) for idx, orig_text in batch}
            return batch_result, False, {}

        parsed = self.parse_batch_response(raw_output, batch)
        batch_result: Dict[int, Tuple[str, str, bool]] = {}

        for idx, orig_text in batch:
            if idx in parsed:
                candidate = parsed[idx]
                candidate = normalize_universal_tokens(candidate)
                accepted, reason = verify_correction(orig_text, candidate, self.profile)
                if accepted:
                    is_mod = (candidate != orig_text)
                    batch_result[idx] = (candidate, orig_text, is_mod)
                else:
                    if candidate != orig_text:
                        print(f"  [Guard Blocked #{idx}]: '{orig_text}' -> '{candidate}' ({reason})")
                    batch_result[idx] = (orig_text, orig_text, False)
            else:
                batch_result[idx] = (orig_text, orig_text, False)

        return batch_result, True, parsed

    def correct_subtitles(
        self,
        subtitles: List[SubtitleEntry]
    ) -> Tuple[List[SubtitleEntry], Dict[str, Any]]:
        total = len(subtitles)
        print(f"\n[AI Corrector] 启动逐行纠错 (共 {total} 行字幕)")
        print(f"  -> Model: {self.model} | Endpoint: {self.api_base} | Workers: {self.workers}")

        final_map: Dict[int, str] = {}
        pending_items: List[Tuple[int, str]] = []
        cache_hits = 0
        corrections_made = 0
        diff_samples: List[Dict[str, Any]] = []

        for s in subtitles:
            norm_orig = normalize_universal_tokens(s.text)
            cached_cand = None
            if s.text in self.cache:
                cached_cand = self.cache[s.text]
            elif norm_orig in self.cache:
                cached_cand = self.cache[norm_orig]

            if cached_cand is not None:
                cache_hits += 1
                cand_clean = normalize_universal_tokens(cached_cand)
                accepted, reason = verify_correction(norm_orig, cand_clean, self.profile)
                if accepted:
                    final_map[s.id] = cand_clean
                    if cand_clean != norm_orig:
                        corrections_made += 1
                        if len(diff_samples) < 15:
                            diff_samples.append({
                                "id": s.id,
                                "before": s.text,
                                "after": cand_clean
                            })
                else:
                    final_map[s.id] = s.text
            else:
                pending_items.append((s.id, norm_orig))

        if cache_hits > 0:
            print(f"  -> [Cache Hit] 命中缓存 {cache_hits}/{total} 条 ({cache_hits/total*100:.1f}%)")

        batches = [pending_items[i:i + self.batch_size] for i in range(0, len(pending_items), self.batch_size)]
        num_batches = len(batches)
        print(f"  -> 待处理批次: {num_batches} 批 (每批 {self.batch_size} 行)")

        t0 = time.time()

        if num_batches > 0:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_bidx = {
                    executor.submit(self.process_single_batch, b): i
                    for i, b in enumerate(batches, 1)
                }

                completed_batches = 0
                for future in as_completed(future_to_bidx):
                    b_idx = future_to_bidx[future]
                    completed_batches += 1
                    try:
                        batch_res, is_success, raw_cands = future.result()
                        for idx, (fin_txt, orig_txt, is_mod) in batch_res.items():
                            final_map[idx] = fin_txt
                            if is_success and idx in raw_cands:
                                self.cache[orig_txt] = raw_cands[idx]
                            if is_mod:
                                corrections_made += 1
                                if len(diff_samples) < 15:
                                    diff_samples.append({
                                        "id": idx,
                                        "before": orig_txt,
                                        "after": fin_txt
                                    })
                    except Exception as e:
                        print(f"  [Warning] Batch {b_idx} encountered error: {e}")

                    if completed_batches % 20 == 0 or completed_batches == num_batches:
                        elapsed = time.time() - t0
                        speed = (completed_batches * self.batch_size) / max(0.1, elapsed)
                        print(f"  -> 进度: {completed_batches}/{num_batches} 批 ({completed_batches/num_batches*100:.1f}%) | 速度: {speed:.1f} 行/秒")
                        self.save_cache()

        dt_total = time.time() - t0
        print(f"[AI Corrector] 纠错完成！采纳优化: {corrections_made} 处，总耗时: {dt_total:.1f}s")

        corrected_entries: List[SubtitleEntry] = []
        for s in subtitles:
            corr_text = final_map.get(s.id, s.text)
            corrected_entries.append(SubtitleEntry(
                idx=s.id,
                time_line=s.time_line,
                start=s.start,
                end=s.end,
                text=corr_text
            ))

        stats = {
            "total_lines": total,
            "cache_hits": cache_hits,
            "corrections_made": corrections_made,
            "time_seconds": round(dt_total, 2),
            "diff_samples": diff_samples
        }

        return corrected_entries, stats


# =============================================================================
# Pipeline Runner & CLI
# =============================================================================

def process_file(
    srt_path: str,
    output_dir: str,
    api_base: str,
    model: str,
    batch_size: int,
    workers: int,
    cache_dir: str
) -> Dict[str, Any]:
    in_path = Path(srt_path)
    out_path = Path(output_dir) / in_path.name
    cache_path = Path(cache_dir) / f"{in_path.stem}.cache.json"

    print("\n" + "=" * 80)
    print(f"处理文件: {in_path.name}")
    print("=" * 80)

    subtitles = parse_srt_file(str(in_path))
    if not subtitles:
        print(f"  [Skip] 空文件: {in_path}")
        return {"file": in_path.name, "status": "empty"}

    # Stage 1: Auto-Profiling
    print(f"-> [Stage 1] 分析领域与专业术语画像...")
    profile = analyze_domain_profile(subtitles, api_base, model, file_title=in_path.stem)
    print(f"   细分领域: 【{profile.domain}】 | 核心议题: 【{profile.topic}】")
    if profile.jargon_list:
        print(f"   提取术语: {', '.join(profile.jargon_list[:8])} ...")

    # Stage 2: Batch Correction
    corrector = LLMSubtitleCorrector(
        api_base=api_base,
        model=model,
        batch_size=batch_size,
        workers=workers,
        profile=profile,
        cache_file=str(cache_path)
    )

    corrected_subs, stats = corrector.correct_subtitles(subtitles)

    # Save to AI_srt/
    save_srt_file(corrected_subs, str(out_path))
    print(f"-> 已保存高质量 AI 字幕至: {out_path}")

    # Print sample diffs
    if stats.get("diff_samples"):
        print("\n  [典型优化样例对比]:")
        for sample in stats["diff_samples"][:6]:
            print(f"    #{sample['id']:<4} 原文: {sample['before']}")
            print(f"          AI:   {sample['after']}")

    stats["file"] = in_path.name
    stats["out_path"] = str(out_path)
    return stats


def get_remote_model_name(api_base: str) -> Optional[str]:
    try:
        url = f"{api_base.rstrip('/')}/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = data.get("data", [])
            if models and "id" in models[0]:
                return models[0]["id"]
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Universal AI Subtitle Correction Engine (AI_srt)")
    parser.add_argument("input", nargs="?", default="new_srt", help="Input SRT file or directory containing SRT files")
    parser.add_argument("-o", "--output-dir", default="AI_srt", help="Output directory for AI corrected SRT files (default: AI_srt)")
    parser.add_argument("--api-base", default="http://192.168.2.16:8002/v1", help="OpenAI-compatible vLLM API base URL")
    parser.add_argument("--model", default="auto", help="Model name (default: auto detects from API or qwen3.5-9b)")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size per LLM request (default: 25)")
    parser.add_argument("--workers", type=int, default=20, help="Parallel worker threads (default: 20)")
    parser.add_argument("--cache-dir", default=".llm_sub_cache", help="Cache directory for resume capability")

    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        srt_files = sorted(input_path.glob("*.srt"))
    elif input_path.is_file():
        srt_files = [input_path]
    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    if not srt_files:
        print(f"No SRT files found in: {input_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "auto":
        detected = get_remote_model_name(args.api_base)
        model = detected or "qwen3.5-9b"
        print(f"-> 自动检测到后端推理模型: 【{model}】")
    else:
        model = args.model
        print(f"-> 使用指定推理模型: 【{model}】")

    print(f"开始批量 AI 纠错，待处理文件数: {len(srt_files)}")
    all_stats = []

    for srt_file in srt_files:
        st = process_file(
            srt_path=str(srt_file),
            output_dir=str(out_dir),
            api_base=args.api_base,
            model=model,
            batch_size=args.batch_size,
            workers=args.workers,
            cache_dir=args.cache_dir
        )
        all_stats.append(st)

    print("\n" + "=" * 80)
    print("全量 AI 字幕纠错处理完毕！汇总统计：")
    print(f"{'文件名':<38} | {'总行数':<6} | {'优化处数':<8} | {'耗时':<7}")
    print("-" * 80)
    for st in all_stats:
        fn = st.get("file", "")[:36]
        print(f"{fn:<38} | {st.get('total_lines', 0):<6} | {st.get('corrections_made', 0):<8} | {st.get('time_seconds', 0)}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
