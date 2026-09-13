#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR 字幕文本确定性规范化（通用规则，不依赖大模型，纠错前后均稳定生效）

解决三类**与领域无关**的高发问题：
1. 英文缩写被逐字母读开：`t c p` → `TCP`、`a p` → `AP`；
   规则本身**不依赖任何领域词表**（英文单字母串一律合并并大写），
   词表只用于还原「按惯例带分隔符」的写法（如 TCP/IP、A/B）。
2. 中文数字读法 → 阿拉伯数字：`一百二十八` → `128`、`三点一四` → `3.14`、`八零二点幺幺` → `802.11`。
3. 排版：数字与单位、中文与英文/数字之间补空格，多余空白归一。

领域专有术语**不写死在代码里**，两个通用入口：
  a) Stage 1 自动领域画像产出的术语库 / 易错规则（见 corrector.AutoProfiler），进提示词；
  b) 外部词表 `service/glossary.json`（可选，用户自行维护），本模块启动时加载并生效。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

# 通用缩写写法表：仅收录「按惯例含分隔符」等规则无法推出的写法（键为去掉分隔符的大写形式）
# 领域专有缩写不要写在这里，放到 service/glossary.json
ACRONYM_CANON = {
    "TCPIP": "TCP/IP",
    "ACDC": "AC/DC",
    "AB": "A/B",
    "YNY": "Y/N",
    "PL": "P/L",
    # 通用计算机 / 办公 / 多媒体缩写
    "HTTP": "HTTP",
    "HTTPS": "HTTPS",
    "FTP": "FTP",
    "DNS": "DNS",
    "DHCP": "DHCP",
    "NAT": "NAT",
    "VPN": "VPN",
    "CPU": "CPU",
    "GPU": "GPU",
    "RAM": "RAM",
    "ROM": "ROM",
    "OS": "OS",
    "DB": "DB",
    "API": "API",
    "SDK": "SDK",
    "IDE": "IDE",
    "PDF": "PDF",
    "URL": "URL",
    "URI": "URI",
    "USB": "USB",
    "HDMI": "HDMI",
    "SSD": "SSD",
    "HDD": "HDD",
    "LCD": "LCD",
    "LED": "LED",
    "GPS": "GPS",
    "NFC": "NFC",
    "AI": "AI",
    "ML": "ML",
    "DL": "DL",
    "CV": "CV",
    "NLP": "NLP",
    "ID": "ID",
    "OK": "OK",
}

# ── 可选外部词表：service/glossary.json ──
# {
#   "acronyms": {"CSMACA": "CSMA/CA"},     // 规则无法推出的缩写写法
#   "terms":    {"社恐": "时隙"}            // 领域专有同音错词（整词替换）
# }
_GLOSSARY_FILE = Path(__file__).with_name("glossary.json")
_glossary_cache: Dict[str, Any] = {}
_glossary_mtime: float = -1.0


def load_glossary(force: bool = False) -> Dict[str, Any]:
    """加载（或热更新）外部词表；文件不存在时返回空表"""
    global _glossary_cache, _glossary_mtime
    try:
        mtime = _GLOSSARY_FILE.stat().st_mtime if _GLOSSARY_FILE.exists() else -1.0
    except OSError:
        mtime = -1.0
    if force or mtime != _glossary_mtime:
        _glossary_mtime = mtime
        data: Dict[str, Any] = {"acronyms": {}, "terms": {}}
        if mtime > 0:
            try:
                raw = json.loads(_GLOSSARY_FILE.read_text(encoding="utf-8"))
                data["acronyms"] = {str(k).upper(): str(v) for k, v in (raw.get("acronyms") or {}).items()}
                data["terms"] = {str(k): str(v) for k, v in (raw.get("terms") or {}).items()}
            except Exception as e:
                print(f"[Normalize] 词表 {_GLOSSARY_FILE.name} 解析失败，已忽略: {e}")
        _glossary_cache = data
        ACRONYM_CANON.update(data["acronyms"])
    return _glossary_cache


def apply_glossary(text: str) -> str:
    """应用用户词表（领域专有同音错词整词替换）"""
    glossary = load_glossary()
    for wrong, right in glossary.get("terms", {}).items():
        if wrong and wrong in text:
            text = text.replace(wrong, right)
    return text

# 中文数字 → 阿拉伯数字（逐位读法）
_DIGIT_MAP = {
    "零": "0", "〇": "0", "一": "1", "幺": "1", "二": "2", "两": "2",
    "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}
_DIGIT_CHARS = "".join(_DIGIT_MAP.keys())
_CN_NUM_CHARS = "零〇一二三四五六七八九十百千万亿两幺"
# 中文数量词后紧跟的单位（含常见英文单位 / 缩写单位）
_UNITS = (
    "微秒|毫秒|纳秒|皮秒|秒|分钟|小时|字节|比特|位|帧|兆|吉|赫兹|"
    "米|公里|千米|厘米|毫米|瓦|伏|安|欧|分贝|倍|"
    "Mbps|Kbps|Gbps|MB|KB|GB|TB|Bit|bit|Byte|byte|Hz|Hz|KHz|MHz|GHz|"
    "ms|us|ns|dB|W|V|A|B"
)

_SPACED_LETTERS = re.compile(r"(?<![A-Za-z0-9])([A-Za-z](?:\s+[A-Za-z]){1,11})(?![A-Za-z0-9])")
_GLUED_ACRONYM = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{3,8})(?![A-Za-z0-9])")
_DOTTED_NUMBER = re.compile(rf"[{_DIGIT_CHARS}]+(?:点[{_DIGIT_CHARS}]+)+")
_DIGIT_RUN = re.compile(
    r"(?<![" + _CN_NUM_CHARS + r"])[" + _DIGIT_CHARS + r"]{3,}(?![" + _CN_NUM_CHARS + r"])"
)
_CN_NUM_WITH_UNIT = re.compile(rf"([{_CN_NUM_CHARS}]+)(?=\s*(?:{_UNITS}))", re.IGNORECASE)
_CN_NUM_PURE = re.compile(rf"([{_CN_NUM_CHARS}]+)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _cjk_adjacent(text: str, start: int, end: int, window: int = 3) -> bool:
    """判断匹配片段左右是否紧邻中文（用于区分「中文讲课里的缩写」与「英文歌词原文」）"""
    left = text[max(0, start - window):start]
    right = text[end:end + window]
    return bool(_CJK_RE.search(left + right))


def _cn_to_arabic(chinese: str) -> str:
    """中文数字转阿拉伯数字（支持 一百二十八 / 二百五十 / 一千零二十四 / 十）"""
    if not chinese:
        return chinese
    if all(ch in _DIGIT_MAP for ch in chinese):
        return "".join(_DIGIT_MAP[ch] for ch in chinese)

    total, section, number = 0, 0, 0
    unit_map = {"十": 10, "百": 100, "千": 1000}
    big_map = {"万": 10000, "亿": 100000000}
    for ch in chinese:
        if ch in _DIGIT_MAP:
            number = int(_DIGIT_MAP[ch])
        elif ch in unit_map:
            section += (number or 1) * unit_map[ch]
            number = 0
        elif ch in big_map:
            section = (section + number) * big_map[ch]
            total += section
            section, number = 0, 0
        else:
            return chinese  # 含未知字符，保持原样
    result = total + section + number
    return str(result) if result > 0 else chinese


def collapse_spaced_acronyms(text: str) -> str:
    """合并被逐字母读开的英文缩写：`c s m a c a` → `CSMA/CA`、`T C P` → `TCP`"""
    def _repl(match: re.Match) -> str:
        raw = match.group(1)
        letters = [c for c in raw if c.isalpha()]
        if len(letters) < 2:
            return raw
        joined = "".join(letters).upper()
        canon = ACRONYM_CANON.get(joined) or ACRONYM_CANON.get(joined.replace("/", ""))
        if canon:
            return canon
        # 仅在中文语境（左右紧邻中文字符）中合并：中文讲课里被读开的字母串基本都是缩写；
        # 英文原文（如歌词里的 "I m a ..."）不做合并，避免破坏原句。
        if _cjk_adjacent(text, match.start(), match.end()):
            return joined
        return raw

    text = _SPACED_LETTERS.sub(_repl, text)

    # 已连写但缺分隔符的缩写：CSMACD → CSMA/CD
    def _glued(match: re.Match) -> str:
        word = match.group(1)
        canon = ACRONYM_CANON.get(word.upper())
        return canon if canon else word

    return _GLUED_ACRONYM.sub(_glued, text)


def normalize_cn_numbers(text: str) -> str:
    """中文数字读法 → 阿拉伯数字（点分读法 / 纯数字串 / 带单位数量词）"""
    # 1) 点分读法：八零二点幺幺 → 802.11、幺六八点幺幺 → 168.11
    def _dotted(match: re.Match) -> str:
        return "".join(_DIGIT_MAP.get(c, ".") for c in match.group(0))

    text = _DOTTED_NUMBER.sub(_dotted, text)

    # 2) 三位以上纯数字读法：二八二 → 282
    def _digits(match: re.Match) -> str:
        return "".join(_DIGIT_MAP[c] for c in match.group(0))

    text = _DIGIT_RUN.sub(_digits, text)

    # 3) 带单位的数量词：一百二十八微秒 → 128微秒
    def _with_unit(match: re.Match) -> str:
        return _cn_to_arabic(match.group(1))

    return _CN_NUM_WITH_UNIT.sub(_with_unit, text)


def pangu_spacing(text: str) -> str:
    """中英文/数字之间补空格，并归一多余空格"""
    cjk = r"[\u4e00-\u9fff]"
    text = re.sub(rf"({cjk})([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(rf"([A-Za-z0-9])({cjk})", r"\1 \2", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_technical_text(text: str) -> str:
    """完整规范化流程（清除格式符 → 用户词表 → 缩写 → 数字 → 排版）"""
    if not text:
        return text
    # 清理残留的 markdown 加粗/斜体/反引号等标记
    text = re.sub(r'[*_`~]+', '', text).strip()
    original = text
    text = apply_glossary(text)
    text = collapse_spaced_acronyms(text)
    text = normalize_cn_numbers(text)
    text = pangu_spacing(text)
    if text != original:
        print(f"[Normalize] {original} -> {text}")
    return text
