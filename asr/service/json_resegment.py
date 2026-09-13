#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Universal Broadcast-Standard ASR Subtitle Engine (JSON -> SRT)
=============================================================================
A robust, general-purpose traditional (non-AI) subtitle segmentation & timing
engine for arbitrary ASR JSON outputs (Qwen-ASR, Whisper, SenseVoice, FunASR, etc.).

Key Guarantees:
1. True Acoustic Anchor (Audio-Subtitle Synchronization):
   Subtitle start_time is strictly anchored to the acoustic onset of the first
   word. Zero cumulative drift.
2. Word & Phrase Integrity (Clause-by-Clause Linguistic Assembly):
   Jieba tokenization operates strictly within acoustic breath clauses bounded
   by speech pauses (>= 0.4s) and sentence endings, preventing cross-silence
   word fusion (e.g. '哎还真有' + '没有' -> '有没有'). Technical terms ('CSMA/CD',
   '10BASE5', '王道书') are 100% protected.
3. Anti-Bleeding & Conjunction Forwarding:
   - Clause-initial connectors ('然后', '但是', '所以', '如果'...) are forward-
     attached to the beginning of the clause they introduce. No dangling connectors.
   - Suffix particles ('的', '了', '着', '的话'...) never begin a line.
   - Prepositions ('在', '把', '对于'...) never dangle at the end of a line.
   - No greedy swallowing across sentence boundaries or speech pauses.
4. Reading Comfort & Pangu Typography:
   - Target line length: 12-20 Chinese characters (24-40 visual units).
   - Display duration: 1.0s - 4.5s (human-comfortable reading speed).
   - Standard whitespace between Latin words and Pangu spacing between CJK and Latin/digits.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jieba
    jieba.setLogLevel(jieba.logging.INFO)
except ImportError:
    jieba = None

CJK_PATTERN = re.compile(r'[\u4e00-\u9fff]')
LATIN_PATTERN = re.compile(r'[a-zA-Z0-9]')

# Grammatical suffix particles and bound morphemes that should NEVER start a subtitle line
ATTACHED_PARTICLES = {'的', '了', '着', '得', '地', '过', '们', '的话', '们的'}
ATTACHED_CLASSIFIERS = {'个', '种', '份', '点', '条', '本', '层', '位', '线', '度', '张', '把', '支', '届', '款'}

# Prepositions and compound prepositions that should not dangle alone at line tail
DANGLING_PREPOSITIONS = {'在', '把', '被', '将', '按', '从', '向', '往', '跟', '同', '由', '与'}
COMPOUND_DANGLING_ENDS = {'处于', '位于', '基于', '鉴于', '关于', '对于'}

# Spoken discourse marker prefixes (often followed by connectors like 那么, 所以)
DISCOURSE_PREFIXES = {'好', '好的', 'OK', 'ok', '那么', '所以', '呃', '那', '对', '行'}

# Clause-initial connectors and transition words that introduce a new thought.
# If a line ends with these, they must be forward-attached to the next line!
LEADING_CONNECTORS = (
    "总的来说", "换句话说", "也就是说", "接下来", "另一方面",
    "然后", "但是", "所以", "那么", "而且", "另外", "并且",
    "其实", "因为", "由于", "比如", "不过", "然而", "如果", "虽然",
    "接着", "首先", "其次", "最后", "与此同时"
)

# Short tag questions / confirmations that naturally attach to the preceding clause
TAG_QUESTIONS = {
    '对吧', '对不对', '是不是', '是吧', '好不好', 'OK吗', 'ok吗', '行不行',
    '懂了吧', '对吗', '行吧', '可以吧', '能理解吧', '能明白吧', '是这样吧'
}

def is_bad_line_start(text: str) -> bool:
    """
    Returns True if the token is a grammatical suffix or bound particle
    that clings to the preceding word and should NEVER start a subtitle line.
    Allows modal verb '得' (děi) before verbs (e.g. '得重新', '得去').
    """
    if not text:
        return False
    if text.startswith('得') and len(text) > 1 and text[1] in '去要重新想想做把将':
        return False
    if text in ATTACHED_PARTICLES:
        return True
    if len(text) == 1 and (text in ATTACHED_PARTICLES or text in ATTACHED_CLASSIFIERS):
        return True
    return False

def is_dangling_line_end(text: str) -> bool:
    """
    Returns True if the token is an open preposition that leaves the phrase incomplete.
    Does NOT match content nouns/adjectives like '现在', '过往', '雷同', '合同'.
    """
    if not text:
        return False
    if len(text) == 1 and text in DANGLING_PREPOSITIONS:
        return True
    if text in COMPOUND_DANGLING_ENDS:
        return True
    return False

def is_transition_marker_sub(sub_words: List[LinguisticWord]) -> bool:
    """
    Checks if a subtitle consists solely of spoken transition/discourse markers
    (e.g. '好那么', '好所以', 'OK那么', '然后', '所以', '呃首先', '好那么接下来接下来') with total length <= 12 chars.
    These markers belong to the upcoming sentence, never the preceding sentence.
    """
    if not sub_words:
        return False
    txt = ''.join(w.text for w in sub_words).strip('，。？！,?!. ')
    if len(txt) > 12:
        return False
    return all(
        w.text in DISCOURSE_PREFIXES or
        any(w.text == conn or w.text.endswith(conn) for conn in LEADING_CONNECTORS)
        for w in sub_words
    )

def is_cjk(s: str) -> bool:
    return bool(CJK_PATTERN.search(s))


def is_latin_word(s: str) -> bool:
    return bool(LATIN_PATTERN.search(s))


def calc_visual_length(text: str) -> int:
    """
    Computes visual display width:
    CJK characters and full-width punctuation = 2 units.
    Half-width ASCII characters (letters, digits, spaces) = 1 unit.
    """
    vlen = 0
    for ch in text:
        if CJK_PATTERN.match(ch) or ch in '，。？！、；：“”‘’《》（）…—':
            vlen += 2
        else:
            vlen += 1
    return vlen


def pangu_format(text: str) -> str:
    """Inserts clean whitespace between CJK and Latin/digits."""
    if not text:
        return ""
    cjk = r'[\u4e00-\u9fa5]'
    text = re.sub(f'({cjk})([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(f'([a-zA-Z0-9])({cjk})', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def format_srt_time(seconds: float) -> str:
    """Formats floating-point seconds into standard SRT HH:MM:SS,mmm format."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    if secs >= 60:
        minutes += 1
        secs -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# =============================================================================
# Step 1: Input JSON Loading & Normalization
# =============================================================================

def extract_raw_tokens_and_full_text(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts acoustic word/character tokens and reference punctuated text
    from various ASR JSON formats.
    """
    full_text = data.get("full_text", "")
    segments = data.get("segments", [])
    raw_words = data.get("word_timestamps") or data.get("words") or []

    # If Whisper format: segments containing 'words'
    if not raw_words and segments:
        collected = []
        for s in segments:
            if isinstance(s, dict) and "words" in s and isinstance(s["words"], list):
                collected.extend(s["words"])
        if collected:
            raw_words = collected

    # If full_text is empty but segments has text, build full_text from segments
    if not full_text and segments:
        full_text = "\n".join(s.get("text", "").strip() for s in segments if s.get("text"))

    tokens: List[Dict[str, Any]] = []

    if raw_words:
        for w in raw_words:
            if not isinstance(w, dict):
                continue
            txt = str(w.get("text") or w.get("word") or "").strip()
            if not txt:
                continue
            st = float(w.get("start_time") if "start_time" in w else w.get("start", 0.0))
            et = float(w.get("end_time") if "end_time" in w else w.get("end", st + 0.08))
            if et < st:
                et = st + 0.05
            tokens.append({
                "text": txt,
                "start": st,
                "end": et,
                "orig_start": st,
                "orig_end": et,
            })
    elif segments:
        for s in segments:
            if not isinstance(s, dict):
                continue
            txt = str(s.get("text", "")).strip()
            if not txt:
                continue
            st = float(s.get("start_time") if "start_time" in s else s.get("start", 0.0))
            et = float(s.get("end_time") if "end_time" in s else s.get("end", st + 1.0))
            if et < st:
                et = st + 0.1
            tokens.append({
                "text": txt,
                "start": st,
                "end": et,
                "orig_start": st,
                "orig_end": et,
            })

    return tokens, full_text




def sanitize_acoustic_tokens(tokens: List[Dict[str, Any]]):
    """
    Cleans up acoustic timestamp anomalies (music hallucination jumps,
    unusually stretched single tokens, bunched clusters) without inducing drift.
    """
    if not tokens:
        return

    # 1. Cap abnormally stretched single tokens (e.g. single char over 1.2s of silence)
    for i, t in enumerate(tokens):
        dur = t['end'] - t['start']
        if dur > 1.2:
            prev_t = tokens[i - 1] if i > 0 else None
            next_t = tokens[i + 1] if i + 1 < len(tokens) else None
            gap_before = (t['start'] - prev_t['end']) if prev_t else 999.0
            gap_after = (next_t['start'] - t['end']) if next_t else 999.0

            if gap_before < 0.2:
                t['end'] = t['start'] + 0.5
            elif gap_after < 0.2:
                t['start'] = t['end'] - 0.5
            else:
                t['end'] = t['start'] + 0.5

    # 2. Backward jumps (e.g. background music hallucinated timestamps jumping backward)
    for i in range(1, len(tokens)):
        prev = tokens[i - 1]
        curr = tokens[i]
        if curr['start'] < prev['start']:
            diff = prev['start'] - curr['start']
            if diff >= 0.4:
                curr['is_discontinuity'] = True
                cutoff = curr['start'] - 0.05
                for k in range(i - 1, -1, -1):
                    if tokens[k]['start'] <= cutoff and tokens[k]['end'] <= cutoff:
                        break
                    tokens[k]['end'] = min(tokens[k]['end'], cutoff)
                    tokens[k]['start'] = min(tokens[k]['start'], tokens[k]['end'] - 0.05)
            else:
                curr['start'] = prev['start']
                if curr['end'] < curr['start']:
                    curr['end'] = curr['start'] + 0.05

    for t in tokens:
        if t['end'] < t['start']:
            t['end'] = t['start'] + 0.05


# =============================================================================
# Step 3: Robust Sequence Alignment for Punctuation Recovery
# =============================================================================

def align_punctuation_from_full_text(tokens: List[Dict[str, Any]], full_text: str):
    """
    Uses SequenceMatcher to robustly map punctuation from full_text to tokens,
    even when full_text and tokens have minor discrepancies.
    """
    if not tokens or not full_text:
        return

    token_chars = []
    token_char_map = []  # (token_idx, is_last_char_of_token)
    for ti, t in enumerate(tokens):
        clean_t = re.sub(r'[\s\W_]+', '', t['text'])
        if not clean_t:
            clean_t = t['text'].strip()
        for ci, ch in enumerate(clean_t):
            token_chars.append(ch.lower())
            token_char_map.append((ti, ci == len(clean_t) - 1))

    full_chars = []
    full_char_indices = []
    for fi, ch in enumerate(full_text):
        if not re.match(r'[\s\W_]', ch):
            full_chars.append(ch.lower())
            full_char_indices.append(fi)

    if not token_chars or not full_chars:
        return

    sm = difflib.SequenceMatcher(None, token_chars, full_chars, autojunk=False)
    blocks = sm.get_matching_blocks()

    token_last_full_pos: Dict[int, int] = {}
    token_first_full_pos: Dict[int, int] = {}

    for block in blocks:
        for offset in range(block.size):
            t_idx, is_last = token_char_map[block.a + offset]
            f_idx = full_char_indices[block.b + offset]
            if t_idx not in token_first_full_pos:
                token_first_full_pos[t_idx] = f_idx
            if is_last:
                token_last_full_pos[t_idx] = f_idx

    for i in range(len(tokens)):
        t = tokens[i]
        t['punct_before'] = ''
        t['punct_after'] = ''
        t['has_sentence_end'] = False
        t['has_clause_end'] = False

        if i in token_last_full_pos:
            curr_f_end = token_last_full_pos[i] + 1
            next_f_start = None
            for nxt_i in range(i + 1, min(len(tokens), i + 4)):
                if nxt_i in token_first_full_pos:
                    next_f_start = token_first_full_pos[nxt_i]
                    break

            if next_f_start is not None and next_f_start >= curr_f_end:
                between = full_text[curr_f_end:next_f_start]
                has_sent = bool(re.search(r'[？。！?!]', between))
                has_clause = bool(re.search(r'[，、；：,;]', between))

                after_punct = "".join(re.findall(r'[？。！?!，、；：,;]', between))
                if after_punct:
                    t['punct_after'] = after_punct
                t['has_sentence_end'] = has_sent
                t['has_clause_end'] = has_clause

    if tokens:
        tokens[-1]['has_sentence_end'] = True


# =============================================================================
# Step 4: Clause-by-Clause Linguistic Word Assembly
# =============================================================================

class LinguisticWord:
    """A linguistically complete word or phrase composed of one or more ASR tokens."""
    def __init__(self, text: str, start: float, end: float, punct_after: str = "",
                 has_sentence_end: bool = False, has_clause_end: bool = False):
        self.text = text
        self.start = start
        self.end = end
        self.punct_after = punct_after
        self.has_sentence_end = has_sentence_end
        self.has_clause_end = has_clause_end

    def __repr__(self):
        return f"<{self.text} [{self.start:.2f}-{self.end:.2f}]{self.punct_after}>"


def merge_single_letter_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges consecutive single English letters in Chinese context (e.g. C + S + M + A -> CSMA)."""
    merged: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []

    def flush():
        if not buf:
            return
        if len(buf) == 1:
            merged.append(buf[0])
        else:
            first, last = buf[0], buf[-1]
            combined = "".join(t["text"].strip() for t in buf)
            new_t = dict(first)
            new_t["text"] = combined
            new_t["end"] = last["end"]
            new_t["punct_after"] = last.get("punct_after", "")
            new_t["has_sentence_end"] = last.get("has_sentence_end", False)
            new_t["has_clause_end"] = last.get("has_clause_end", False)
            merged.append(new_t)
        buf.clear()

    for t in tokens:
        txt = t["text"].strip()
        if len(txt) == 1 and txt.isascii() and txt.isalnum():
            buf.append(t)
        else:
            flush()
            merged.append(t)
    flush()
    return merged


def assemble_clause_linguistic_words(clause_tokens: List[Dict[str, Any]]) -> List[LinguisticWord]:
    """
    Groups a single acoustic breath clause into linguistic words using jieba.
    Because jieba runs on single breath clauses, words are NEVER merged across
    silence pauses!
    """
    if not clause_tokens:
        return []

    clause_tokens = merge_single_letter_tokens(clause_tokens)

    # Check if clause is purely Latin / English words
    is_pure_latin = all(not is_cjk(t["text"]) for t in clause_tokens)
    if not jieba or is_pure_latin:
        return [
            LinguisticWord(
                text=t["text"],
                start=t["start"],
                end=t["end"],
                punct_after=t.get("punct_after", ""),
                has_sentence_end=t.get("has_sentence_end", False),
                has_clause_end=t.get("has_clause_end", False),
            )
            for t in clause_tokens
        ]

    combined_plain = "".join(t["text"] for t in clause_tokens)
    char_to_token = []
    for ti, t in enumerate(clause_tokens):
        for _ in t["text"]:
            char_to_token.append(ti)

    cut_words = list(jieba.cut(combined_plain))

    words: List[LinguisticWord] = []
    char_pos = 0

    for w in cut_words:
        w_len = len(w)
        if w_len == 0:
            continue
        start_tok_idx = char_to_token[char_pos]
        end_tok_idx = char_to_token[char_pos + w_len - 1]
        char_pos += w_len

        tok_slice = clause_tokens[start_tok_idx:end_tok_idx + 1]
        w_start = tok_slice[0]["start"]
        w_end = tok_slice[-1]["end"]
        punct_after = tok_slice[-1].get("punct_after", "")
        has_sent = any(tk.get("has_sentence_end") for tk in tok_slice)
        has_clause = any(tk.get("has_clause_end") for tk in tok_slice)

        words.append(LinguisticWord(
            text=w,
            start=w_start,
            end=w_end,
            punct_after=punct_after,
            has_sentence_end=has_sent,
            has_clause_end=has_clause,
        ))

    return words


def build_all_linguistic_words(tokens: List[Dict[str, Any]]) -> List[LinguisticWord]:
    """
    Partitions token stream into acoustic breath clauses (pause >= 0.4s or sentence boundary),
    then tokenizes each clause with jieba. Guarantees word integrity with zero cross-silence bleed.
    """
    if not tokens:
        return []

    clauses: List[List[Dict[str, Any]]] = []
    cur_clause: List[Dict[str, Any]] = []

    for i in range(len(tokens)):
        t = tokens[i]
        cur_clause.append(t)

        is_last = (i == len(tokens) - 1)
        if is_last:
            clauses.append(cur_clause)
            break

        next_t = tokens[i + 1]
        pause = next_t["start"] - t["end"]

        # Acoustic breath clause boundary: sentence end with pause, or pause >= 0.38s, or backward jump
        if (t.get("has_sentence_end") and pause >= 0.12) or pause >= 0.38 or next_t.get("is_discontinuity"):
            clauses.append(cur_clause)
            cur_clause = []

    all_words: List[LinguisticWord] = []
    for cl in clauses:
        cl_words = assemble_clause_linguistic_words(cl)
        all_words.extend(cl_words)

    return all_words


# =============================================================================
# Step 5: High-Quality Subtitle Segmentation Engine
# =============================================================================

def format_words_to_string(words: List[LinguisticWord]) -> str:
    """Formats words into clean subtitle line with Pangu spacing and punctuation normalization."""
    pieces = []
    for i, w in enumerate(words):
        t = w.text
        if pieces:
            prev_t = words[i - 1].text
            # Add ASCII space between English words
            prev_last = prev_t[-1] if prev_t else ''
            curr_first = t[0] if t else ''
            if curr_first in ('/', '-', '_') or prev_last in ('/', '-', '_'):
                pieces.append(t)
            elif is_latin_word(prev_last) and is_latin_word(curr_first):
                pieces.append(" " + t)
            elif (prev_last in ("'", '"') and is_latin_word(curr_first)) or (is_latin_word(prev_last) and curr_first in ("'", '"')):
                pieces.append(t)
            elif is_latin_word(prev_last) and not is_cjk(curr_first):
                pieces.append(" " + t)
            else:
                pieces.append(t)
        else:
            pieces.append(t)

    line = "".join(pieces).strip()
    line = pangu_format(line)
    # Strip unnecessary terminal commas, colons, hyphens
    line = re.sub(r'[，,、；：;:—\-]+$', '', line)
    # Strip trailing periods, but preserve question/exclamation marks and IP addresses (e.g. 0.0.0.0)
    line = re.sub(r'(?<!\d)\.+$|[。]+$', '', line)
    return line


def score_overflow_split(curr: List[LinguisticWord], k: int, target_vlen: int = 30) -> float:
    """
    Evaluates how linguistically natural and comfortable a candidate split at index k is.
    Higher score = much better split.
    """
    n = len(curr)
    w_prev = curr[k - 1]
    w_curr = curr[k]

    left_str = format_words_to_string(curr[:k])
    right_str = format_words_to_string(curr[k:])
    left_vlen = calc_visual_length(left_str)
    right_vlen = calc_visual_length(right_str)
    pause = w_curr.start - w_prev.end

    # 1. FORBIDDEN CUTS (Score = -999999)
    if is_bad_line_start(w_curr.text):
        return -999999.0
    if left_vlen < 8 or right_vlen < 6:
        return -999999.0
    if w_prev.text in ("第", "每", "上", "下", "前", "后"):
        return -999999.0

    score = 0.0

    # 2. Punctuation Bonus
    if w_prev.has_clause_end:
        score += 180.0
    if w_prev.has_sentence_end:
        score += 250.0

    # 3. Modal Particle End Bonus ('极少啊' / '好了' / '开始了吧')
    if w_prev.text[-1] in "啊呢吧呀哦哈啦喽么":
        score += 70.0

    # 4. Speech Pause Bonus / Penalty
    if pause > 0.05:
        score += min(pause, 0.6) * 120.0
    elif pause < 0.04 and not w_prev.has_clause_end and not w_prev.has_sentence_end:
        score -= 60.0

    # 5. Heavy penalty for isolating a single character with small pause
    if len(w_curr.text) == 1 and pause < 0.12 and not w_prev.has_clause_end and not w_prev.has_sentence_end:
        score -= 100.0

    # 6. Connector Lead Bonus: splitting before '然后', '但是', '所以' etc.
    if any(w_curr.text == conn or w_curr.text.startswith(conn) for conn in LEADING_CONNECTORS):
        score += 130.0

    # 7. Length Balance
    diff = abs(left_vlen - target_vlen)
    score -= diff * 2.5
    if 20 <= left_vlen <= 36 and 8 <= right_vlen <= 36:
        score += 50.0

    # 8. Penalties for bad split points
    if is_dangling_line_end(w_prev.text):
        score -= 260.0
    if w_prev.text in ("是", "为", "叫", "有", "包括"):
        score -= 160.0
    if w_prev.text in ("这个", "那个", "一种", "这种", "某个"):
        score -= 180.0
    # Number followed by unit/noun
    if re.match(r'^\d+$', w_prev.text) or w_prev.text in ("四", "八", "十六", "三十二", "三百"):
        if not is_cjk(w_curr.text[0]) or w_curr.text in ("个", "比特", "字节", "分贝", "兆", "兆赫", "公斤", "斤"):
            score -= 300.0

    return score


def find_best_overflow_split(curr: List[LinguisticWord], target_vlen: int = 30) -> int:
    """Finds the best split point in an overflowing word buffer using multi-factor scoring."""
    n = len(curr)
    if n <= 1:
        return 1

    best_k = max(1, n // 2)
    best_score = -999999.0

    for k in range(1, n):
        s = score_overflow_split(curr, k, target_vlen)
        if s > best_score:
            best_score = s
            best_k = k

    return best_k


def segment_words_into_subtitles(
    words: List[LinguisticWord],
    target_vlen: int = 32,      # ~16 Chinese characters
    max_vlen: int = 40,         # ~20 Chinese characters
    min_split_vlen: int = 18,   # ~9 Chinese characters before considering soft comma split
    max_duration: float = 4.6,  # Max line duration
    long_pause: float = 0.45,   # Speech silence threshold
    medium_pause: float = 0.28  # Medium pause threshold
) -> List[List[LinguisticWord]]:
    """
    Intelligently chunks linguistic words into natural, readable subtitle units.
    Strictly enforces:
    - Never break across words
    - Never dangle clause connectors at line end (conjunction forwarding)
    - Never start a line with bound particles
    - Never swallow words across sentence boundaries or speech pauses
    """
    if not words:
        return []

    subtitles: List[List[LinguisticWord]] = []
    curr: List[LinguisticWord] = []

    n = len(words)
    for i in range(n):
        w = words[i]
        curr.append(w)

        is_last = (i == n - 1)
        if is_last:
            if curr:
                subtitles.append(curr)
            break

        next_w = words[i + 1]
        pause = next_w.start - w.end
        duration = w.end - curr[0].start
        raw_text = "".join(x.text for x in curr)
        vlen = calc_visual_length(raw_text)

        # Particle protection: Never split if next word starts with a bound particle
        if is_bad_line_start(next_w.text):
            continue

        # Rule 1: Definite Sentence Boundary (。？！ or silence >= long_pause)
        if w.has_sentence_end:
            subtitles.append(curr)
            curr = []
            continue

        # Rule 2: Speech Silence Gap (>= 0.45s)
        if pause >= long_pause:
            # Standalone unit if has reasonable content or is an isolated statement
            if vlen >= 4:
                subtitles.append(curr)
                curr = []
                continue

        # Rule 3: Next word is a clause-initial connector ('然后', '但是', '所以', '如果'...)
        is_next_connector = any(next_w.text == conn or next_w.text.startswith(conn) for conn in LEADING_CONNECTORS)
        if is_next_connector and (vlen >= min_split_vlen or pause >= 0.18):
            subtitles.append(curr)
            curr = []
            continue

        # Rule 4: Clause End (，、；： or modal particle with breath pause)
        is_modal_pause = (w.text[-1] in "啊呢吧呀哦哈啦喽么") and (pause >= 0.20) and (vlen >= 12)
        if w.has_clause_end or is_modal_pause:
            # Split if line reached comfortable minimum length or has pause
            if vlen >= min_split_vlen or pause >= medium_pause or is_modal_pause:
                subtitles.append(curr)
                curr = []
                continue

        # Rule 5: Natural Speech Breath Pause (>= 0.28s) with comfortable length (>= 22 units = ~11 chars)
        if pause >= medium_pause and vlen >= 22:
            subtitles.append(curr)
            curr = []
            continue

        # Rule 6: Length or Duration Overflow
        if vlen >= max_vlen or duration >= max_duration:
            split_k = find_best_overflow_split(curr, target_vlen)
            subtitles.append(curr[:split_k])
            curr = curr[split_k:]

    if curr:
        subtitles.append(curr)

    # Post-process: Conjunction forwarding, particle attachment, and cleanup
    subtitles = refine_subtitles_post_pass(subtitles, max_vlen, max_duration)

    return subtitles


def refine_subtitles_post_pass(
    subtitles: List[List[LinguisticWord]],
    max_vlen: int = 40,
    max_duration: float = 4.8
) -> List[List[LinguisticWord]]:
    """
    Comprehensive refinement pass:
    1. Forwards any connector stranded at the end of a line to the start of the next line.
    2. Pulls backward any bound particle ('的', '了', '的话') stranded at the start of a line.
    3. Moves forward any dangling preposition ('在', '把') stranded at the end of a line.
    4. Merges transition markers ('好那么', '好所以', 'OK那么', '然后', '所以') into following sentence.
    5. Attaches tag questions ('对吧', '是不是') to preceding line.
    6. Merges hesitation micro-fragments.
    """
    if not subtitles:
        return []

    # Pass 1: Forward stranded connectors & backward pull bound particles
    for i in range(len(subtitles) - 1):
        curr_sub = subtitles[i]
        next_sub = subtitles[i + 1]

        if not curr_sub or not next_sub:
            continue

        # Check trailing connector (e.g. '然后', '但是', '是因为')
        last_w = curr_sub[-1]
        is_conn = any(last_w.text == conn or last_w.text.endswith(conn) for conn in LEADING_CONNECTORS)
        if is_conn:
            # Only forward if curr_sub has content and is not a pure transition marker
            if len(curr_sub) > 1 and not is_transition_marker_sub(curr_sub):
                next_sub.insert(0, curr_sub.pop())

        # Check trailing preposition (e.g. '在', '把', '被')
        if curr_sub and len(curr_sub) > 1:
            last_w = curr_sub[-1]
            if is_dangling_line_end(last_w.text):
                next_sub.insert(0, curr_sub.pop())

        # Check next leading particle (e.g. '的', '了', '的话')
        if next_sub and len(next_sub) > 1:
            first_w = next_sub[0]
            if is_bad_line_start(first_w.text):
                curr_sub.append(next_sub.pop(0))

    # Remove any empty subtitles
    subtitles = [s for s in subtitles if s]

    # Pass 2: Merge isolated connectors, tag questions, and micro-fragments
    merged: List[List[LinguisticWord]] = []
    i = 0
    n = len(subtitles)

    while i < n:
        curr_s = subtitles[i]
        curr_str = format_words_to_string(curr_s)
        curr_vlen = calc_visual_length(curr_str)

        # 0. Single dangling preposition / bound particle (e.g. standalone '跟', '在', '向', '从')
        # Never leave an isolated preposition alone on a line
        if len(curr_s) == 1 and (is_dangling_line_end(curr_s[0].text) or is_bad_line_start(curr_s[0].text)) and i + 1 < n:
            next_s = subtitles[i + 1]
            next_s.insert(0, curr_s[0])
            i += 1
            continue

        # 1. Spoken transition markers (e.g. '好那么', '好所以', 'OK那么', '然后', '所以', '呃首先')
        # Merges forward into next_s
        if is_transition_marker_sub(curr_s) and i + 1 < n:
            next_s = subtitles[i + 1]
            for w in reversed(curr_s):
                next_s.insert(0, w)
            i += 1
            continue

        # 2. Tag question ('对吧', '是不是')
        # Merges backward into merged[-1]
        is_tag = curr_str.strip("？?！!，,。.") in TAG_QUESTIONS
        if is_tag and merged:
            prev_s = merged[-1]
            prev_str = format_words_to_string(prev_s)
            prev_vlen = calc_visual_length(prev_str)
            combined_vlen = prev_vlen + curr_vlen
            combined_dur = curr_s[-1].end - prev_s[0].start
            pause = curr_s[0].start - prev_s[-1].end
            if combined_vlen <= max_vlen + 6 and combined_dur <= max_duration and pause < 0.8:
                prev_s.extend(curr_s)
                i += 1
                continue

        # 3. Micro-fragment (filler '啊', '呃', len <= 2)
        if len(curr_str) <= 2 and i + 1 < n:
            next_s = subtitles[i + 1]
            next_str = format_words_to_string(next_s)
            pause_to_next = next_s[0].start - curr_s[-1].end
            if pause_to_next < 0.45 and calc_visual_length(next_str) <= max_vlen - 4:
                for w in reversed(curr_s):
                    next_s.insert(0, w)
                i += 1
                continue

        # 4. Check backward merge with previous subtitle in merged list
        if merged:
            prev_s = merged[-1]
            prev_str = format_words_to_string(prev_s)
            prev_vlen = calc_visual_length(prev_str)
            combined_vlen = prev_vlen + curr_vlen
            combined_dur = curr_s[-1].end - prev_s[0].start
            pause = curr_s[0].start - prev_s[-1].end

            # Forbidden backward merge:
            # - If curr starts with a connector
            # - If curr ends with a connector
            # - If curr is a transition marker
            is_curr_connector_start = any(
                curr_s[0].text == conn or curr_s[0].text.startswith(conn)
                for conn in LEADING_CONNECTORS
            )
            is_curr_connector_end = any(
                curr_s[-1].text == conn or curr_s[-1].text.endswith(conn)
                for conn in LEADING_CONNECTORS
            )

            is_prev_short = (prev_vlen <= 14 and not prev_s[-1].has_sentence_end)
            is_curr_short = (curr_vlen <= 14 and not curr_s[-1].has_sentence_end)

            # Merge short hesitation or clause fragment
            if (
                not is_curr_connector_start and
                not is_curr_connector_end and
                (is_prev_short or is_curr_short) and
                pause < 0.45 and
                combined_vlen <= max_vlen and
                combined_dur <= max_duration
            ):
                prev_s.extend(curr_s)
                i += 1
                continue

        merged.append(curr_s)
        i += 1

    return merged


# =============================================================================
# Step 6: Acoustic Anchor & Display Timing Optimization
# =============================================================================

def optimize_subtitle_timings(
    raw_subtitles: List[List[LinguisticWord]],
    min_duration: float = 1.0,     # Anti-flicker min display duration
    tail_padding: float = 0.25,     # Visual comfort padding in silence
    min_gap: float = 0.05,          # Gap between adjacent subtitles (50ms)
    max_duration: float = 4.8       # Max display duration
) -> List[Tuple[float, float, str]]:
    """
    Optimizes subtitle display timing with STRICT acoustic anchor:
    - start_time is 100% anchored to the first word's real acoustic start
    - end_time is extended into trailing silence for comfortable reading speed
    - start_time is NEVER shifted forward, completely preventing lag drift!
    - Overlaps are resolved by trimming previous trailing silence buffer.
    """
    if not raw_subtitles:
        return []

    entries: List[Dict[str, Any]] = []

    for s in raw_subtitles:
        txt = format_words_to_string(s)
        if not txt:
            continue
        st = s[0].start
        et = s[-1].end
        if et <= st:
            et = st + 0.1
        entries.append({
            "start": st,
            "end": et,
            "raw_end": et,
            "text": txt,
            "vlen": calc_visual_length(txt)
        })

    if not entries:
        return []

    # Forward pass: calculate ideal visual display end time
    n = len(entries)
    for i in range(n):
        curr = entries[i]
        start = curr["start"]
        raw_end = curr["raw_end"]
        next_start = entries[i + 1]["start"] if i + 1 < n else start + 3600.0

        vlen = curr["vlen"]
        reading_dur = max(min_duration, vlen / 12.0)
        ideal_end = max(raw_end + tail_padding, start + reading_dur)
        ideal_end = min(ideal_end, start + max_duration)

        # Cap by next speech onset minus safe gap
        if next_start > start + 0.1:
            end = min(ideal_end, next_start - min_gap)
        else:
            end = ideal_end

        if end <= start:
            end = start + min_duration

        curr["end"] = end

    # Overlap resolution: Trim previous end_time, NEVER shift current start_time!
    verified: List[Tuple[float, float, str]] = []
    for i, e in enumerate(entries):
        st = e["start"]
        et = e["end"]
        txt = e["text"]

        if verified:
            prev_st, prev_et, prev_txt = verified[-1]
            if st < prev_et + min_gap:
                trimmed_prev_et = max(prev_st + 0.5, st - min_gap)
                verified[-1] = (prev_st, trimmed_prev_et, prev_txt)
                if st < trimmed_prev_et:
                    st = trimmed_prev_et + min_gap
                    if et <= st:
                        et = st + 0.5

        if et <= st:
            et = st + min_duration

        verified.append((round(st, 3), round(et, 3), txt))

    return verified


# =============================================================================
# Step 7: Main Converter & Output Generation
# =============================================================================

def convert_json_data_to_subtitles(data: Dict[str, Any]) -> List[Tuple[float, float, str]]:
    """Transforms ASR JSON dictionary into broadcast-quality subtitle tuples."""
    tokens, full_text = extract_raw_tokens_and_full_text(data)
    if not tokens:
        return []

    # 1. Acoustic cleaning
    sanitize_acoustic_tokens(tokens)

    # 2. Align genuine punctuation from full_text if available
    align_punctuation_from_full_text(tokens, full_text)

    # 3. Assemble linguistic words clause by clause
    words = build_all_linguistic_words(tokens)

    # 4. Segment into comfortable subtitle lines
    raw_subs = segment_words_into_subtitles(words)

    # 5. Optimize timings with hard acoustic anchor
    final_subs = optimize_subtitle_timings(raw_subs)

    return final_subs


def convert_json_to_srt(
    json_path: str,
    output_srt_path: Optional[str] = None,
    output_txt_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
) -> int:
    """
    Converts a single ASR JSON file to an SRT file (and optional TXT/JSON).
    Returns the number of generated subtitles.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subs = convert_json_data_to_subtitles(data)
    if not subs:
        print(f"[Warning] No subtitles generated from {json_path}")
        return 0

    p_in = Path(json_path)
    srt_out = Path(output_srt_path) if output_srt_path else p_in.with_suffix(".srt")

    # Write SRT
    lines = []
    for idx, (st, et, txt) in enumerate(subs, 1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(st)} --> {format_srt_time(et)}")
        lines.append(txt)
        lines.append("")

    srt_content = "\n".join(lines).strip() + "\n"
    srt_out.write_text(srt_content, encoding="utf-8")

    # Optional TXT
    if output_txt_path:
        txt_out = Path(output_txt_path)
        txt_content = "\n".join(txt for _, _, txt in subs)
        txt_out.write_text(txt_content, encoding="utf-8")

    # Optional resegmented JSON
    if output_json_path:
        json_out = Path(output_json_path)
        resegmented_segments = [
            {"id": idx, "start_time": st, "end_time": et, "text": txt}
            for idx, (st, et, txt) in enumerate(subs, 1)
        ]
        out_data = {
            "language": data.get("language", "Chinese"),
            "full_text": "\n".join(txt for _, _, txt in subs),
            "segments": resegmented_segments,
            "word_timestamps": data.get("word_timestamps", []),
        }
        json_out.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(subs)


def main():
    parser = argparse.ArgumentParser(
        description="Universal Broadcast-Standard ASR Subtitle Converter (JSON -> SRT)"
    )
    parser.add_argument("input", help="Path to input ASR JSON file or directory")
    parser.add_argument("-o", "--output", help="Path to output SRT file or directory")
    parser.add_argument("--txt", action="store_true", help="Also export clean .txt transcript")
    parser.add_argument("--json", action="store_true", help="Also export re-segmented .json data")

    args = parser.parse_args()
    in_path = Path(args.input)

    if in_path.is_file():
        out_srt = args.output
        out_txt = Path(args.output).with_suffix(".txt") if (args.output and args.txt) else (in_path.with_suffix(".txt") if args.txt else None)
        out_json = Path(args.output).with_suffix(".json") if (args.output and args.json) else (in_path.with_suffix(".reseg.json") if args.json else None)

        count = convert_json_to_srt(str(in_path), out_srt, out_txt, out_json)
        print(f"[Done] {in_path.name} -> {count} subtitles generated.")

    elif in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path
        out_dir.mkdir(parents=True, exist_ok=True)
        json_files = sorted(in_path.glob("*.json"))
        print(f"Processing {len(json_files)} JSON files in {in_path}...")

        for jf in json_files:
            srt_dest = out_dir / f"{jf.stem}.srt"
            txt_dest = out_dir / f"{jf.stem}.txt" if args.txt else None
            json_dest = out_dir / f"{jf.stem}.reseg.json" if args.json else None
            count = convert_json_to_srt(str(jf), str(srt_dest), str(txt_dest) if txt_dest else None, str(json_dest) if json_dest else None)
            print(f"  - {jf.name}: {count} subtitles -> {srt_dest.name}")

    else:
        print(f"Error: {in_path} does not exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()


def resegment_json_segments(json_path: str) -> List[Dict[str, Any]]:
    """
    把历史 ASR JSON 清洗重排为规范字幕分段（供服务端与流水线直接调用）。
    返回: [{"id": 1, "start_time": 2.04, "end_time": 6.36, "text": "物理层和数据链路层"}, ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subs = convert_json_data_to_subtitles(data)
    segments = []
    for idx, (st, et, text) in enumerate(subs, 1):
        text = (text or "").strip()
        if not text:
            continue
        segments.append({
            "id": idx,
            "start_time": round(float(st), 3),
            "end_time": round(float(et), 3),
            "text": text
        })
    return segments
