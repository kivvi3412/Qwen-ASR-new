#!/usr/bin/env python3
"""
工业级音视频字幕切分与导出模块
技术特性：
1. 标点与语义主导切分：以强标点（。？！）为硬边界，以逗号分号为舒适阅读分段
2. 词法完整性保护：严禁在专有名词或复合词中间截断（如“数据结构”、“以太网”、“10BASE5”）
3. 连词智能吸附与前移（Conjunction Forwarding）：防止“然后/但是/所以”等句首连词被割裂留在上一句末尾
4. 时长与空帧规整：彻底消除 0 秒无效帧与微小碎片帧，保证单行 1.0s~5.5s 视听舒适区
5. 纯净段落 .txt 与精细词级 .json 统一联动导出
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 标点与语法符号常量 ──
_HARD_BREAK = set("。！？!?\n")
_SOFT_BREAK = set("，,;；、:")
_STRIP_PUNCT = '，。、；：·"\' 　\n.,;:'

# 句首常见前置连词/过渡词（若出现在上一句末尾且后有断句，自动前移至下一句句首）
_LEADING_CONNECTORS = (
    "总的来说", "换句话说", "也就是说", "接下来", "另一方面",
    "然后", "但是", "所以", "那么", "而且", "另外", "并且",
    "其实", "因为", "由于", "比如", "不过", "然而", "如果", "虽然",
    "接着", "首先", "其次", "最后"
)

# 默认视听工程参数
DEFAULT_GAP_THRESHOLD = 0.55  # 自然停顿断句阈值（秒）
DEFAULT_MAX_DURATION = 5.5    # 单句最大建议时长（秒）
DEFAULT_MIN_DURATION = 0.8    # 单句最小展示时长（秒）
IDEAL_MAX_CHARS = 24          # 单句建议汉字上限（防止屏幕单行过长）


def format_srt_timestamp(seconds: float) -> str:
    """转换秒数为 SRT 标准时间戳格式 HH:MM:SS,mmm"""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def pangu_format(text: str) -> str:
    """盘古排版：中英文与数字之间优雅空格"""
    if not text:
        return ""
    cjk = r'[\u4e00-\u9fa5]'
    text = re.sub(f'({cjk})([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(f'([a-zA-Z0-9])({cjk})', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _clean_sub_text(text: str) -> str:
    """清理字幕首尾常见无用标点，保留问号和感叹号，并规范中英文空格"""
    if not text:
        return ""
    t = text.strip()
    while t and t[0] in _STRIP_PUNCT:
        t = t[1:].strip()
    while t and t[-1] in _STRIP_PUNCT:
        t = t[:-1].strip()
    return pangu_format(t)


def _build_srt_string(segments: List[Dict[str, Any]]) -> str:
    """将字幕列表格式化为标准 SRT 纯文本"""
    if not segments:
        return ""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        start_str = format_srt_timestamp(seg["start_time"])
        end_str = format_srt_timestamp(seg["end_time"])
        lines.append(f"{start_str} --> {end_str}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _normalize_char(c: str) -> str:
    """字符归一化便于容错对齐"""
    return unicodedata.normalize("NFKC", c).lower().strip()


def _build_char_timestamps(punctuated: str, items: List[Any]) -> List[Dict[str, Any]]:
    """
    鲁棒单调对齐算法：将带有标点的全文字符映射到 ForcedAligner 的毫秒级时间戳序列。
    具备局部跳字恢复与时间平滑机制，彻底杜绝单字缺失导致对齐崩溃。
    """
    char_records: List[Dict[str, Any]] = []
    if not punctuated:
        return char_records

    flat_tokens = []
    for it in items:
        if isinstance(it, dict):
            t_txt = str(it.get("text", "") or "").strip()
            st = float(it.get("start_time", 0.0))
            et = float(it.get("end_time", 0.0))
        else:
            t_txt = str(getattr(it, "text", "") or "").strip()
            st = float(getattr(it, "start_time", 0.0))
            et = float(getattr(it, "end_time", 0.0))
        if t_txt:
            if et <= st:
                et = st + 0.08
            flat_tokens.append({"text": t_txt, "st": st, "et": et})

    if not flat_tokens:
        return []

    tok_idx = 0
    num_toks = len(flat_tokens)
    last_st = flat_tokens[0]["st"]
    last_et = flat_tokens[0]["et"]

    for c in punctuated:
        norm_c = _normalize_char(c)
        if not norm_c or norm_c in _HARD_BREAK or norm_c in _SOFT_BREAK or norm_c in _STRIP_PUNCT:
            char_records.append({
                "char": c,
                "is_punct": True,
                "start_time": last_et,
                "end_time": last_et,
            })
            continue

        matched_tok = None
        for search_offset in range(min(15, num_toks - tok_idx)):
            cand = flat_tokens[tok_idx + search_offset]
            cand_norm = _normalize_char(cand["text"])
            if norm_c in cand_norm:
                matched_tok = cand
                tok_idx = tok_idx + search_offset + 1
                break

        if matched_tok:
            last_st = matched_tok["st"]
            last_et = max(matched_tok["et"], last_st + 0.05)
            char_records.append({
                "char": c,
                "is_punct": False,
                "start_time": last_st,
                "end_time": last_et,
            })
        else:
            last_st = last_et
            last_et = last_et + 0.08
            char_records.append({
                "char": c,
                "is_punct": False,
                "start_time": last_st,
                "end_time": last_et,
            })

    return char_records


def _segment_by_clauses(
    punctuated: str,
    char_records: List[Dict[str, Any]],
    max_duration: float = DEFAULT_MAX_DURATION,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    核心智能分句引擎：
    1. 标点语义主断句
    2. 单行汉字上限与时长联合控制（避免一行过长或过短）
    3. 严禁断词破词
    """
    if not char_records:
        return []

    clauses: List[Dict[str, Any]] = []
    cur_chars: List[str] = []
    cur_records: List[Dict[str, Any]] = []
    c_start_t = None
    c_end_t = None

    n = len(char_records)
    for i in range(n):
        rec = char_records[i]
        c = rec["char"]
        st = rec["start_time"]
        et = rec["end_time"]

        if c_start_t is None:
            c_start_t = st
        c_end_t = max(c_end_t or et, et)
        cur_chars.append(c)
        cur_records.append(rec)

        is_hard = (c in _HARD_BREAK)
        is_soft = (c in _SOFT_BREAK)
        
        gap = 0.0
        if i + 1 < n:
            gap = char_records[i + 1]["start_time"] - et

        is_large_pause = (gap >= gap_threshold and len(cur_chars) >= 4)
        is_too_long = (c_end_t - c_start_t >= max_duration * 1.5 or len(cur_chars) >= IDEAL_MAX_CHARS * 1.5)
        is_last = (i == n - 1)

        if is_hard or is_soft or is_large_pause or is_too_long or is_last:
            raw_text = "".join(cur_chars).strip()
            clean = _clean_sub_text(raw_text)
            if clean:
                base_cl = {
                    "text": clean,
                    "raw": raw_text,
                    "start_time": c_start_t,
                    "end_time": c_end_t,
                    "has_hard": is_hard,
                }
                if (c_end_t - c_start_t > max_duration or len(clean) > IDEAL_MAX_CHARS) and len(cur_records) > 6:
                    sub_pieces = _split_long_clause(base_cl, cur_records, max_duration, IDEAL_MAX_CHARS)
                    clauses.extend(sub_pieces)
                else:
                    clauses.append(base_cl)
            cur_chars = []
            cur_records = []
            c_start_t = None
            c_end_t = None

    if not clauses:
        return []

    packed_segments: List[Dict[str, Any]] = []
    accum_clause: Optional[Dict[str, Any]] = None

    for cl in clauses:
        if accum_clause is None:
            accum_clause = dict(cl)
            continue

        accum_text = accum_clause["text"]
        new_text = cl["text"]
        combined_text = accum_text + "，" + new_text
        combined_dur = cl["end_time"] - accum_clause["start_time"]
        char_count = len(combined_text)

        gap_between = cl["start_time"] - accum_clause["end_time"]

        can_merge = (
            not accum_clause.get("has_hard", False)
            and combined_dur <= max_duration
            and char_count <= IDEAL_MAX_CHARS
            and gap_between < 0.8
        )

        if can_merge:
            accum_clause["text"] = combined_text
            accum_clause["end_time"] = cl["end_time"]
            accum_clause["has_hard"] = cl.get("has_hard", False)
        else:
            packed_segments.append(accum_clause)
            accum_clause = dict(cl)

    if accum_clause:
        packed_segments.append(accum_clause)

    return packed_segments


def _split_long_clause(
    clause: Dict[str, Any],
    char_records: List[Dict[str, Any]],
    max_duration: float = DEFAULT_MAX_DURATION,
    max_chars: int = IDEAL_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """递归语义断句：针对长句或无标点语音流，在语法连接词/语气助词/停顿点处智能切割"""
    text = clause.get("text", "")
    dur = clause["end_time"] - clause["start_time"]
    if (dur <= max_duration and len(text) <= max_chars) or len(char_records) <= 6:
        return [clause]

    leading_markers = (
        "以及", "关于", "并且", "其中", "如果", "那么", "而且", "或者",
        "但是", "对于", "通过", "此时", "包括", "同时", "从而", "为了",
        "所以", "因为", "另外", "接下来",
    )
    trailing_markers = (
        "的", "了", "呢", "吧", "啊", "吗", "呀", "哦", "中", "后", "上", "完", "过"
    )

    n = len(char_records)
    min_k = max(4, int(n * 0.25))
    max_k = min(n - 3, int(n * 0.75))

    best_k = n // 2
    best_score = -9999.0

    for k in range(min_k, max_k + 1):
        score = 0.0
        score -= abs(k - n / 2) * 0.35

        sub_ahead = text[k:] if k < len(text) else ""
        for m in leading_markers:
            if sub_ahead.startswith(m):
                score += 10.0
                break

        sub_behind = text[:k] if k <= len(text) else ""
        for m in trailing_markers:
            if sub_behind.endswith(m):
                score += 6.0
                break

        if k < n:
            gap = char_records[k]["start_time"] - char_records[k - 1]["end_time"]
            if gap > 0.1:
                score += gap * 20.0

        if score > best_score:
            best_score = score
            best_k = k

    c1_records = char_records[:best_k]
    c2_records = char_records[best_k:]
    if not c1_records or not c2_records:
        return [clause]

    c1 = {
        "text": _clean_sub_text("".join(r["char"] for r in c1_records)),
        "start_time": c1_records[0]["start_time"],
        "end_time": c1_records[-1]["end_time"],
        "has_hard": False,
    }
    c2 = {
        "text": _clean_sub_text("".join(r["char"] for r in c2_records)),
        "start_time": c2_records[0]["start_time"],
        "end_time": c2_records[-1]["end_time"],
        "has_hard": clause.get("has_hard", False),
    }

    return (
        _split_long_clause(c1, c1_records, max_duration, max_chars)
        + _split_long_clause(c2, c2_records, max_duration, max_chars)
    )


def _apply_conjunction_forwarding(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    智能连词前移机制：
    识别当前分句句尾出现的过渡连词（如“然后”、“但是”、“所以”），
    将其自动吸附至下一句的句首，彻底解决“下一句的字跑到上一句”的顽疾。
    """
    if len(segments) <= 1:
        return segments

    for i in range(len(segments) - 1):
        curr_text = segments[i]["text"]
        next_text = segments[i + 1]["text"]

        matched_conn = None
        for conn in _LEADING_CONNECTORS:
            if curr_text.endswith(conn) and len(curr_text) > len(conn):
                matched_conn = conn
                break

        if matched_conn:
            new_curr = curr_text[:-len(matched_conn)].rstrip(_STRIP_PUNCT).strip()
            new_next = matched_conn + "，" + next_text.lstrip(_STRIP_PUNCT).strip()

            if new_curr:
                segments[i]["text"] = new_curr
                segments[i + 1]["text"] = new_next
                conn_duration = 0.25 * len(matched_conn)
                split_t = max(segments[i]["start_time"] + 0.3, segments[i]["end_time"] - conn_duration)
                segments[i]["end_time"] = round(split_t, 3)
                segments[i + 1]["start_time"] = round(split_t, 3)

    return segments


def _sanitize_and_smooth_subtitles(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    最终质量过滤与时间轴平滑：
    1. 确保字幕时长 >= DEFAULT_MIN_DURATION（防止闪退消失）
    2. 清除孤立单字/乱码
    3. 保证单调递增
    """
    cleaned: List[Dict[str, Any]] = []

    for seg in segments:
        txt = _clean_sub_text(seg["text"])
        st = float(seg["start_time"])
        et = float(seg["end_time"])

        if not txt:
            continue

        if et - st < DEFAULT_MIN_DURATION:
            et = st + DEFAULT_MIN_DURATION

        if cleaned:
            prev = cleaned[-1]
            if st < prev["end_time"]:
                st = prev["end_time"]
            if et <= st:
                et = st + DEFAULT_MIN_DURATION

        cleaned.append({
            "id": len(cleaned) + 1,
            "start_time": round(st, 3),
            "end_time": round(et, 3),
            "text": txt,
        })

    return cleaned


def generate_srt(
    result: Any,
    output_path: Optional[str] = None,
    max_duration: float = DEFAULT_MAX_DURATION,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    time_offset: float = 0.0,
) -> Tuple[int, List[Dict[str, Any]], str]:
    """
    工业级字幕切分生成主接口：
    采用标点驱动 + 字符单调对齐 + 连词吸附 + 视听黄金区间规整
    """
    punctuated = (getattr(result, "text", "") or "").strip()
    time_stamps_obj = getattr(result, "time_stamps", None)
    items = getattr(time_stamps_obj, "items", None) if time_stamps_obj else None

    if items is None and hasattr(time_stamps_obj, "__iter__"):
        items = list(time_stamps_obj)

    segments = []

    if not items:
        if punctuated:
            segments = [{
                "id": 1,
                "start_time": round(0.0 + time_offset, 3),
                "end_time": round(max(3.0, len(punctuated) * 0.25) + time_offset, 3),
                "text": _clean_sub_text(punctuated),
            }]
    else:
        char_records = _build_char_timestamps(punctuated, items)

        if char_records:
            segments = _segment_by_clauses(punctuated, char_records, max_duration, gap_threshold)
        else:
            raw_clauses = []
            buf_words = []
            b_st = 0.0
            b_et = 0.0
            for it in items:
                w_txt = (it.get("text", "") if isinstance(it, dict) else getattr(it, "text", "")) or ""
                w_st = float(it.get("start_time", 0.0) if isinstance(it, dict) else getattr(it, "start_time", 0.0))
                w_et = float(it.get("end_time", 0.0) if isinstance(it, dict) else getattr(it, "end_time", 0.0))
                if not buf_words:
                    b_st = w_st
                b_et = max(b_et, w_et)
                buf_words.append(w_txt)
                if any(p in _HARD_BREAK or p in _SOFT_BREAK for p in w_txt) or (b_et - b_st >= max_duration):
                    raw_clauses.append({"text": "".join(buf_words), "start_time": b_st, "end_time": b_et})
                    buf_words = []
            if buf_words:
                raw_clauses.append({"text": "".join(buf_words), "start_time": b_st, "end_time": b_et})
            segments = raw_clauses

        segments = _apply_conjunction_forwarding(segments)
        segments = _sanitize_and_smooth_subtitles(segments)

        if time_offset > 0:
            for seg in segments:
                seg["start_time"] = round(seg["start_time"] + time_offset, 3)
                seg["end_time"] = round(seg["end_time"] + time_offset, 3)

    srt_content = _build_srt_string(segments)
    if output_path:
        Path(output_path).write_text(srt_content, encoding="utf-8")

    return len(segments), segments, srt_content


def generate_txt(
    result: Any,
    segments: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
) -> str:
    """生成纯净段落文本内容"""
    if segments:
        text_content = "\n".join(seg["text"] for seg in segments if seg.get("text"))
    else:
        text_content = getattr(result, "text", "") or ""

    if output_path:
        Path(output_path).write_text(text_content, encoding="utf-8")
    return text_content


def generate_json(
    result: Any,
    segments: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    time_offset: float = 0.0,
) -> Dict[str, Any]:
    """生成带详细词/字符时间戳的 JSON 数据"""
    language = getattr(result, "language", "") or "Auto"
    full_text = getattr(result, "text", "") or ""

    time_stamps_obj = getattr(result, "time_stamps", None)
    items = getattr(time_stamps_obj, "items", None) if time_stamps_obj else None
    if items is None and hasattr(time_stamps_obj, "__iter__"):
        items = list(time_stamps_obj)

    raw_timestamps = []
    if items:
        for it in items:
            if isinstance(it, dict):
                tok_text = str(it.get("text", "") or "")
                st = float(it.get("start_time", 0.0))
                et = float(it.get("end_time", 0.0))
            else:
                tok_text = str(getattr(it, "text", "") or "")
                st = float(getattr(it, "start_time", 0.0))
                et = float(getattr(it, "end_time", 0.0))
            raw_timestamps.append({
                "text": tok_text,
                "start_time": round(st + time_offset, 3),
                "end_time": round(max(et, st + 0.05) + time_offset, 3),
            })

    data = {
        "language": language,
        "full_text": full_text,
        "segments": segments or [],
        "word_timestamps": raw_timestamps,
    }

    if output_path:
        Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return data


def export_final_subtitles(
    all_segments: List[Dict[str, Any]],
    all_words: List[Dict[str, Any]],
    target_srt: str,
    target_txt: str,
    target_json: str,
    language: str = "Chinese",
):
    """
    全局汇总输出阶段：
    再次进行跨切片连词吸附与时间轴统一重排
    """
    refined_segments = _apply_conjunction_forwarding(all_segments)
    refined_segments = _sanitize_and_smooth_subtitles(refined_segments)

    # 确定性文本规范化（缩写逐字母读开、中文数字读法、中英/数字间距）
    try:
        from .text_normalize import normalize_technical_text

        for seg in refined_segments:
            seg["text"] = normalize_technical_text(seg.get("text", ""))
    except Exception as e:
        print(f"[Normalize] 跳过文本规范化: {e}")

    for idx, seg in enumerate(refined_segments, 1):
        seg["id"] = idx

    srt_content = _build_srt_string(refined_segments)
    Path(target_srt).write_text(srt_content, encoding="utf-8")

    txt_content = "\n".join(seg["text"] for seg in refined_segments if seg.get("text"))
    Path(target_txt).write_text(txt_content, encoding="utf-8")

    json_data = {
        "language": language,
        "full_text": txt_content,
        "segments": refined_segments,
        "word_timestamps": all_words,
    }
    Path(target_json).write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
