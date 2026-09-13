#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双阶段自适应大模型字幕纠错引擎 (Stage 1 AutoProfiler + Stage 2 LLMSubtitleCorrector)
特性：
1. 领域知识自主分析 (Stage 1)：自适应多点采样音频文本，发现细分领域、核心议题、专业术语库与易错发音陷阱；
2. 锚定逐行纠错 (Stage 2)：严格基于 [ID] 编号逐行纠错，杜绝时间戳脱节与漏句；
3. 思维链极速绕过 (Thinking Bypass)：针对本地部署的 Qwen3.5 9B 模型通过 prefill 绕过长思维链，单批 25 句仅需 2~4 秒；
4. 容错与断点缓存：自动缓存已纠错条目，失败自动重试，保证超大并发稳定输出；
5. 盘古中文排版 (Pangu Spacing)：自动规范中英文与数字间隙。
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(jieba.logging.INFO)
except ImportError:
    jieba = None
    pseg = None

CJK_PATTERN = re.compile(r'[\u4e00-\u9fff]')
LATIN_PATTERN = re.compile(r'[a-zA-Z0-9]')


class CorrectionCancelled(Exception):
    """任务被用户强制终止时抛出：让纠错流水线在批次边界立即中断"""


# 词性规则保护常量（通用无硬编码）
BOUND_MORPHEME_POS = {'uj', 'ul', 'uz', 'ud', 'uv', 'ug', 'y', 'k', 'zg'}
LINKING_WORD_POS = {'c', 'p'}
UNIVERSAL_LATIN_CONNECTORS = {
    'and', 'but', 'or', 'so', 'because', 'then', 'with', 'in', 'on',
    'at', 'to', 'for', 'of', 'by', 'as', 'if', 'that', 'which', 'who'
}


def is_cjk(s: str) -> bool:
    return bool(CJK_PATTERN.search(s))


def is_latin_word(s: str) -> bool:
    return bool(LATIN_PATTERN.search(s))


def calc_visual_length(text: str) -> int:
    """计算视觉显示宽度 (中文字符与全角标点=2，半角ASCII字符=1)"""
    vlen = 0
    for ch in text:
        if CJK_PATTERN.match(ch) or ch in '，。？！、；：“”‘’《》（）…—':
            vlen += 2
        else:
            vlen += 1
    return vlen


def pangu_format(text: str) -> str:
    """盘古排版：中英文与数字之间优雅空格"""
    if not text:
        return ""
    cjk = r'[\u4e00-\u9fa5]'
    text = re.sub(f'({cjk})([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(f'([a-zA-Z0-9])({cjk})', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def format_srt_time(seconds: float) -> str:
    """秒数转 SRT 标准时间戳 HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
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


def parse_srt_time(time_str: str) -> float:
    """SRT 时间戳转浮点秒数"""
    parts = time_str.strip().replace('.', ',').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    secs, millis = parts[2].split(',')
    return hours * 3600 + minutes * 60 + int(secs) + int(millis) / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: 全局领域与术语画像分析 (AutoProfiler)
# ─────────────────────────────────────────────────────────────────────────────

class DomainProfile:
    def __init__(
        self,
        domain: str = "通用综合",
        topic: str = "综合主题演讲与交流",
        jargon_list: Optional[List[str]] = None,
        confusion_rules: Optional[Dict[str, str]] = None,
        style_notes: str = ""
    ):
        self.domain = domain
        self.topic = topic
        self.jargon_list = jargon_list or []
        self.confusion_rules = confusion_rules or {}
        self.style_notes = style_notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "topic": self.topic,
            "jargon_list": self.jargon_list,
            "confusion_rules": self.confusion_rules,
            "style_notes": self.style_notes
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainProfile":
        return cls(
            domain=d.get("domain", "通用综合"),
            topic=d.get("topic", "综合主题演讲与交流"),
            jargon_list=d.get("jargon_list", []),
            confusion_rules=d.get("confusion_rules", {}),
            style_notes=d.get("style_notes", "")
        )


def resolve_llm_model(api_base: str, model: str = "auto") -> str:
    if model and model != "auto":
        return model
    try:
        req = urllib.request.Request(f"{api_base.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = [m['id'] for m in data.get('data', []) if 'embed' not in m.get('id', '').lower()]
            if models:
                return models[0]
    except Exception:
        pass
    return "qwen3.5-9b"


class AutoProfiler:
    """
    Stage 1 分析器：自适应在全文多点取样，自动提取领域类别、专业术语与易错同音字。
    """

    def __init__(self, api_base: str = "http://127.0.0.1:8002/v1", model: str = "auto", timeout: int = 35):
        self.api_base = api_base.rstrip('/')
        self.model = resolve_llm_model(self.api_base, model)
        self.timeout = timeout

    def sample_full_text(self, full_text: str, target_total: int = 2500) -> str:
        n = len(full_text)
        if n <= target_total:
            return full_text

        p1 = int(n * 0.10)
        p2 = int(n * 0.45)
        p3 = int(n * 0.75)
        chunk_len = target_total // 3

        s1 = full_text[p1:p1 + chunk_len]
        s2 = full_text[p2:p2 + chunk_len]
        s3 = full_text[p3:p3 + chunk_len]

        return (
            f"【采样片段一 (前段)】\n{s1}\n\n"
            f"【采样片段二 (中段)】\n{s2}\n\n"
            f"【采样片段三 (后段)】\n{s3}"
        )

    def analyze(self, full_text: str) -> DomainProfile:
        sample = self.sample_full_text(full_text)
        if not sample.strip():
            return DomainProfile()

        system_prompt = """你是一个顶级的跨领域 ASR 语音识别与计算语言学专家。
任务：请阅读给出的语音转录文本采样片段，自主对该音频进行全方位的领域画像分析。

请严格输出纯 JSON 对象，包含以下 5 个字段（不要输出任何多余解释）：
1. "domain": 细分专业领域（如："计算机网络考研", "心血管临床医学", "民法典法考", "法式西点烘焙", "数码硬件评测", "游戏赛事解说" 等）
2. "topic": 本期音频的核心讨论议题与主线内容
3. "jargon_list": 核心专业术语、专有名词、标准英文缩写（统一为规范大小写，数组上限 25 个）
4. "confusion_rules": 预测/检测到的常见 ASR 同音错别字、口误、不合理字母拆开断字（字典形式，如 {"原音/错词": "正确规范词"}，上限 15 个）
5. "style_notes": 说话人口语风格与语速特征（简要描述）"""

        url = f"{self.api_base}/chat/completions"
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': sample}
            ],
            'temperature': 0.1,
            'max_tokens': 1536,
            'chat_template_kwargs': {'enable_thinking': False}
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                raw = res['choices'][0]['message'].get('content', '')
                return self._parse_profile_json(raw)
        except Exception as e:
            print(f"[AutoProfiler] Warning: Auto-profiling error, retrying without chat_template_kwargs: {e}")
            try:
                payload.pop('chat_template_kwargs', None)
                req2 = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req2, timeout=self.timeout) as resp2:
                    res2 = json.loads(resp2.read().decode('utf-8'))
                    raw2 = res2['choices'][0]['message'].get('content', '')
                    return self._parse_profile_json(raw2)
            except Exception as e2:
                print(f"[AutoProfiler] Warning: Auto-profiling fallback to generic: {e2}")
                return DomainProfile()

    def _parse_profile_json(self, raw: str) -> DomainProfile:
        profile = DomainProfile()
        clean = re.sub(r'<think>[\s\S]*?</think>', '', raw)
        clean = re.sub(r'^```json\s*', '', clean.strip())
        clean = re.sub(r'```$', '', clean).strip()

        m = re.search(r'\{[\s\S]*\}', clean)
        if m:
            try:
                data = json.loads(m.group(0))
                return DomainProfile.from_dict(data)
            except Exception:
                pass

        d_m = re.search(r'"domain"\s*:\s*"([^"]+)"', raw)
        if d_m:
            profile.domain = d_m.group(1)

        t_m = re.search(r'"topic"\s*:\s*"([^"]+)"', raw)
        if t_m:
            profile.topic = t_m.group(1)

        j_m = re.search(r'"jargon_list"\s*:\s*\[([\s\S]*?)(\]|\Z)', raw)
        if j_m:
            items = re.findall(r'"([^"]+)"', j_m.group(1))
            profile.jargon_list = items[:25]

        return profile


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: 锚定动态大模型字幕纠错器 (LLMSubtitleCorrector)
# ─────────────────────────────────────────────────────────────────────────────

class LLMSubtitleCorrector:
    """
    Stage 2 纠错器：动态注入 Stage 1 领域画像，逐行锚定 [ID] 执行纠错。
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8002/v1",
        model: str = "auto",
        batch_size: int = 25,
        workers: int = 20,
        profile: Optional[DomainProfile] = None,
        cache_file: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_base = api_base.rstrip('/')
        self.model = resolve_llm_model(self.api_base, model)
        self.batch_size = batch_size
        self.workers = workers
        self.profile = profile or DomainProfile()
        self.cache_file = cache_file
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache: Dict[str, str] = {}

        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"[Corrector] Warning: Failed to load cache file {self.cache_file}: {e}")

    def save_cache(self):
        if not self.cache_file:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def build_dynamic_system_prompt(self) -> str:
        p = self.profile
        jargon_preview = ", ".join(p.jargon_list[:25]) if p.jargon_list else "标准专业学术术语与英文缩写"
        confusion_examples = []
        for k, v in list(p.confusion_rules.items())[:12]:
            confusion_examples.append(f"  - {k} -> {v}")
        confusion_str = "\n".join(confusion_examples) if confusion_examples else "  - 规范常见同音错别字与字母拆开"

        return f"""你是一个顶级的通用 ASR 语音识别高精度字幕逐行纠错专家。
当前音频元知识画像（来自全局自主分析）：
- 细分领域：【{p.domain}】
- 核心主题：【{p.topic}】
- 领域术语库：{jargon_preview}
- 常见混淆与音译参考：
{confusion_str}

【硬性约束，必须逐条遵守】：
1. 必须且仅能输出 [ID] 编号开头的逐行纠错文本，例如：
   [1] 第一行纠错后文本
   [2] 第二行纠错后文本
   编号必须与输入完全一一对应，严禁增行、减行、合并行或跨行搬移文字！
2. 严禁使用任何 Markdown 格式标记（严禁输出 ** 加粗、* 斜体、` 反引号、# 标题等），必须直接输出干净纯文本，严禁添加任何修饰符！
3. 专业术语与专有名词必须准确规范（同音字校正）：
   - 根据【{p.domain}】学科专业背景与术语库，准确校正口语中被 ASR 误识的同音词、近音词（包含口语化拆词与离合词，如“隔什么命”应校正为“革什么命”）；
   - 英文缩写与专有名词统一为标准大小写格式（如 MAC、TCP、IP、DNA、pH 等）。
4. 口语化数值、单位与专业标识规范化：
   - 连续念出的数字地址、版本号等规范为标准点分或阿拉伯数字（例如 "零点零点零点零" -> "0.0.0.0"）；
   - 学术计量单位规范为标准科学记号（例如 "100千赫兹" -> "100 kHz", "50兆赫" -> "50 MHz", "2.5摩尔每升" -> "2.5 mol/L" 等）；
5. 严禁语义脑补或篡改：
   - 绝对禁止将同音专业术语篡改为无关的日常词汇！校正后的词语必须与原词在汉语拼音上发音相近或属于标准外来语/缩写音译。
   - 完整保留说话人的原声语气词（呃、啊、呢、吧、对吧、你看、其实等），严禁擅自增删句子主干或总结修饰！
6. 无法确定的词语必须保持原样，禁止凭想象自作聪明魔改。
7. 严禁输出任何解释、分析过程、思维链标签或多余说明。"""

    def call_api(self, messages: List[Dict[str, Any]], max_tokens: int = 1500) -> str:
        url = f"{self.api_base}/chat/completions"

        attempts = [
            messages,
            messages + [{'role': 'assistant', 'content': '</think>', 'prefix': True}]
        ]

        for attempt_msgs in attempts:
            payload = {
                'model': self.model,
                'messages': attempt_msgs,
                'temperature': 0.1,
                'max_tokens': max_tokens,
                'chat_template_kwargs': {'enable_thinking': False}
            }
            req_data = json.dumps(payload).encode('utf-8')

            for retry in range(self.max_retries):
                try:
                    req = urllib.request.Request(
                        url,
                        data=req_data,
                        headers={'Content-Type': 'application/json'}
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        res = json.loads(resp.read().decode('utf-8'))
                        msg = res['choices'][0]['message']
                        content = msg.get('content', '') or ''
                        content = re.sub(r'<think>[\s\S]*?</think>', '', content)
                        content = re.sub(r'^</think>', '', content).strip()
                        if content:
                            return content
                        reasoning = msg.get('reasoning_content', '')
                        if reasoning and not content:
                            break
                except urllib.error.HTTPError as he:
                    err_body = he.read().decode('utf-8', errors='ignore')
                    if 'prefix' in err_body:
                        break
                    time.sleep(0.5)
                except Exception:
                    time.sleep(0.5)

        return ""

    def parse_batch_response(self, raw_text: str, original_batch: List[Tuple[int, str]]) -> Dict[int, str]:
        line_pattern = re.compile(r'^(?:\[(\d+)\]|(\d+)[\.\:：\s])\s*(.*)$')
        parsed: Dict[int, str] = {}
        batch_ids = {idx for idx, _ in original_batch}
        orig_dict = {idx: txt for idx, txt in original_batch}

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
                    corrected_text = m.group(3).strip()
                    # 彻底清除大模型自作主张输出的 Markdown 加粗、斜体、代码块或多余星号
                    corrected_text = re.sub(r'[*_`~]+', '', corrected_text).strip()
                    orig_len = len(orig_dict[line_id])
                    corr_len = len(corrected_text)
                    if corr_len > 0 and 0.25 <= (corr_len / max(1, orig_len)) <= 2.5:
                        parsed[line_id] = pangu_format(corrected_text)

        return parsed

    def _process_one_batch(
        self,
        b_idx: int,
        batch: List[Tuple[int, str]],
        system_prompt: str
    ) -> Tuple[int, Dict[int, str], Dict[int, str], List[Tuple[int, str, str]], int]:
        prompt_lines = [f"[{idx}] {text}" for idx, text in batch]
        user_prompt = "请对以下字幕进行逐行纠错：\n" + "\n".join(prompt_lines)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        raw_response = self.call_api(messages)
        parsed_batch = self.parse_batch_response(raw_response, batch)

        from .correction_guard import verify_correction

        batch_final: Dict[int, str] = {}
        batch_raw: Dict[int, str] = {}
        batch_accepted: List[Tuple[int, str, str]] = []
        batch_rejected = 0

        for idx, orig_text in batch:
            if idx in parsed_batch:
                c_text = parsed_batch[idx]
                batch_raw[idx] = c_text
                try:
                    accepted, reason = verify_correction(orig_text, c_text, self.profile)
                except Exception:
                    accepted, reason = True, "guard-unavailable"

                if not accepted:
                    batch_rejected += 1
                    c_text = orig_text
                elif c_text != orig_text:
                    batch_accepted.append((idx, orig_text, c_text))

                batch_final[idx] = c_text
            else:
                batch_final[idx] = orig_text

        return b_idx, batch_final, batch_raw, batch_accepted, batch_rejected

    def correct_subtitles(
        self,
        subtitles: List[Tuple[float, float, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        on_correction: Optional[Callable[[int, str, str], None]] = None
    ) -> List[Tuple[float, float, str]]:
        total = len(subtitles)
        if total == 0:
            return []

        print(f"[Stage 2 Corrector] 启动 {total} 行字幕批量纠错...")
        print(f"  -> Model: {self.model} | Batch: {self.batch_size} | Workers: {self.workers} | API: {self.api_base}")

        system_prompt = self.build_dynamic_system_prompt()
        pending_items = []
        final_map: Dict[int, str] = {}
        cache_hits = 0

        from .correction_guard import verify_correction

        for idx, (start, end, text) in enumerate(subtitles, 1):
            if text in self.cache:
                cached_cand = self.cache[text]
                try:
                    accepted, _ = verify_correction(text, cached_cand, self.profile)
                except Exception:
                    accepted = True
                final_map[idx] = cached_cand if accepted else text
                cache_hits += 1
            else:
                pending_items.append((idx, text))

        if cache_hits > 0:
            print(f"  -> [Cache Hit] 命中缓存 {cache_hits}/{total} 条 ({cache_hits/total*100:.1f}%)")

        batches = [pending_items[i:i + self.batch_size] for i in range(0, len(pending_items), self.batch_size)]
        num_batches = len(batches)
        corrections_made = 0

        t0_all = time.time()

        if num_batches > 0:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(self._process_one_batch, i, b, system_prompt): i
                    for i, b in enumerate(batches, 1)
                }

                completed_batches = 0
                for future in as_completed(futures):
                    if should_stop and should_stop():
                        print("[Stage 2] 检测到用户终止信号，取消剩余任务")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise CorrectionCancelled()

                    b_idx, b_final, b_raw, b_accepted, b_rej = future.result()
                    completed_batches += 1

                    for idx, orig_txt in batches[b_idx - 1]:
                        fin_txt = b_final.get(idx, orig_txt)
                        final_map[idx] = fin_txt
                        if idx in b_raw:
                            self.cache[orig_txt] = b_raw[idx]

                    for idx, o_txt, c_txt in b_accepted:
                        corrections_made += 1
                        if on_correction:
                            try:
                                on_correction(idx, o_txt, c_txt)
                            except Exception:
                                pass

                    if completed_batches % 10 == 0 or completed_batches == num_batches:
                        self.save_cache()

                    if progress_callback:
                        processed_lines = cache_hits + sum(len(batches[futures[f] - 1]) for f in futures if f.done())
                        progress_callback(processed_lines, total)

        dt_total = time.time() - t0_all
        print(f"[Stage 2 Complete] 共完成 {corrections_made} 处字幕修正，总耗时 {dt_total:.1f}s")

        corrected_results: List[Tuple[float, float, str]] = []
        for idx, (start, end, orig_text) in enumerate(subtitles, 1):
            corr_text = final_map.get(idx, orig_text)
            corrected_results.append((start, end, corr_text))

        return corrected_results


# ─────────────────────────────────────────────────────────────────────────────
# 顶层集成管道接口 (用于与 TaskQueue 和 Server 联动)
# ─────────────────────────────────────────────────────────────────────────────

def is_llm_available(api_base: str = "http://127.0.0.1:8002/v1") -> bool:
    """快速探测本地 LLM API 服务是否可用"""
    try:
        req = urllib.request.Request(f"{api_base.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_llm_status(api_base: str = "http://127.0.0.1:8002/v1") -> Dict[str, Any]:
    """快速探测本地 LLM API 服务是否可用，并返回当前模型信息"""
    try:
        req = urllib.request.Request(f"{api_base.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                models = [m['id'] for m in data.get('data', []) if 'embed' not in m.get('id', '').lower()]
                model_name = models[0] if models else "qwen3.5-9b"
                return {"available": True, "model": model_name, "models": models}
    except Exception:
        pass
    return {"available": False, "model": "", "models": []}


def correct_segments_pipeline(
    segments: List[Dict[str, Any]],
    full_text: str = "",
    api_base: str = "http://127.0.0.1:8002/v1",
    model: str = "auto",
    batch_size: int = 25,
    workers: int = 20,
    cache_file: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_correction: Optional[Callable[[int, str, str], None]] = None,
) -> Tuple[List[Dict[str, Any]], DomainProfile]:
    """
    对已有的字幕分句 segments 列表执行 Stage 1 画像 + Stage 2 逐行纠错。
    返回更新后的 segments 列表与 Profile。
    """
    if not segments:
        return segments, DomainProfile()

    if should_stop and should_stop():
        raise CorrectionCancelled()

    if not full_text:
        full_text = " ".join(seg.get("text", "") for seg in segments)

    # Stage 1: Auto-Profiling
    profiler = AutoProfiler(api_base=api_base, model=model)
    profile = profiler.analyze(full_text)
    print(f"[Pipeline] 识别领域: 【{profile.domain}】 | 核心主题: 【{profile.topic}】")

    if should_stop and should_stop():
        raise CorrectionCancelled()

    # 格式转换
    sub_tuples = [(float(s["start_time"]), float(s["end_time"]), str(s["text"])) for s in segments]

    # Stage 2: Batch GER Correction
    corrector = LLMSubtitleCorrector(
        api_base=api_base,
        model=model,
        batch_size=batch_size,
        workers=workers,
        profile=profile,
        cache_file=cache_file,
    )
    corrected_tuples = corrector.correct_subtitles(
        sub_tuples,
        progress_callback=progress_callback,
        should_stop=should_stop,
        on_correction=on_correction,
    )

    # 回填
    new_segments = []
    for idx, (st, et, text) in enumerate(corrected_tuples, 1):
        seg = dict(segments[idx - 1])
        seg["id"] = idx
        seg["start_time"] = round(st, 3)
        seg["end_time"] = round(et, 3)
        seg["text"] = text
        new_segments.append(seg)

    return new_segments, profile


def process_json_file_standalone(
    json_path: str,
    output_srt: Optional[str] = None,
    output_txt: Optional[str] = None,
    api_base: str = "http://127.0.0.1:8002/v1",
    model: str = "auto",
    batch_size: int = 25,
    workers: int = 20,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_correction: Optional[Callable[[int, str, str], None]] = None,
    resegment: bool = True,
) -> Dict[str, Any]:
    """
    独立处理单个已有 ASR JSON 文件并覆盖/导出 SRT、TXT 及纠错后 JSON。

    resegment=True 时，会先用 json_resegment（迁移自早期 asr_to_srt.py 的清洗重排引擎）
    修复历史 JSON 里 6 秒硬切造成的断词碎片与错乱时间戳，再做 LLM 纠错，
    保证输出的 SRT 与「新转录」流程同样规整。
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    data = json.loads(p.read_text(encoding="utf-8"))
    full_text = data.get("full_text", "")
    segments = data.get("segments", [])

    if not segments and not full_text:
        raise ValueError("JSON 中不含有效的 segments 或 full_text 数据")

    if resegment:
        try:
            from .json_resegment import resegment_json_segments

            resegmented = resegment_json_segments(str(p))
        except Exception as e:
            resegmented = []
            print(f"[JSON 重排] 引擎异常，回退原始 segments: {e}")

        if resegmented:
            print(f"[JSON 重排] {len(segments)} 段（原始） -> {len(resegmented)} 段（清洗重排）")
            segments = resegmented
            full_text = "".join(s["text"] for s in segments)

    # 准备输出路径
    target_srt = Path(output_srt) if output_srt else p.with_suffix(".srt")
    target_txt = Path(output_txt) if output_txt else p.with_suffix(".txt")

    cache_file = None
    if cache_dir:
        c_dir = Path(cache_dir)
        c_dir.mkdir(parents=True, exist_ok=True)
        cache_file = str(c_dir / f"{p.stem}.corrections.json")

    # 执行纠错流水线
    corrected_segments, profile = correct_segments_pipeline(
        segments=segments,
        full_text=full_text,
        api_base=api_base,
        model=model,
        batch_size=batch_size,
        workers=workers,
        cache_file=cache_file,
        progress_callback=progress_callback,
        should_stop=should_stop,
        on_correction=on_correction,
    )

    # 确定性文本规范化：缩写逐字母读开、中文数字读法、中英/数字间距
    try:
        from .text_normalize import normalize_technical_text

        for seg in corrected_segments:
            seg["text"] = normalize_technical_text(seg.get("text", ""))
    except Exception as e:
        print(f"[Normalize] 跳过文本规范化: {e}")

    # 导出 SRT
    lines = []
    for idx, seg in enumerate(corrected_segments, 1):
        lines.append(str(idx))
        st_str = format_srt_time(seg["start_time"])
        et_str = format_srt_time(seg["end_time"])
        lines.append(f"{st_str} --> {et_str}")
        lines.append(seg["text"])
        lines.append("")
    srt_content = "\n".join(lines).strip() + "\n"
    target_srt.write_text(srt_content, encoding="utf-8")

    # 导出 TXT
    txt_content = "\n".join(seg["text"] for seg in corrected_segments if seg.get("text"))
    target_txt.write_text(txt_content, encoding="utf-8")

    # 覆盖更新 JSON 的 segments 和 full_text
    data["segments"] = corrected_segments
    data["full_text"] = txt_content
    data["domain_profile"] = profile.to_dict()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "success": True,
        "json_path": str(p),
        "srt_path": str(target_srt),
        "txt_path": str(target_txt),
        "segments_count": len(corrected_segments),
        "domain": profile.domain,
        "topic": profile.topic,
        "jargon_count": len(profile.jargon_list),
    }
