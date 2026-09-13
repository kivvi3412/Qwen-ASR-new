#!/usr/bin/env python3
"""
异步任务队列与流水线调度器
1. CPU 多线程并发抽取音频 (Double-Buffering)，消除 GPU 等待
2. 任务状态与文件独立目录持久化存储 (data/tasks/{task_id}/)
3. 页面刷新状态保持、历史任务追溯、一键 TAR.GZ 打包下载所有字幕
4. SSE (Server-Sent Events) 实时广播处理进度、实时倍速 (RTF) 与状态
"""

import asyncio
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from .corrector import (
    CorrectionCancelled,
    correct_segments_pipeline,
    process_json_file_standalone,
)
from .engine import engine
from .subtitle import export_final_subtitles, generate_json, generate_srt, generate_txt


MEDIA_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
}

BASE_DIR = Path(__file__).resolve().parent.parent
# 任务数据统一放在项目根目录 data/ 下（asr/ 与 llm4b/ 之外），便于历史任务长期保留
DATA_DIR = Path(os.environ.get("ASR_DATA_DIR") or (BASE_DIR.parent / "data"))
TASKS_DIR = DATA_DIR / "tasks"
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# 任务卡片上最多保留/展示的纠错示例条数（避免 SSE 与 task.json 膨胀）
MAX_CORRECTION_PREVIEW = 12


def get_media_duration(file_path: str) -> Optional[float]:
    """通过 ffprobe 快速获取音视频时长（秒）"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(file_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            return round(float(res.stdout.strip()), 2)
    except Exception:
        pass
    return None


def extract_audio_pcm(media_path: str, timeout: int = 600) -> Tuple[np.ndarray, int]:
    """使用 ffmpeg 管道直接解码输出 16kHz mono float32 WAV 数据"""
    cmd = [
        "ffmpeg", "-i", str(media_path),
        "-ac", "1", "-ar", "16000", "-f", "wav",
        "-v", "quiet", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 解码失败: {proc.stderr.decode(errors='replace')}")

    with io.BytesIO(proc.stdout) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def find_chunk_boundaries(
    total_duration: float,
    max_chunk_sec: float = 300.0,
    wav: Optional[np.ndarray] = None,
    sr: int = 16000,
) -> List[Tuple[float, float]]:
    """
    将长音频按时间切片，并在预定切割点附近寻找低能量/静音点，防止破坏单词或语音流
    """
    if total_duration <= max_chunk_sec:
        return [(0.0, total_duration)]

    boundaries = [0.0]
    cur = 0.0
    while cur + max_chunk_sec < total_duration:
        target = cur + max_chunk_sec
        split_point = target

        if wav is not None and sr > 0:
            search_start_sec = max(cur + 60.0, target - 3.0)
            search_end_sec = min(total_duration - 1.0, target + 3.0)
            if search_end_sec > search_start_sec:
                s_idx = int(search_start_sec * sr)
                e_idx = int(search_end_sec * sr)
                segment_audio = wav[s_idx:e_idx]

                win_len = int(0.2 * sr)
                if len(segment_audio) > win_len:
                    hop = int(0.05 * sr)
                    num_windows = (len(segment_audio) - win_len) // hop
                    if num_windows > 0:
                        min_energy = float("inf")
                        best_offset = 0
                        for w_i in range(num_windows):
                            w_slice = segment_audio[w_i * hop : w_i * hop + win_len]
                            energy = float(np.mean(np.abs(w_slice)))
                            if energy < min_energy:
                                min_energy = energy
                                best_offset = w_i * hop
                        split_point = search_start_sec + (best_offset + win_len / 2) / sr

        boundaries.append(round(split_point, 2))
        cur = split_point

    boundaries.append(round(total_duration, 2))

    chunks = []
    for i in range(len(boundaries) - 1):
        if boundaries[i + 1] > boundaries[i]:
            chunks.append((boundaries[i], boundaries[i + 1]))
    return chunks


@dataclass
class FileItem:
    path: str
    name: str
    duration: float = 0.0
    status: str = "pending"  # pending, extracting, transcribing, done, skipped, failed, cancelled
    progress: float = 0.0    # 单文件独立进度 (0 ~ 100.0%)
    current_chunk: int = 0
    total_chunks: int = 0
    srt_path: str = ""
    txt_path: str = ""
    json_path: str = ""
    text_preview: str = ""
    segments_count: int = 0
    correction_count: int = 0
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TaskState:
    task_id: str
    task_type: str  # server_batch 或 upload
    status: str = "pending"  # pending, running, completed, completed_with_errors, failed, cancelled
    progress: float = 0.0  # 0 ~ 100
    current_file: str = ""
    total_files: int = 0
    completed_files: int = 0
    total_audio_duration: float = 0.0
    processed_audio_duration: float = 0.0
    elapsed_time: float = 0.0
    speed_rtf: float = 0.0  # 实时倍速 (已处理音频时长 / 已消耗总用时)
    files: List[FileItem] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    params: Dict[str, Any] = field(default_factory=dict)
    is_cancelled: bool = False



class TaskManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.tasks: Dict[str, TaskState] = {}
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.cpu_pool = ThreadPoolExecutor(max_workers=max(8, os.cpu_count() or 8))
        self._load_persisted_tasks()

    def _get_task_dir(self, task_id: str) -> Path:
        p = TASKS_DIR / task_id
        p.mkdir(parents=True, exist_ok=True)
        (p / "inputs").mkdir(parents=True, exist_ok=True)
        (p / "outputs").mkdir(parents=True, exist_ok=True)
        return p

    def _load_persisted_tasks(self):
        """服务启动时扫描并加载历史任务"""
        try:
            for task_folder in TASKS_DIR.iterdir():
                if not task_folder.is_dir():
                    continue
                meta_file = task_folder / "task.json"
                if meta_file.exists():
                    try:
                        data = json.loads(meta_file.read_text(encoding="utf-8"))
                        files = [
                            FileItem(
                                path=f.get("path", ""),
                                name=f.get("name", ""),
                                duration=f.get("duration", 0.0),
                                status=f.get("status", "done"),
                                progress=f.get("progress", 100.0 if f.get("status") in ("done", "skipped") else 0.0),
                                current_chunk=f.get("current_chunk", 0),
                                total_chunks=f.get("total_chunks", 0),
                                srt_path=f.get("srt_path", ""),
                                txt_path=f.get("txt_path", ""),
                                json_path=f.get("json_path", ""),
                                text_preview=f.get("text_preview", ""),
                                segments_count=f.get("segments_count", 0),
                                correction_count=f.get("correction_count", 0),
                                corrections=f.get("corrections", []),
                                error=f.get("error"),
                            )
                            for f in data.get("files", [])
                        ]
                        task = TaskState(
                            task_id=data["task_id"],
                            task_type=data.get("task_type", "upload"),
                            status=data.get("status", "completed"),
                            progress=data.get("progress", 100.0),
                            current_file=data.get("current_file", ""),
                            total_files=data.get("total_files", len(files)),
                            completed_files=data.get("completed_files", len(files)),
                            total_audio_duration=data.get("total_audio_duration", 0.0),
                            processed_audio_duration=data.get("processed_audio_duration", 0.0),
                            elapsed_time=data.get("elapsed_time", 0.0),
                            speed_rtf=data.get("speed_rtf", 0.0),
                            files=files,
                            created_at=data.get("created_at", task_folder.stat().st_mtime),
                            params=data.get("params", {}),
                            is_cancelled=data.get("is_cancelled", False),
                        )
                        self.tasks[task.task_id] = task
                    except Exception as e:
                        print(f"[TaskManager] 加载历史任务 {task_folder.name} 失败: {e}")
        except Exception as e:
            print(f"[TaskManager] 扫描历史任务目录异常: {e}")

    def _save_task_to_disk(self, task: TaskState):
        """持久化任务状态至 task.json"""
        try:
            task_dir = self._get_task_dir(task.task_id)
            payload = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status,
                "progress": round(task.progress, 1),
                "current_file": task.current_file,
                "total_files": task.total_files,
                "completed_files": task.completed_files,
                "total_audio_duration": round(task.total_audio_duration, 1),
                "processed_audio_duration": round(task.processed_audio_duration, 1),
                "elapsed_time": round(task.elapsed_time, 1),
                "speed_rtf": round(task.speed_rtf, 1),
                "created_at": task.created_at,
                "params": task.params,
                "is_cancelled": task.is_cancelled,
                "files": [asdict(f) for f in task.files],
            }
            (task_dir / "task.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[TaskManager] 保存任务 {task.task_id} 异常: {e}")


    def create_task(
        self,
        task_type: str,
        files: List[Dict[str, Any]],
        params: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> TaskState:
        if not task_id:
            task_id = str(uuid.uuid4())[:8]
        self._get_task_dir(task_id)

        file_items = [
            FileItem(
                path=f["path"],
                name=f.get("name", Path(f["path"]).name),
                duration=f.get("duration", 0.0),
                status=f.get("status", "pending"),
                segments_count=int(f.get("segments_count", 0) or 0),
            )
            for f in files
        ]

        total_dur = sum(f.duration for f in file_items)

        task = TaskState(
            task_id=task_id,
            task_type=task_type,
            status="pending",
            total_files=len(file_items),
            total_audio_duration=total_dur,
            files=file_items,
            params=params,
        )
        self.tasks[task_id] = task
        self.subscribers[task_id] = []
        self._save_task_to_disk(task)
        return task

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.tasks.get(task_id)
        if not task:
            task_file = TASKS_DIR / task_id / "task.json"
            if task_file.exists():
                try:
                    return json.loads(task_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return None

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": round(task.progress, 1),
            "current_file": task.current_file,
            "total_files": task.total_files,
            "completed_files": task.completed_files,
            "total_audio_duration": round(task.total_audio_duration, 1),
            "processed_audio_duration": round(task.processed_audio_duration, 1),
            "elapsed_time": round(task.elapsed_time, 1),
            "speed_rtf": round(task.speed_rtf, 1),
            "created_at": task.created_at,
            "params": task.params,
            "files": [asdict(f) for f in task.files],
        }

    def list_tasks(self, limit: int = 30) -> List[Dict[str, Any]]:
        """获取所有历史任务简要列表"""
        task_list = []
        for tid, t in self.tasks.items():
            task_list.append({
                "task_id": t.task_id,
                "task_type": t.task_type,
                "status": t.status,
                "progress": round(t.progress, 1),
                "total_files": t.total_files,
                "completed_files": t.completed_files,
                "total_duration": round(t.total_audio_duration, 1),
                "created_at": t.created_at,
            })
        task_list.sort(key=lambda x: x["created_at"], reverse=True)
        return task_list[:limit]

    def package_task_subtitles(self, task_id: str) -> Optional[Path]:
        """打包任务生成的所有 .srt 字幕为 tar.gz 压缩包"""
        task = self.tasks.get(task_id)
        task_dir = self._get_task_dir(task_id)
        outputs_dir = task_dir / "outputs"
        tar_path = task_dir / f"{task_id}_subtitles.tar.gz"

        # 收集所有字幕文件
        srt_files_to_pack = []
        if outputs_dir.exists():
            for f in outputs_dir.glob("*.srt"):
                if f.stat().st_size > 0:
                    srt_files_to_pack.append((f, f.name))

        if task:
            for item in task.files:
                if item.srt_path:
                    p = Path(item.srt_path)
                    if p.exists() and p.stat().st_size > 0:
                        arcname = f"{Path(item.name).stem}.srt"
                        if not any(name == arcname for _, name in srt_files_to_pack):
                            srt_files_to_pack.append((p, arcname))

        if not srt_files_to_pack:
            return None

        with tarfile.open(tar_path, "w:gz") as tar:
            for filepath, arcname in srt_files_to_pack:
                tar.add(filepath, arcname=arcname)

        return tar_path

    def cancel_task(self, task_id: str) -> bool:
        """强制取消/终止正在进行的任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.is_cancelled = True
        task.status = "cancelled"
        for f in task.files:
            if f.status in ("pending", "extracting", "transcribing", "correcting"):
                f.status = "cancelled"
                if f.progress <= 0:
                    f.progress = 0.0

        self.notify(task_id)
        return True

    def delete_task(self, task_id: str, delete_files: bool = True) -> bool:
        """删除指定历史任务，并可彻底清除其占用磁盘的音视频与字幕文件"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.is_cancelled = True
            del self.tasks[task_id]

        if task_id in self.subscribers:
            del self.subscribers[task_id]

        task_dir = TASKS_DIR / task_id
        if delete_files and task_dir.exists():
            try:
                shutil.rmtree(task_dir, ignore_errors=True)
            except Exception as e:
                print(f"[TaskManager] 删除任务目录 {task_id} 失败: {e}")
        return True

    def clear_history(self, delete_files: bool = True) -> int:
        """一键清空所有历史任务并释放垃圾磁盘空间"""
        deleted_count = 0
        all_ids = list(self.tasks.keys())
        for tid in all_ids:
            t = self.tasks.get(tid)
            if t and t.status in ("completed", "completed_with_errors", "failed", "cancelled"):
                self.delete_task(tid, delete_files=delete_files)
                deleted_count += 1

        # 扫描并清理 TASKS_DIR 下未在内存中的所有孤儿残留目录
        if delete_files and TASKS_DIR.exists():
            for folder in TASKS_DIR.iterdir():
                if folder.is_dir() and folder.name not in self.tasks:
                    try:
                        shutil.rmtree(folder, ignore_errors=True)
                        deleted_count += 1
                    except Exception:
                        pass
        return deleted_count

    def notify(self, task_id: str):
        """异步通知所有订阅者当前任务状态更新并落盘"""

        task = self.tasks.get(task_id)
        if not task:
            return

        payload = json.dumps({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": round(task.progress, 1),
            "current_file": task.current_file,
            "total_files": task.total_files,
            "completed_files": task.completed_files,
            "total_audio_duration": task.total_audio_duration,
            "processed_audio_duration": round(task.processed_audio_duration, 1),
            "elapsed_time": round(task.elapsed_time, 1),
            "speed_rtf": round(task.speed_rtf, 1),
            "files": [asdict(f) for f in task.files],
        }, ensure_ascii=False)

        subs = self.subscribers.get(task_id, [])
        for q in subs:
            try:
                q.put_nowait(payload)
            except Exception:
                pass

        self._save_task_to_disk(task)

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        if task_id not in self.subscribers:
            self.subscribers[task_id] = []
        self.subscribers[task_id].append(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue):
        if task_id in self.subscribers and q in self.subscribers[task_id]:
            self.subscribers[task_id].remove(q)

    def start_task(self, task_id: str):
        """在后台线程启动流水线执行"""
        t = threading.Thread(target=self._run_task_pipeline, args=(task_id,), daemon=True)
        t.start()

    def _make_correction_recorder(self, item: FileItem, task_id: str):
        """构造纠错记录回调：累计「原 → 改」样例，并节流推送 SSE 让看板卡片实时刷新"""
        state = {"last_notify": 0.0}

        def record(idx: int, orig: str, corrected: str):
            item.correction_count += 1
            item.corrections.append({"idx": idx, "from": orig, "to": corrected})
            if len(item.corrections) > MAX_CORRECTION_PREVIEW:
                del item.corrections[:-MAX_CORRECTION_PREVIEW]
            item.text_preview = (
                f"已修正 {item.correction_count} 处：{orig[:24]} → {corrected[:24]}"
            )
            now = time.time()
            if now - state["last_notify"] >= 1.5:
                state["last_notify"] = now
                self.notify(task_id)

        return record

    def _run_task_pipeline(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task:
            return

        # 已有 ASR JSON 文件的大模型纠错任务（与转录任务共用同一套看板与 SSE 进度）
        if task.task_type == "json_correction":
            return self._run_json_correction(task_id)

        task.status = "running"
        t_start = time.time()
        self.notify(task_id)

        task_dir = self._get_task_dir(task_id)
        outputs_dir = task_dir / "outputs"

        params = task.params
        force = params.get("force", True)
        srt_folder = params.get("srt_folder", False)
        language = params.get("language", None)
        max_duration = float(params.get("max_duration", 6.0))
        gap_threshold = float(params.get("gap_threshold", 0.3))
        concurrency = int(params.get("concurrency", 5))
        enable_llm = bool(params.get("enable_llm", True))
        llm_api_base = params.get("llm_api_base", "http://127.0.0.1:8002/v1")

        # 确保推理引擎就绪
        if not engine.is_loaded:
            engine.load_model()

        task_stat_lock = threading.Lock()

        def process_single_file(item: FileItem):
            if task.is_cancelled:
                item.status = "cancelled"
                return

            file_path = Path(item.path)
            item.progress = 0.0

            # 决定最终字幕保存路径
            if task.task_type == "server_batch":
                if srt_folder:
                    out_dir = file_path.parent / "srt"
                    out_dir.mkdir(exist_ok=True)
                    target_srt = out_dir / (file_path.stem + ".srt")
                    target_txt = out_dir / (file_path.stem + ".txt")
                    target_json = out_dir / (file_path.stem + ".json")
                else:
                    target_srt = file_path.with_suffix(".srt")
                    target_txt = file_path.with_suffix(".txt")
                    target_json = file_path.with_suffix(".json")
            else:
                target_srt = outputs_dir / f"{file_path.stem}.srt"
                target_txt = outputs_dir / f"{file_path.stem}.txt"
                target_json = outputs_dir / f"{file_path.stem}.json"

            item.srt_path = str(target_srt)
            item.txt_path = str(target_txt)
            item.json_path = str(target_json)

            # 跳过检查 (如果不强制覆盖且字幕文件已存在)
            if not force and target_srt.exists() and target_srt.stat().st_size > 0:
                item.status = "skipped"
                item.progress = 100.0
                task_out_srt = outputs_dir / f"{file_path.stem}.srt"
                if not task_out_srt.exists():
                    try:
                        shutil.copy2(target_srt, task_out_srt)
                    except Exception:
                        pass

                with task_stat_lock:
                    task.completed_files += 1
                    task.processed_audio_duration += item.duration
                    task.elapsed_time = time.time() - t_start
                    if task.elapsed_time > 0:
                        task.speed_rtf = task.processed_audio_duration / task.elapsed_time
                    task.progress = min(100.0, (task.completed_files / task.total_files) * 100.0)
                self.notify(task_id)
                return

            # 1. CPU 异步抽取音频
            if task.is_cancelled:
                item.status = "cancelled"
                return

            item.status = "extracting"
            item.progress = 5.0
            with task_stat_lock:
                task.current_file = item.name
            self.notify(task_id)

            try:
                future = self.cpu_pool.submit(extract_audio_pcm, str(file_path))
                wav, sr = future.result()
                actual_dur = len(wav) / sr
                if item.duration <= 0:
                    item.duration = round(actual_dur, 2)
            except Exception as e:
                item.status = "failed"
                item.error = f"音频抽取失败: {str(e)}"
                item.progress = 0.0
                with task_stat_lock:
                    task.completed_files += 1
                self.notify(task_id)
                return

            if task.is_cancelled:
                item.status = "cancelled"
                return

            # 2. GPU 识别与对齐字幕生成
            item.status = "transcribing"
            item.progress = 10.0
            with task_stat_lock:
                task.current_file = item.name
            self.notify(task_id)

            try:
                total_sec = len(wav) / sr
                chunks = find_chunk_boundaries(total_sec, max_chunk_sec=300.0, wav=wav, sr=sr)
                num_chunks = len(chunks)
                item.total_chunks = num_chunks

                all_segments: List[Dict[str, Any]] = []
                all_words: List[Dict[str, Any]] = []

                for c_idx, (s_sec, e_sec) in enumerate(chunks):
                    if task.is_cancelled:
                        item.status = "cancelled"
                        break

                    item.current_chunk = c_idx + 1

                    s_sample = int(s_sec * sr)
                    e_sample = int(e_sec * sr)
                    sub_wav = wav[s_sample:e_sample]

                    sub_res = engine.transcribe(
                        audio=(sub_wav, sr),
                        language=language,
                        return_time_stamps=True,
                    )[0]

                    if task.is_cancelled:
                        item.status = "cancelled"
                        break

                    _, chunk_segs, _ = generate_srt(
                        sub_res,
                        max_duration=max_duration,
                        gap_threshold=gap_threshold,
                        time_offset=s_sec,
                    )
                    chunk_json = generate_json(sub_res, time_offset=s_sec)

                    all_segments.extend(chunk_segs)
                    all_words.extend(chunk_json.get("word_timestamps", []))

                    # 阶段性保存字幕文件（实时可查可下）
                    export_final_subtitles(
                        all_segments,
                        all_words,
                        target_srt=str(target_srt),
                        target_txt=str(target_txt),
                        target_json=str(target_json),
                        language=language or "Chinese",
                    )

                    task_out_srt = outputs_dir / f"{file_path.stem}.srt"
                    task_out_txt = outputs_dir / f"{file_path.stem}.txt"
                    task_out_json = outputs_dir / f"{file_path.stem}.json"
                    if str(task_out_srt) != str(target_srt):
                        try:
                            shutil.copy2(target_srt, task_out_srt)
                            shutil.copy2(target_txt, task_out_txt)
                            shutil.copy2(target_json, task_out_json)
                        except Exception:
                            pass

                    item.segments_count = len(all_segments)
                    item.text_preview = all_segments[-1]["text"] if all_segments else ""

                    # 单文件独立进度与总耗时/倍速计算
                    item.progress = round(((c_idx + 1) / num_chunks) * 100.0, 1)

                    chunk_dur = e_sec - s_sec
                    with task_stat_lock:
                        task.processed_audio_duration += chunk_dur
                        task.elapsed_time = time.time() - t_start
                        if task.elapsed_time > 0:
                            task.speed_rtf = task.processed_audio_duration / task.elapsed_time
                        task.progress = min(
                            99.0,
                            sum(f.progress for f in task.files) / max(1, len(task.files)),
                        )

                    self.notify(task_id)

                if not task.is_cancelled and item.status != "failed":
                    if enable_llm and all_segments:
                        item.status = "correcting"
                        item.text_preview = "大模型领域画像分析与逐行纠错中..."
                        with task_stat_lock:
                            task.current_file = f"{item.name} (LLM纠错中)"
                        self.notify(task_id)

                        def llm_cb(cur, tot):
                            item.progress = min(99.0, round(90.0 + (cur / max(1, tot)) * 9.0, 1))
                            self.notify(task_id)

                        try:
                            workers_for_file = max(1, 20 // max_workers)
                            corrected_segs, _ = correct_segments_pipeline(
                                segments=all_segments,
                                full_text="",
                                api_base=llm_api_base,
                                workers=workers_for_file,
                                progress_callback=llm_cb,
                                should_stop=lambda: task.is_cancelled,
                                on_correction=self._make_correction_recorder(item, task_id),
                            )
                            all_segments = corrected_segs

                            # 覆盖输出纠错后的字幕与文本
                            export_final_subtitles(
                                all_segments,
                                all_words,
                                target_srt=str(target_srt),
                                target_txt=str(target_txt),
                                target_json=str(target_json),
                                language=language or "Chinese",
                            )

                            task_out_srt = outputs_dir / f"{file_path.stem}.srt"
                            task_out_txt = outputs_dir / f"{file_path.stem}.txt"
                            task_out_json = outputs_dir / f"{file_path.stem}.json"
                            if str(task_out_srt) != str(target_srt):
                                try:
                                    shutil.copy2(target_srt, task_out_srt)
                                    shutil.copy2(target_txt, task_out_txt)
                                    shutil.copy2(target_json, task_out_json)
                                except Exception:
                                    pass

                            if all_segments:
                                item.text_preview = all_segments[-1]["text"]
                        except CorrectionCancelled:
                            item.status = "cancelled"
                            with task_stat_lock:
                                task.completed_files += 1
                                task.progress = min(
                                    100.0, sum(f.progress for f in task.files) / max(1, len(task.files))
                                )
                            self.notify(task_id)
                            return
                        except Exception as e:
                            print(f"[TaskQueue] LLM 纠错异常 (保留原ASR字幕): {e}")

                    item.status = "cancelled" if task.is_cancelled else "done"
                    item.progress = 100.0

            except Exception as e:
                item.status = "failed"
                item.error = f"识别处理失败: {str(e)}"
                item.progress = 0.0

            with task_stat_lock:
                task.completed_files += 1
                task.elapsed_time = time.time() - t_start
                if task.elapsed_time > 0:
                    task.speed_rtf = task.processed_audio_duration / task.elapsed_time
                task.progress = min(
                    100.0,
                    sum(f.progress for f in task.files) / max(1, len(task.files)),
                )

            self.notify(task_id)

        try:
            # 使用可配置的并发线程池处理任务下的多视频文件（最大化压榨显卡吞吐与消除 CPU 等待）
            max_workers = max(1, min(concurrency, 16))
            with ThreadPoolExecutor(max_workers=max_workers) as file_executor:
                futures = [file_executor.submit(process_single_file, item) for item in task.files]
                for fut in futures:
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[TaskQueue] 文件处理线程异常: {e}")

            # 评估最终任务整体状态
            if task.is_cancelled:
                task.status = "cancelled"
            else:
                has_failed = any(f.status == "failed" for f in task.files)
                has_done = any(f.status in ("done", "skipped") for f in task.files)
                if has_failed and not has_done:
                    task.status = "failed"
                elif has_failed and has_done:
                    task.status = "completed_with_errors"
                else:
                    task.status = "completed"

            task.progress = 100.0 if task.status in ("completed", "completed_with_errors") else task.progress
            task.elapsed_time = time.time() - t_start
            if task.elapsed_time > 0:
                task.speed_rtf = task.processed_audio_duration / task.elapsed_time
            self.notify(task_id)

        except Exception as e:
            if not task.is_cancelled:
                task.status = "failed"
            self.notify(task_id)
            print(f"[TaskQueue] 任务异常: {e}")

    # ------------------------------------------------------------------
    # 已有 ASR JSON 文件纠错流水线
    # ------------------------------------------------------------------
    def _run_json_correction(self, task_id: str):
        """对服务器上已有的 ASR JSON 执行 LLM 智能纠错，原地覆盖 SRT/TXT/JSON 并同步进度。"""
        task = self.tasks.get(task_id)
        if not task:
            return

        params = task.params
        batch_size = int(params.get("batch_size", 25))
        concurrency = max(1, min(int(params.get("concurrency", 5)), 20))
        llm_api_base = params.get("llm_api_base", "http://127.0.0.1:8002/v1")
        llm_model = params.get("llm_model", "auto")
        max_workers = max(1, min(concurrency, len(task.files) or 1))

        task.status = "running"
        t_start = time.time()
        self.notify(task_id)

        task_dir = self._get_task_dir(task_id)
        outputs_dir = task_dir / "outputs"
        cache_dir = task_dir / "llm_cache"
        stat_lock = threading.Lock()

        def recompute_progress():
            task.elapsed_time = time.time() - t_start
            task.progress = min(
                100.0,
                sum(f.progress for f in task.files) / max(1, len(task.files)),
            )

        def process_single_json(item: FileItem):
            if task.is_cancelled:
                item.status = "cancelled"
                return

            src = Path(item.path)
            if not src.exists():
                item.status = "failed"
                item.error = "JSON 文件不存在"
                with stat_lock:
                    task.completed_files += 1
                    recompute_progress()
                self.notify(task_id)
                return

            item.status = "correcting"
            item.progress = 3.0
            item.text_preview = "Stage 1 领域画像分析中..."
            with stat_lock:
                task.current_file = item.name
                recompute_progress()
            self.notify(task_id)

            def llm_cb(cur: int, tot: int):
                # Stage 2 逐行纠错占整体进度 10% ~ 99%
                item.progress = min(99.0, round(10.0 + (cur / max(1, tot)) * 89.0, 1))
                with stat_lock:
                    recompute_progress()
                self.notify(task_id)

            try:
                workers_for_file = max(1, 20 // max_workers)
                result = process_json_file_standalone(
                    json_path=str(src),
                    api_base=llm_api_base,
                    model=llm_model,
                    batch_size=batch_size,
                    workers=workers_for_file,
                    cache_dir=str(cache_dir),
                    progress_callback=llm_cb,
                    should_stop=lambda: task.is_cancelled,
                    on_correction=self._make_correction_recorder(item, task_id),
                )
            except CorrectionCancelled:
                item.status = "cancelled"
                item.text_preview = "任务已被用户终止"
                with stat_lock:
                    task.completed_files += 1
                    recompute_progress()
                self.notify(task_id)
                return
            except Exception as e:
                item.status = "failed"
                item.error = f"纠错失败: {str(e)}"
                item.progress = 0.0
                with stat_lock:
                    task.completed_files += 1
                    recompute_progress()
                self.notify(task_id)
                return

            if task.is_cancelled:
                item.status = "cancelled"
                return

            # 结果同时拷贝到任务目录，便于看板内直接下载
            srt_path = outputs_dir / f"{src.stem}.srt"
            txt_path = outputs_dir / f"{src.stem}.txt"
            json_path = outputs_dir / f"{src.stem}.json"
            for source, target in (
                (result.get("srt_path"), srt_path),
                (result.get("txt_path"), txt_path),
                (result.get("json_path"), json_path),
            ):
                try:
                    if source and Path(source).exists():
                        shutil.copy2(source, target)
                except Exception:
                    pass

            item.srt_path = str(result.get("srt_path") or "")
            item.txt_path = str(result.get("txt_path") or "")
            item.json_path = str(result.get("json_path") or "")
            item.segments_count = int(result.get("segments_count") or 0)
            item.text_preview = (
                f"领域【{result.get('domain') or '通用综合'}】"
                f"纠错完成 {item.segments_count} 行字幕"
            )
            item.status = "cancelled" if task.is_cancelled else "done"
            item.progress = 100.0
            with stat_lock:
                task.completed_files += 1
                recompute_progress()
            self.notify(task_id)

        try:
            max_workers = max(1, min(concurrency, len(task.files) or 1))
            with ThreadPoolExecutor(max_workers=max_workers) as file_executor:
                futures = [file_executor.submit(process_single_json, item) for item in task.files]
                for fut in futures:
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[TaskQueue] JSON 纠错线程异常: {e}")

            if task.is_cancelled:
                task.status = "cancelled"
            else:
                has_failed = any(f.status == "failed" for f in task.files)
                has_done = any(f.status == "done" for f in task.files)
                if has_failed and not has_done:
                    task.status = "failed"
                elif has_failed and has_done:
                    task.status = "completed_with_errors"
                else:
                    task.status = "completed"

            task.progress = 100.0 if task.status in ("completed", "completed_with_errors") else task.progress
            task.elapsed_time = time.time() - t_start
            self.notify(task_id)
        except Exception as e:
            if not task.is_cancelled:
                task.status = "failed"
            self.notify(task_id)
            print(f"[TaskQueue] JSON 纠错任务异常: {e}")



task_manager = TaskManager()
