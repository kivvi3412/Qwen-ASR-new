#!/usr/bin/env python3
"""
高吞吐 ASR 与 ForcedAligner 推理引擎单例封装
针对 32GB 显卡极限吞吐调优：
- vLLM 连续批处理 (Continuous Batching)
- max_model_len=4096（彻底解决 65536 虚高长度导致的显存黑洞）
- gpu_memory_utilization=0.55（安全预留显存给 ForcedAligner 与激活值）
"""

import gc
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 默认选用极速省显存的 0.6B 模型，支持环境变量覆盖
DEFAULT_MODEL = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
DEFAULT_ALIGNER = os.environ.get("QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEFAULT_GPU_UTIL = float(os.environ.get("QWEN_GPU_UTIL", "0.20"))
DEFAULT_BATCH_SIZE = int(os.environ.get("QWEN_BATCH_SIZE", "4"))
DEFAULT_MAX_LEN = int(os.environ.get("QWEN_MAX_MODEL_LEN", "4096"))


class ASRInferenceEngine:
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
        self.asr_model = None
        self.model_name = None
        self.aligner_name = DEFAULT_ALIGNER
        self.gpu_memory_utilization = DEFAULT_GPU_UTIL
        self.max_model_len = DEFAULT_MAX_LEN
        self.max_inference_batch_size = DEFAULT_BATCH_SIZE
        self.is_loaded = False
        self._infer_lock = threading.Lock()

    def load_model(
        self,
        model_name: str = DEFAULT_MODEL,
        aligner_name: str = DEFAULT_ALIGNER,
        gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
        max_model_len: int = DEFAULT_MAX_LEN,
        max_inference_batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """加载或重载 vLLM ASR 引擎与 Forced Aligner"""
        with self._infer_lock:
            if self.is_loaded and self.model_name == model_name:
                return

            print(f"[Engine] 正在初始化 ASR 引擎 (vLLM): {model_name} ...")
            print(f"[Engine] 参数: gpu_util={gpu_memory_utilization}, max_len={max_model_len}, batch_size={max_inference_batch_size}")

            # 释放现有资源
            if self.asr_model is not None:
                del self.asr_model
                self.asr_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            from qwen_asr import Qwen3ASRModel

            # 使用调优参数初始化 vLLM + ForcedAligner
            self.asr_model = Qwen3ASRModel.LLM(
                model=model_name,
                forced_aligner=aligner_name,
                forced_aligner_kwargs={
                    "device_map": "cuda:0",
                    "dtype": "bfloat16",
                },
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=1024,
                enforce_eager=False,
            )

            self.model_name = model_name
            self.aligner_name = aligner_name
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_model_len = max_model_len
            self.max_inference_batch_size = max_inference_batch_size
            self.is_loaded = True
            print(f"[Engine] ASR 引擎就绪！")

    def transcribe(
        self,
        audio: Union[str, Tuple[np.ndarray, int], List[Union[str, Tuple[np.ndarray, int]]]],
        language: Optional[str] = None,
        return_time_stamps: bool = True,
    ) -> List[Any]:
        """
        执行语音识别与时间戳对齐
        支持传入 (waveform, sample_rate) 或本地文件路径列表
        """
        if not self.is_loaded:
            self.load_model()

        with self._infer_lock:
            return self.asr_model.transcribe(
                audio=audio,
                language=language if language and language != "Auto" else None,
                return_time_stamps=return_time_stamps,
            )

    @staticmethod
    def get_gpu_status() -> Dict[str, Any]:
        """获取当前显卡实时显存与利用率"""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                parts = [x.strip() for x in res.stdout.strip().split(",")]
                if len(parts) >= 5:
                    return {
                        "total_mb": float(parts[0]),
                        "used_mb": float(parts[1]),
                        "free_mb": float(parts[2]),
                        "utilization_gpu": float(parts[3]),
                        "temperature_c": float(parts[4]),
                    }
        except Exception:
            pass

        return {
            "total_mb": 32768.0,
            "used_mb": 0.0,
            "free_mb": 32768.0,
            "utilization_gpu": 0.0,
            "temperature_c": 0.0,
        }


# 全局单例
engine = ASRInferenceEngine()
