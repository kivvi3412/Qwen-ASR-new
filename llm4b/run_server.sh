#!/bin/bash
# ==============================================================================
#  Qwen3.5-9B-AWQ-4bit 纠错大模型独立启动脚本 (端口 8002)
# ==============================================================================
set -e
cd "$(dirname "$0")"

export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY=8

exec uv run vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit \
  --host 0.0.0.0 --port 8002 --served-model-name qwen3.5-9b \
  --language-model-only \
  --reasoning-parser qwen3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45 \
  --max-num-seqs 20 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
