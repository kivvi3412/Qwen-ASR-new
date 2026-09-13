#!/bin/bash
# ==============================================================================
#  Qwen-ASR-new 一键启动脚本
#
#  目录结构（两个完全独立的 uv 环境，互不干扰）:
#    asr/     ASR 字幕服务 + Web 控制台   -> asr/.venv    (vLLM 0.14.0 + transformers 4.57.6)
#    llm4b/   Qwen3.5-4B 纠错大模型服务   -> llm4b/.venv  (vLLM 0.29.0 + transformers 5.x)
#
#  本脚本依次拉起：
#    1) 8002: Qwen3.5-4B 纠错服务（vLLM OpenAI 兼容 API，供 ASR 服务调用纠错）
#    2) 8001: ASR 字幕服务（FastAPI + Web 控制台，前台运行）
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$PWD"
ASR_DIR="$ROOT/asr"
LLM_DIR="$ROOT/llm4b"

# ── 全局环境变量 ──
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# 本机没有系统级 CUDA Toolkit（/usr/local/cuda 不存在、nvcc 不在 PATH）。
# vLLM 默认启用 FlashInfer top-k/top-p 采样器，该算子需要首次调用时 JIT 编译并依赖 nvcc，
# 缺失时引擎会在 warmup 阶段直接崩溃（Engine core initialization failed）。
# 关闭 FlashInfer 采样 → 回退 PyTorch 原生实现，字幕/短文本场景无性能损失。
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY=${HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY:-8}

# ── ASR 引擎参数 ──
export QWEN_ASR_MODEL=${QWEN_ASR_MODEL:-"Qwen/Qwen3-ASR-0.6B"}
export QWEN_ALIGNER_MODEL=${QWEN_ALIGNER_MODEL:-"Qwen/Qwen3-ForcedAligner-0.6B"}
export QWEN_GPU_UTIL=${QWEN_GPU_UTIL:-"0.25"}
export QWEN_MAX_MODEL_LEN=${QWEN_MAX_MODEL_LEN:-"4096"}
export QWEN_BATCH_SIZE=${QWEN_BATCH_SIZE:-"4"}

# ── 纠错大模型参数 (Qwen3.5-9B-AWQ) ──
export LLM_MODEL=${LLM_MODEL:-"cyankiwi/Qwen3.5-9B-AWQ-4bit"}
export LLM_SERVED_NAME=${LLM_SERVED_NAME:-"qwen3.5-9b"}
export LLM_PORT=${LLM_PORT:-8002}
export LLM_GPU_UTIL=${LLM_GPU_UTIL:-"0.45"}
export LLM_MAX_MODEL_LEN=${LLM_MAX_MODEL_LEN:-"4096"}
export LLM_MAX_NUM_SEQS=${LLM_MAX_NUM_SEQS:-"20"}

PORT=${PORT:-8001}
HOST=${HOST:-"0.0.0.0"}

# ── uv 环境准备（缺失时自动 uv sync，两套环境各自独立）──
ensure_env() {
    local dir="$1"
    if [ ! -x "$dir/.venv/bin/python" ]; then
        echo "⚡ 未检测到 $dir/.venv，正在自动同步环境 (uv sync)..."
        ( cd "$dir" && uv sync )
    fi
}
ensure_env "$LLM_DIR"
ensure_env "$ASR_DIR"

# venv 内置 CUDA 工具链兜底（pip 安装的 nvidia-cuda-nvcc 自带 nvcc，供 vLLM JIT 使用）
if ! command -v nvcc >/dev/null 2>&1; then
    for _nvcc in "$LLM_DIR"/.venv/lib/python*/site-packages/nvidia/cu*/bin/nvcc; do
        [ -x "$_nvcc" ] || continue
        export CUDA_HOME="$(cd "$(dirname "$_nvcc")/.." && pwd)"
        export PATH="$(dirname "$_nvcc"):$PATH"
        echo "🧰 未检测到系统 nvcc，已启用内置 CUDA 工具链: $CUDA_HOME"
        break
    done
    unset _nvcc
fi

# ── 1) 启动 Qwen3.5-9B 纠错服务（端口 8002）──
echo "🔍 检查 Qwen3.5-9B 纠错服务状态 (端口 ${LLM_PORT})..."
if curl -s --max-time 2 "http://127.0.0.1:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "✅ 纠错服务已在运行中 (端口 ${LLM_PORT})"
else
    echo "⚡ 正在启动 Qwen3.5-9B 纠错服务 (vLLM, 显存分配: ${LLM_GPU_UTIL}, 最大并发: ${LLM_MAX_NUM_SEQS})..."
    fuser -k ${LLM_PORT}/tcp 2>/dev/null || true

    nohup "$LLM_DIR/.venv/bin/vllm" serve "${LLM_MODEL}" \
        --host "0.0.0.0" \
        --port "${LLM_PORT}" \
        --served-model-name "${LLM_SERVED_NAME}" \
        --language-model-only \
        --reasoning-parser qwen3 \
        --max-model-len "${LLM_MAX_MODEL_LEN}" \
        --max-num-seqs "${LLM_MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${LLM_GPU_UTIL}" \
        --enable-prefix-caching \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        > "$ROOT/llm4b.log" 2>&1 &

    echo "⏳ 等待纠错模型加载权重 (实时日志: llm4b.log)..."
    for i in $(seq 1 90); do
        if curl -s --max-time 1 "http://127.0.0.1:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo "✅ Qwen3.5-9B 纠错服务已就绪！"
            break
        fi
        sleep 2
        echo -n "."
    done
    echo ""
fi

# ── 2) 启动 ASR 服务（端口 8001，前台运行）──
pkill -9 -f "service.server:app" 2>/dev/null || true
fuser -k ${PORT}/tcp 2>/dev/null || true

# 清理上一次 ASR 运行可能残留的 vLLM EngineCore 孤儿进程
# （父进程被 kill 后子进程会被 init 收养并继续占住 ~10GB 显存，
#   导致新引擎报 "Free memory on device cuda:0 ... is less than desired"）
for _pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
    _ppid=$(ps -o ppid= -p "$_pid" 2>/dev/null | tr -d ' ')
    if [ "$_ppid" = "1" ]; then
        echo "🧹 清理残留的 ASR EngineCore 进程 (pid=$_pid)"
        kill -9 "$_pid" 2>/dev/null || true
    fi
done
unset _pid _ppid
sleep 1

echo "======================================================================"
echo " 🚀 正在启动 Qwen3-ASR 极速视频字幕生成系统"
echo " 📌 ASR 核心模型: ${QWEN_ASR_MODEL} (显存 ${QWEN_GPU_UTIL})"
echo " 📌 ASR 对齐模型: ${QWEN_ALIGNER_MODEL}"
echo " 📌 纠错大模型:   ${LLM_MODEL} (端口 ${LLM_PORT})"
echo " 📌 服务地址:     http://${HOST}:${PORT}"
echo " 📌 运行环境:     asr/.venv (ASR) + llm4b/.venv (LLM，两套独立 uv 环境)"
echo "======================================================================"

cd "$ASR_DIR"
exec "$ASR_DIR/.venv/bin/python" -m uvicorn service.server:app --host "${HOST}" --port "${PORT}"
