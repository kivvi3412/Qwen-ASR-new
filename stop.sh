#!/bin/bash
# ==============================================================================
#  停止 Qwen-ASR 字幕服务 (8001) 与 Qwen3.5-4B 纠错服务 (8002)，并释放显存
# ==============================================================================

PORT=${PORT:-8001}
LLM_PORT=${LLM_PORT:-8002}

echo "🛑 正在停止 ASR 字幕服务与 Qwen3.5-4B 纠错服务..."
pkill -9 -f "service.server:app" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "llm4b/.venv/bin/vllm" 2>/dev/null || true
fuser -k ${PORT}/tcp 2>/dev/null || true
fuser -k ${LLM_PORT}/tcp 2>/dev/null || true
sleep 1
echo "✅ 所有服务已停止，显存已完全释放。"
