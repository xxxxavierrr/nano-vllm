#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/huggingface/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENGINE_PORT="${ENGINE_PORT:-5555}"
MASTER_PORT="${MASTER_PORT:-2333}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
PIECEWISE_MAX_TOKENS="${PIECEWISE_MAX_TOKENS:-512}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c "import fastapi, uvicorn, zmq" >/dev/null 2>&1; then
    echo "Installing the small Web serving dependencies without pip cache..."
    "$PYTHON_BIN" -m pip install --no-cache-dir fastapi uvicorn pyzmq
fi

cd "$REPO_ROOT"
echo "Starting nano-vLLM on http://$HOST:$PORT"
echo "Model: $MODEL_PATH (served as $SERVED_MODEL_NAME)"

exec "$PYTHON_BIN" -m nanovllm.serve.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --engine-port "$ENGINE_PORT" \
    --master-port "$MASTER_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --cudagraph-mode "$CUDAGRAPH_MODE" \
    --piecewise-max-tokens "$PIECEWISE_MAX_TOKENS" \
    "$@"

