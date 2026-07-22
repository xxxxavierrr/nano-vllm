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
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
DELTA_STATE_DTYPE="${DELTA_STATE_DTYPE:-auto}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-none}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"
MTP_MODEL="${MTP_MODEL:-}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIMULATE="${DATA_PARALLEL_SIMULATE:-0}"
DEVICE_IDS="${DEVICE_IDS:-}"

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

EXTRA_ARGS=()
if [[ -n "$MTP_MODEL" ]]; then
    EXTRA_ARGS+=(--mtp-model "$MTP_MODEL")
fi
if [[ -n "$DEVICE_IDS" ]]; then
    EXTRA_ARGS+=(--device-ids "$DEVICE_IDS")
fi
if [[ "$DATA_PARALLEL_SIMULATE" == "1" ]]; then
    EXTRA_ARGS+=(--data-parallel-simulate)
fi

exec "$PYTHON_BIN" -m nanovllm.serve.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --engine-port "$ENGINE_PORT" \
    --master-port "$MASTER_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --delta-state-dtype "$DELTA_STATE_DTYPE" \
    --speculative-method "$SPECULATIVE_METHOD" \
    --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS" \
    --data-parallel-size "$DATA_PARALLEL_SIZE" \
    --cudagraph-mode "$CUDAGRAPH_MODE" \
    --piecewise-max-tokens "$PIECEWISE_MAX_TOKENS" \
    "${EXTRA_ARGS[@]}" \
    "$@"
