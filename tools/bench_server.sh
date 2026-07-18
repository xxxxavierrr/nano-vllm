#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/huggingface/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-0.6B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

usage() {
    cat <<'EOF'
Usage:
  tools/bench_server.sh online [benchmark options]
  tools/bench_server.sh offline [benchmark options]

Environment overrides:
  PYTHON_BIN, MODEL_PATH, SERVED_MODEL_NAME, BASE_URL

Examples:
  tools/bench_server.sh online --profile smoke
  tools/bench_server.sh online --profile throughput --max-concurrency 16
  tools/bench_server.sh offline --num-requests 64 --input-len 256 --output-len 128
EOF
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 1
fi

MODE="${1:-}"
if [[ "$MODE" != "online" && "$MODE" != "offline" ]]; then
    usage >&2
    exit 2
fi
shift

cd "$REPO_ROOT"
if [[ "$MODE" == "online" ]]; then
    if ! "$PYTHON_BIN" -c "import httpx" >/dev/null 2>&1; then
        echo "Online benchmark requires httpx; install the benchmark extra first." >&2
        exit 1
    fi
    exec "$PYTHON_BIN" benchmark.py \
        --mode online \
        --base-url "$BASE_URL" \
        --model "$SERVED_MODEL_NAME" \
        "$@"
fi

if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    echo "The online server is using the GPU at $BASE_URL." >&2
    echo "Stop it before running an offline benchmark, or use a different GPU." >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    exit 1
fi
exec "$PYTHON_BIN" benchmark.py --mode offline "$MODEL_PATH" "$@"
