#!/bin/bash
set -euo pipefail

PORT=8888
HOST="127.0.0.1"
MODEL_PATH="/home/scott/.lmstudio/models/IFM/K2-Horizon-0.9B-GGUF/K2-Horizon-1B-BF16.gguf"
SERVER_BIN="/tmp/llama-cpp-upstream/build/bin/llama-server"
LOG_FILE="/tmp/k2h_server.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

start_server() {
    if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q ok; then
        warn "K2-Horizon server already running on port ${PORT}"
        return 0
    fi

    info "Starting K2-Horizon-0.9B server on ${HOST}:${PORT}..."
    export LD_LIBRARY_PATH="/tmp/llama-cpp-upstream/build/bin:/home/scott/ROCmFPX/build/bin:$LD_LIBRARY_PATH"
    nohup "${SERVER_BIN}" \
        -m "${MODEL_PATH}" \
        -c 4096 --host "${HOST}" --port "${PORT}" \
        --no-warmup --no-webui \
        > "${LOG_FILE}" 2>&1 &
    echo $! > /tmp/k2h_server.pid
    
    info "Waiting for server to be ready..."
    for i in $(seq 1 30); do
        if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q ok; then
            info "Server is ready!"
            return 0
        fi
        sleep 1
    done
    error "Server failed to start after 30 seconds"
    cat "${LOG_FILE}" | tail -20
    return 1
}

stop_server() {
    if [ -f /tmp/k2h_server.pid ]; then
        PID=$(cat /tmp/k2h_server.pid)
        if kill -0 "$PID" 2>/dev/null; then
            info "Stopping K2-Horizon server (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
            sleep 2
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f /tmp/k2h_server.pid
    fi
    pkill -f "llama-server.*port 8888" 2>/dev/null || true
    info "Server stopped."
}

status() {
    if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q ok; then
        info "K2-Horizon server is RUNNING on ${HOST}:${PORT}"
    else
        error "K2-Horizon server is NOT RUNNING"
    fi
}

benchmark_completion() {
    info "Running completion benchmark..."
    local url="http://${HOST}:${PORT}/completion"
    
    info "Test 1: Simple completion..."
    curl -s -X POST "${url}" -H "Content-Type: application/json" \
        -d '{"prompt": "The capital of France is", "n_predict": 20, "temperature": 0.1}' > /tmp/k2h_test1.json 2>/dev/null
    info "Simple completion done."
    
    info "Test 2: Reasoning..."
    curl -s -X POST "${url}" -H "Content-Type: application/json" \
        -d '{"prompt": "Explain the difference between a compiler and interpreter, then write Python code to add two numbers.", "n_predict": 50, "temperature": 0.1}' > /tmp/k2h_test2.json 2>/dev/null
    info "Reasoning done."
    
    info "Test 3: Chat API..."
    curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model": "k2-horizon", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 20, "temperature": 0.1}' > /tmp/k2h_test3.json 2>/dev/null
    info "Chat API done."
    
    info "Benchmark complete!"
}

benchmark_api() {
    info "Running OpenAI-compatible API benchmark..."
    local url="http://${HOST}:${PORT}/v1/chat/completions"
    
    info "Testing chat completions..."
    curl -s -X POST "${url}" -H "Content-Type: application/json" \
        -d '{"model": "k2-horizon", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 50, "temperature": 0.1}' > /tmp/k2h_api_test.json 2>/dev/null
    
    local tokens=$(python3 -c "import json; d=json.load(open('/tmp/k2h_api_test.json')); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "N/A")
    info "API test: ${tokens} tokens generated"
}

run_terminal_bench() {
    info "Running Terminal-Bench evaluation..."
    local tb_dir="/home/scott/.local/share/kipnerter/ruview/source/vendor/metaharness/packages/darwin-mode/bench/terminal-bench"
    
    if [ ! -d "${tb_dir}" ]; then
        error "Terminal-bench not found at ${tb_dir}"
        return 1
    fi
    
    cd "${tb_dir}"
    
    # Set PATH and PYTHONPATH for tb and darwin agent
    export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
    export PYTHONPATH="${tb_dir}:${PYTHONPATH:-}"
    export DARWIN_BASE_URL="http://127.0.0.1:8888/v1"
    
    info "Checking terminal-bench dependencies..."
    which tb >/dev/null || { warn "terminal-bench not installed"; info "Installing..."; uv tool install terminal-bench; }
    
    info "Downloading datasets..."
    tb datasets download -d terminal-bench-core==0.1.1 2>/dev/null || true
    
    info "Running eval-validation..."
    tb run -d terminal-bench-core==0.1.1 -t hello-world --agent oracle --output-path ./runs/val-oracle --cleanup 2>/dev/null || warn "Oracle validation failed"
    tb run -d terminal-bench-core==0.1.1 -t hello-world --agent nop --output-path ./runs/val-nop --cleanup 2>/dev/null || warn "NOP validation failed"
    
    info "Building manifest..."
    node build-manifest.mjs 2>/dev/null || warn "Could not build manifest"
    
    info "Running hardest-first tasks with K2-Horizon at http://127.0.0.1:8888/v1..."
    local band="${2:-}"
    local n_tasks="${3:-6}"
    BAND_ARG=""; [ -n "$band" ] && BAND_ARG="--band $band"
    node hardest-first.mjs --model "k2-horizon" --n "${n_tasks}" --base-url "http://127.0.0.1:8888/v1" $BAND_ARG \
        --max-steps 30 --per-task-cost 1.5 --concurrent 1 --out "runs/k2h-$(date +%Y%m%d_%H%M%S)"
    
    cd - > /dev/null
}

case "${1:-}" in
    start)   start_server ;;
    stop)    stop_server ;;
    status)  status ;;
    bench)   start_server; benchmark_completion; benchmark_api ;;
    terminal) start_server; run_terminal_bench "$2" "$3" ;;
    full)    start_server; benchmark_completion; benchmark_api; run_terminal_bench "$2" "$3"; stop_server ;;
    *)
        echo "Usage: $0 {start|stop|status|bench|terminal [band] [n_tasks]|full [band] [n_tasks]}"
        exit 1
        ;;
esac
