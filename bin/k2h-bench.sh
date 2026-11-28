#!/bin/bash
# K2-Horizon benchmark helper.
#
# Configuration (defaults preserve the original machine setup):
#   K2H_PORT, K2H_HOST, K2H_HEALTH_HOST, K2H_MODEL_PATH, K2H_SERVER_ROOT,
#   K2H_SERVER_BIN, K2H_LOG_FILE, K2H_PID_FILE, K2H_RUN_DIR, K2H_GATEWAY_IP,
#   K2H_TB_DIR, K2H_TB_BIN, K2H_TB_DATASET, K2H_NODE_BIN
#
#   k2h-bench.sh --check                # validate config; no services/downloads/installs
#   k2h-bench.sh --dry-run bench        # print the plan; execute nothing
#   k2h-bench.sh bench|terminal|full    # unchanged behavior when prerequisites exist
set -euo pipefail

HOST="${K2H_HOST:-0.0.0.0}"
HEALTH_HOST="${K2H_HEALTH_HOST:-127.0.0.1}"
PORT="${K2H_PORT:-8888}"
MODEL_PATH="${K2H_MODEL_PATH:-/home/scott/.lmstudio/models/IFM/K2-Horizon-0.9B-GGUF/K2-Horizon-1B-BF16.gguf}"
# Durable server install. The previous path (/tmp/llama-cpp-upstream/...) was
# deleted out from under three running llama-server processes, leaving them
# alive on unlinked inodes and unrestartable. Everything lives under /data now.
SERVER_ROOT="${K2H_SERVER_ROOT:-/data/lib/llama-server}"
SERVER_BIN="${K2H_SERVER_BIN:-${SERVER_ROOT}/llama-server}"
LOG_FILE="${K2H_LOG_FILE:-/data/logs/k2h_server.log}"
PID_FILE="${K2H_PID_FILE:-/data/run/k2h_server.pid}"
RUN_DIR="${K2H_RUN_DIR:-/data/run}"
GATEWAY_IP="${K2H_GATEWAY_IP:-172.17.0.1}"
TB_DIR="${K2H_TB_DIR:-/home/scott/.local/share/kipnerter/ruview/source/vendor/metaharness/packages/darwin-mode/bench/terminal-bench}"
TB_BIN="${K2H_TB_BIN:-/home/scott/.local/bin/tb}"
TB_DATASET="${K2H_TB_DATASET:-terminal-bench-core==0.1.1}"
NODE_BIN="${K2H_NODE_BIN:-node}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CHECK_ONLY=0
DRY_RUN=0
PROBE_SERVER=1
PREFLIGHT_FAILURES=0

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [--check] [--dry-run] {start|stop|status|bench|terminal [band] [n_tasks]|full [band] [n_tasks]}"
}

preflight_fail() {
    error "$1"
    PREFLIGHT_FAILURES=$((PREFLIGHT_FAILURES + 1))
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || preflight_fail "$1 not found: $2"
}

health_url() { echo "http://${HEALTH_HOST}:${PORT}/health"; }

server_healthy() {
    curl -s --max-time 2 "$(health_url)" 2>/dev/null | grep -q ok
}

preflight() {
    local action="$1"
    info "Preflight checks for '${action}'..."
    PREFLIGHT_FAILURES=0

    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
        preflight_fail "invalid port: ${PORT} (set K2H_PORT to a value between 1 and 65535)"
    fi
    need_cmd curl "install curl (for example: apt install curl)"

    case "${action}" in
        start|bench|terminal|full|check)
            case "${action}" in
                bench|terminal|full|check) need_cmd python3 "install python3 to parse benchmark results" ;;
            esac
            # In live mode, an already-running server makes the model/binary
            # irrelevant (start_server() will detect it and return early).
            if [ "${PROBE_SERVER}" -eq 1 ] && server_healthy; then
                info "llama-server already reachable at ${HEALTH_HOST}:${PORT}"
            else
                if [ "${PROBE_SERVER}" -eq 1 ]; then
                    warn "llama-server not reachable at ${HEALTH_HOST}:${PORT}; the start step will try to launch it"
                fi
                if [ ! -x "${SERVER_BIN}" ]; then
                    preflight_fail "llama-server binary not found: ${SERVER_BIN} (set K2H_SERVER_BIN to the llama-server executable)"
                fi
                if [ ! -r "${MODEL_PATH}" ]; then
                    preflight_fail "model file not found: ${MODEL_PATH} (set K2H_MODEL_PATH to a GGUF model)"
                fi
            fi
            if [ "${PROBE_SERVER}" -eq 0 ]; then
                info "Server reachability not checked in --check/--dry-run mode"
            fi
            ;;
    esac

    case "${action}" in
        terminal|full|check)
            if [ ! -d "${TB_DIR}" ]; then
                preflight_fail "terminal-bench directory not found: ${TB_DIR} (set K2H_TB_DIR to the darwin-mode/bench/terminal-bench checkout)"
            fi
            for script in build-manifest.mjs hardest-first.mjs score.py; do
                if [ ! -f "${TB_DIR}/${script}" ]; then
                    preflight_fail "node script missing: ${TB_DIR}/${script} (set K2H_TB_DIR to a checkout containing it)"
                fi
            done
            if [ -x "${TB_BIN}" ]; then
                info "tb CLI found at ${TB_BIN}"
            elif command -v tb >/dev/null 2>&1; then
                info "tb CLI found on PATH: $(command -v tb)"
            else
                preflight_fail "tb CLI not found: ${TB_BIN} (install with: uv tool install terminal-bench, or set K2H_TB_BIN)"
            fi
            need_cmd "${NODE_BIN}" "install Node.js to run the terminal-bench scripts"
            need_cmd docker "install Docker; terminal-bench tasks run in containers"
            ;;
    esac

    if [ "${PREFLIGHT_FAILURES}" -gt 0 ]; then
        error "Preflight found ${PREFLIGHT_FAILURES} issue(s)"
        return 1
    fi
    info "Preflight OK"
    return 0
}

dry_run_plan() {
    local action="$1"
    info "[dry-run] No services started, no files written, no downloads or installs."
    info "Config: port=${PORT} host=${HOST} model=${MODEL_PATH} server_bin=${SERVER_BIN}"
    info "Config: tb_dir=${TB_DIR} tb_bin=${TB_BIN} dataset=${TB_DATASET} gateway=${GATEWAY_IP}"
    case "${action}" in
        start)
            echo "  nohup ${SERVER_BIN} -m ${MODEL_PATH} -c 2048 --host ${HOST} --port ${PORT} --no-warmup --no-webui > ${LOG_FILE} 2>&1 &"
            echo "  poll $(health_url) up to 30 times"
            ;;
        stop)
            echo "  kill \$(cat ${PID_FILE}) and pkill -f 'llama-server.*port ${PORT}'"
            ;;
        status)
            echo "  curl $(health_url)"
            ;;
        bench)
            echo "  start server if needed"
            echo "  POST http://${HOST}:${PORT}/completion (3 prompts) -> ${RUN_DIR}/k2h_test{1,2,3}.json"
            echo "  POST http://${HOST}:${PORT}/v1/chat/completions -> ${RUN_DIR}/k2h_api_test.json"
            ;;
        terminal)
            echo "  cd ${TB_DIR}"
            echo "  DARWIN_BASE_URL=http://${GATEWAY_IP}:${PORT}/v1 ${TB_BIN} datasets download -d ${TB_DATASET}"
            echo "  ${TB_BIN} run -d ${TB_DATASET} -t hello-world --agent oracle|nop --cleanup"
            echo "  ${NODE_BIN} build-manifest.mjs"
            echo "  ${NODE_BIN} hardest-first.mjs --model k2-horizon --n \${n_tasks} --base-url http://${GATEWAY_IP}:${PORT}/v1 --band \${band}"
            ;;
        full)
            echo "  bench steps, then terminal steps, then stop server"
            ;;
    esac
}

start_server() {
    if server_healthy; then
        warn "K2-Horizon server already running on port ${PORT}"
        return 0
    fi
    if [ ! -x "${SERVER_BIN}" ]; then
        error "llama-server binary not found: ${SERVER_BIN} (set K2H_SERVER_BIN)"
        return 1
    fi
    if [ ! -r "${MODEL_PATH}" ]; then
        error "model file not found: ${MODEL_PATH} (set K2H_MODEL_PATH)"
        return 1
    fi
    info "Starting K2-Horizon-0.9B server on ${HOST}:${PORT}..."
    export LD_LIBRARY_PATH="${SERVER_ROOT}:${LD_LIBRARY_PATH:-}"
    mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"
    nohup "${SERVER_BIN}" \
        -m "${MODEL_PATH}" \
        -c 2048 --host "${HOST}" --port "${PORT}" \
        --no-warmup --no-webui \
        > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    info "Waiting for server to be ready..."
    for i in $(seq 1 30); do
        if server_healthy; then
            info "Server is ready! (accessible at http://${GATEWAY_IP}:${PORT})"
            return 0
        fi
        sleep 1
    done
    error "Server failed to start after 30 seconds; last lines of ${LOG_FILE}:"
    tail -20 "${LOG_FILE}" || true
    return 1
}

stop_server() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if kill -0 "$PID" 2>/dev/null; then
            info "Stopping K2-Horizon server (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
            sleep 2
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "${PID_FILE}"
    fi
    pkill -f "llama-server.*port ${PORT}" 2>/dev/null || true
    info "Server stopped."
}

status() {
    if server_healthy; then
        info "K2-Horizon server is RUNNING on ${HOST}:${PORT} (gateway: ${GATEWAY_IP}:${PORT})"
    else
        error "K2-Horizon server is NOT RUNNING at ${HEALTH_HOST}:${PORT}"
    fi
}

benchmark_completion() {
    info "Running completion benchmark..."
    local url="http://${HEALTH_HOST}:${PORT}/completion"
    mkdir -p "${RUN_DIR}"
    info "Test 1: Simple completion..."
    curl -s -X POST "${url}" -H "Content-Type: application/json" \
        -d '{"prompt": "The capital of France is", "n_predict": 20, "temperature": 0.1}' > "${RUN_DIR}/k2h_test1.json" 2>/dev/null
    info "Simple completion done."
    info "Test 2: Reasoning..."
    curl -s -X POST "${url}" -H "Content-Type: application/json" \
        -d '{"prompt": "Explain the difference between a compiler and interpreter, then write Python code to add two numbers.", "n_predict": 50, "temperature": 0.1}' > "${RUN_DIR}/k2h_test2.json" 2>/dev/null
    info "Reasoning done."
    info "Test 3: Chat API..."
    curl -s -X POST "http://${HEALTH_HOST}:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model": "k2-horizon", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 20, "temperature": 0.1}' > "${RUN_DIR}/k2h_test3.json" 2>/dev/null
    info "Chat API done."
    info "Benchmark complete!"
}

benchmark_api() {
    info "Running OpenAI-compatible API benchmark..."
    local url="http://${HEALTH_HOST}:${PORT}/v1/chat/completions"
    curl -s -X POST "${url}" -H "Content-Type: application/json" \
        -d '{"model": "k2-horizon", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 50, "temperature": 0.1}' > "${RUN_DIR}/k2h_api_test.json" 2>/dev/null
    local tokens
    tokens=$(python3 -c "import json; d=json.load(open('${RUN_DIR}/k2h_api_test.json')); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "N/A")
    info "API test: ${tokens} tokens generated"
}

run_terminal_bench() {
    info "Running Terminal-Bench evaluation..."
    if [ ! -d "${TB_DIR}" ]; then
        error "Terminal-bench directory not found: ${TB_DIR} (set K2H_TB_DIR)"
        return 1
    fi
    local tb_cmd=""
    if [ -x "${TB_BIN}" ]; then
        tb_cmd="${TB_BIN}"
    elif command -v tb >/dev/null 2>&1; then
        tb_cmd="$(command -v tb)"
    else
        error "tb CLI not found: ${TB_BIN}; install with: uv tool install terminal-bench (or set K2H_TB_BIN)"
        return 1
    fi
    for script in build-manifest.mjs hardest-first.mjs; do
        if [ ! -f "${TB_DIR}/${script}" ]; then
            error "node script missing: ${TB_DIR}/${script} (set K2H_TB_DIR)"
            return 1
        fi
    done
    if ! command -v "${NODE_BIN}" >/dev/null 2>&1; then
        error "${NODE_BIN} not found on PATH; install Node.js or set K2H_NODE_BIN"
        return 1
    fi
    cd "${TB_DIR}"
    # Set PATH and environment for Docker container access
    export PATH="$(dirname "${tb_cmd}"):/usr/local/bin:/usr/bin:/bin:${PATH:-}"
    export PYTHONPATH="${TB_DIR}:${PYTHONPATH:-}"
    export DARWIN_BASE_URL="http://${GATEWAY_IP}:${PORT}/v1"
    info "Docker container URL: ${DARWIN_BASE_URL}"
    info "Downloading datasets..."
    "${tb_cmd}" datasets download -d "${TB_DATASET}" 2>/dev/null || warn "Dataset download failed; run: ${tb_cmd} datasets download -d ${TB_DATASET}"
    info "Running eval-validation..."
    "${tb_cmd}" run -d "${TB_DATASET}" -t hello-world --agent oracle --output-path ./runs/val-oracle --cleanup 2>/dev/null || warn "Oracle validation failed"
    "${tb_cmd}" run -d "${TB_DATASET}" -t hello-world --agent nop --output-path ./runs/val-nop --cleanup 2>/dev/null || warn "NOP validation failed"
    info "Building manifest..."
    "${NODE_BIN}" build-manifest.mjs 2>/dev/null || warn "Could not build manifest"
    info "Running hardest-first tasks..."
    local band="${2:-}"
    local n_tasks="${3:-6}"
    local BAND_ARG=""
    [ -n "$band" ] && BAND_ARG="--band $band"
    "${NODE_BIN}" hardest-first.mjs --model "k2-horizon" --n "${n_tasks}" --base-url "http://${GATEWAY_IP}:${PORT}/v1" $BAND_ARG \
        --max-steps 30 --per-task-cost 1.5 --concurrent 1 --out "runs/k2h-$(date +%Y%m%d_%H%M%S)"
    cd - > /dev/null
}

# Parse global flags before the action.
while [ "$#" -gt 0 ]; do
    case "$1" in
        --check|check) CHECK_ONLY=1; shift ;;
        --dry-run|dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) error "unknown option: $1"; usage >&2; exit 2 ;;
        *) break ;;
    esac
done

ACTION="${1:-}"
# `--check` on its own is a valid invocation with no action argument.
if [ "${CHECK_ONLY}" -eq 1 ] && [ -z "${ACTION}" ]; then
    ACTION=check
fi
BAND="${2:-}"
N_TASKS="${3:-6}"

case "${ACTION}" in
    start|stop|status|bench|terminal|full|check) ;;
    "") usage >&2; exit 2 ;;
    *) error "unknown action: ${ACTION}"; usage >&2; exit 2 ;;
esac

if [ "${CHECK_ONLY}" -eq 1 ]; then
    ACTION=check
fi
if [ "${CHECK_ONLY}" -eq 1 ] || [ "${DRY_RUN}" -eq 1 ]; then
    PROBE_SERVER=0
fi

if [ "${CHECK_ONLY}" -eq 1 ]; then
    preflight check || exit 2
    exit 0
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    info "[dry-run] configuration validation only"
    preflight "${ACTION}" || warn "Preflight reported issues; continuing in dry-run mode"
    dry_run_plan "${ACTION}"
    exit 0
fi

preflight "${ACTION}" || { error "Preflight failed; aborting"; exit 2; }

case "${ACTION}" in
    start)   start_server ;;
    stop)    stop_server ;;
    status)  status ;;
    bench)   start_server; benchmark_completion; benchmark_api ;;
    terminal) start_server; run_terminal_bench "$BAND" "$N_TASKS" ;;
    full)    start_server; benchmark_completion; benchmark_api; run_terminal_bench "$BAND" "$N_TASKS"; stop_server ;;
esac
