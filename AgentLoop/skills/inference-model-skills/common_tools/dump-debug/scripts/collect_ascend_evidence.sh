#!/usr/bin/env bash
set +u

LOG_DIR="${1:-${VLLM_LOG_DIR:-$PWD/vllm_logs}}"
PORT="${2:-}"

echo "=== date ==="
date

echo "=== relevant processes ==="
ps -eo pid,ppid,etime,stat,cmd | grep -E 'pytest|vllm.entrypoints|api_server|EngineCore|VLLM::' | grep -v grep || true

echo "=== listening ports ==="
if command -v ss >/dev/null 2>&1; then
  ss -ltnp 2>/dev/null | grep -E "(:${PORT:-[0-9]+})" || true
elif command -v netstat >/dev/null 2>&1; then
  netstat -ltnp 2>/dev/null | grep -E "(:${PORT:-[0-9]+})" || true
fi

if [ -n "$PORT" ]; then
  echo "=== health ==="
  curl -sS -m 3 "http://127.0.0.1:${PORT}/health" 2>&1 || true
  echo
fi

echo "=== recent logs ==="
if [ -d "$LOG_DIR" ]; then
  ls -lt "$LOG_DIR" | head -n 20
  while IFS= read -r log; do
    echo "--- $log"
    grep -n -E 'EngineCore encountered a fatal error|Traceback|RuntimeError|AssertionError|AclNN_Parameter_Error|npu_grouped_matmul|aclnn[A-Za-z0-9]+ failed|EngineDeadError|out of memory|OOM|Connection refused|500 Internal Server Error' "$log" | tail -n 60 || true
    echo "--- tail 30: $log"
    tail -n 30 "$log" || true
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 5 | cut -d' ' -f2-)
else
  echo "log directory not found: $LOG_DIR"
fi