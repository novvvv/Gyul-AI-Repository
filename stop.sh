#!/usr/bin/env bash
# Gyul 데모 서버 프로세스 종료
set -euo pipefail

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "포트 $port 종료: $pids"
    kill $pids 2>/dev/null || true
  else
    echo "포트 $port 에 실행 중인 프로세스 없음"
  fi
}

kill_port "$API_PORT"
kill_port "$WEB_PORT"
echo "완료."
