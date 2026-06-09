#!/usr/bin/env bash
# Gyul 데모 일괄 실행: API(8000) + React 프론트(5173)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
API_HOST="${API_HOST:-127.0.0.1}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"

API_PID=""
WEB_PID=""

cleanup() {
  echo ""
  echo "종료 중..."
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "$WEB_PID" ]] && kill "$WEB_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "종료 완료."
}

trap cleanup EXIT INT TERM

setup_python() {
  if command -v pyenv >/dev/null 2>&1; then
    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command -v pyenv >/dev/null 2>&1; then
      eval "$(pyenv init - bash 2>/dev/null || pyenv init - 2>/dev/null || true)"
    fi
    if pyenv versions --bare 2>/dev/null | grep -qx "3.11.6"; then
      pyenv shell 3.11.6
    fi
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3가 필요합니다."
    exit 1
  fi

  if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo "Python 패키지 설치 중..."
    python3 -m pip install -r requirements.txt
  fi
}

check_model() {
  local model_path="$ROOT/model/model.safetensors"
  if [[ -e "$model_path" ]]; then
    return 0
  fi
  echo "[경고] model/model.safetensors 가 없습니다."
  echo "       예: ln -s ~/Desktop/kyul/model/model.safetensors model/model.safetensors"
}

setup_frontend() {
  if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
    echo "프론트 npm install 중..."
    (cd "$ROOT/frontend" && npm install)
  fi
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local i
  for i in $(seq 1 40); do
    if curl -s -o /dev/null "$url" 2>/dev/null; then
      echo "  ✓ $name 준비됨"
      return 0
    fi
    sleep 0.5
  done
  echo "  ✗ $name 응답 없음 ($url)"
  return 1
}

setup_python
check_model
setup_frontend

echo "============================================"
echo " Gyul 데모 서버 시작"
echo " API  : http://${API_HOST}:${API_PORT}"
echo " Demo : http://${WEB_HOST}:${WEB_PORT}/demo"
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo " LLM  : gpt-4o-mini (OPENAI_API_KEY)"
else
  echo " LLM  : Kanana (OPENAI_API_KEY 없음)"
fi
echo "============================================"
echo ""

python3 -m uvicorn ser_api:app --reload --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

(
  cd "$ROOT/frontend"
  npm run dev -- --host "$WEB_HOST" --port "$WEB_PORT"
) &
WEB_PID=$!

echo "서버 기동 대기 중..."
wait_for_url "http://${API_HOST}:${API_PORT}/health" "API" || true
wait_for_url "http://${WEB_HOST}:${WEB_PORT}/" "프론트" || true

echo ""
echo "브라우저에서 열기: http://${WEB_HOST}:${WEB_PORT}/demo"
echo "종료: Ctrl+C"
echo ""

wait
