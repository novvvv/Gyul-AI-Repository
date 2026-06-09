#!/usr/bin/env bash
# Gyul 데모 일괄 실행: API(8000) + React 프론트(5173)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
API_HOST="${API_HOST:-127.0.0.1}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
PYENV_PYTHON_VERSION="${PYENV_PYTHON_VERSION:-3.11.6}"

API_PID=""
WEB_PID=""
PYTHON=""

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
    export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
    if pyenv versions --bare 2>/dev/null | grep -qx "$PYENV_PYTHON_VERSION"; then
      export PYENV_VERSION="$PYENV_PYTHON_VERSION"
      PYTHON="$PYENV_ROOT/versions/$PYENV_PYTHON_VERSION/bin/python3"
    fi
  fi

  if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || true)"
  fi

  if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo "python3가 필요합니다. pyenv로 Python ${PYENV_PYTHON_VERSION} 설치를 권장합니다."
    echo "  pyenv install ${PYENV_PYTHON_VERSION}"
    exit 1
  fi

  local py_major py_minor
  read -r py_major py_minor < <("$PYTHON" -c 'import sys; print(sys.version_info.major, sys.version_info.minor)')
  echo "Python: $("$PYTHON" --version) ($PYTHON)"

  if (( py_major < 3 || (py_major == 3 && py_minor < 10) )); then
    echo ""
    echo "[오류] Python 3.10 이상이 필요합니다 (현재 ${py_major}.${py_minor})."
    echo "       conda(base)나 시스템 python3 대신 pyenv 3.11.6을 사용하세요:"
    echo "         pyenv install ${PYENV_PYTHON_VERSION}"
    echo "         PYENV_VERSION=${PYENV_PYTHON_VERSION} ./start.sh"
    exit 1
  fi

  if ! "$PYTHON" -c "import uvicorn, transformers; assert int(transformers.__version__.split('.')[0]) >= 5" 2>/dev/null; then
    echo "Python 패키지 설치 중..."
    "$PYTHON" -m pip install -U pip
    "$PYTHON" -m pip install -r requirements.txt
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
if [[ -n "${FISH_AUDIO_API_KEY:-}" ]]; then
  echo " TTS  : Fish Audio"
else
  echo " TTS  : off (FISH_AUDIO_API_KEY 없음)"
fi
echo "============================================"
echo ""

"$PYTHON" -m uvicorn ser_api:app --reload --host "$API_HOST" --port "$API_PORT" &
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
