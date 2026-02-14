#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt -r requirements-dev.txt >/dev/null

OLLAMA_HOST="${AUTOMOM_OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_AUTOSTART="${AUTOMOM_OLLAMA_AUTOSTART:-1}"
OLLAMA_STARTED_BY_SCRIPT=0
OLLAMA_LOG_PATH="${ROOT_DIR}/data/ollama.log"

cleanup() {
  if [[ "${OLLAMA_STARTED_BY_SCRIPT}" == "1" ]] && [[ -n "${OLLAMA_PID:-}" ]]; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

require_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Error: 'ollama' is not installed."
    echo "Install it first: https://docs.ollama.com/linux"
    exit 1
  fi
}

ollama_api_ready() {
  curl --silent --show-error --max-time 2 "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1
}

start_ollama_if_needed() {
  require_ollama
  if ollama_api_ready; then
    return 0
  fi
  if [[ "${OLLAMA_AUTOSTART}" != "1" ]]; then
    echo "Error: Ollama API is not reachable at ${OLLAMA_HOST} and autostart is disabled."
    echo "Run 'ollama serve' manually or set AUTOMOM_OLLAMA_AUTOSTART=1."
    exit 1
  fi
  mkdir -p "${ROOT_DIR}/data"
  echo "Starting Ollama service..."
  ollama serve >"${OLLAMA_LOG_PATH}" 2>&1 &
  OLLAMA_PID=$!
  OLLAMA_STARTED_BY_SCRIPT=1

  for _ in $(seq 1 30); do
    if ollama_api_ready; then
      echo "Ollama ready at ${OLLAMA_HOST}"
      return 0
    fi
    sleep 1
  done

  echo "Error: Ollama failed to start. Check logs at ${OLLAMA_LOG_PATH}"
  exit 1
}

start_ollama_if_needed

exec uvicorn backend.app.main:app --host "${AUTOMOM_HOST:-127.0.0.1}" --port "${AUTOMOM_PORT:-8000}"
