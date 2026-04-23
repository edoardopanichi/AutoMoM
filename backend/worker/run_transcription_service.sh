#!/usr/bin/env bash
set -euo pipefail

: "${AUTOMOM_REMOTE_WORKER_HOST:=0.0.0.0}"
: "${AUTOMOM_REMOTE_WORKER_PORT:=8011}"
: "${AUTOMOM_REMOTE_WORKER_ENABLED_STAGES:=transcription}"
: "${AUTOMOM_REMOTE_TRANSCRIPTION_BIN:=/opt/whisper.cpp/build/bin/whisper-cli}"
: "${AUTOMOM_REMOTE_TRANSCRIPTION_MODEL:=/models/whisper/model.gguf}"
: "${AUTOMOM_REMOTE_TRANSCRIPTION_MODEL_NAME:=large-v3}"

exec uvicorn backend.worker.main:app --host "${AUTOMOM_REMOTE_WORKER_HOST}" --port "${AUTOMOM_REMOTE_WORKER_PORT}"
