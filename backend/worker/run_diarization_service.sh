#!/usr/bin/env bash
set -euo pipefail

: "${AUTOMOM_REMOTE_WORKER_HOST:=0.0.0.0}"
: "${AUTOMOM_REMOTE_WORKER_PORT:=8010}"
: "${AUTOMOM_REMOTE_WORKER_ENABLED_STAGES:=diarization}"
: "${AUTOMOM_REMOTE_DIARIZATION_PIPELINE:=/models/pyannote/config.yaml}"
: "${AUTOMOM_REMOTE_DIARIZATION_MODEL_NAME:=pyannote-community-1}"
: "${AUTOMOM_REMOTE_PROFILE_MODEL_REF:=pyannote-community-1}"
: "${AUTOMOM_REMOTE_DIARIZATION_EMBEDDING_MODEL:=pyannote/wespeaker-voxceleb-resnet34-LM}"

exec uvicorn backend.worker.main:app --host "${AUTOMOM_REMOTE_WORKER_HOST}" --port "${AUTOMOM_REMOTE_WORKER_PORT}"
