#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/models/diarization/pyannote-speaker-diarization-community-1 data/models/voxtral data/models/formatter

if [[ ! -f data/models/diarization/pyannote-speaker-diarization-community-1/config.yaml ]]; then
  cat > data/models/diarization/pyannote-speaker-diarization-community-1/config.yaml <<'YAML'
version: 3.1
pipeline:
  name: mock
YAML
fi
if [[ ! -f data/models/voxtral/model.gguf ]]; then
  printf 'mock-voxtral' > data/models/voxtral/model.gguf
fi
if [[ ! -f data/models/formatter/selected_model.txt ]]; then
  printf 'llama3.1:8b-instruct-q4_K_M' > data/models/formatter/selected_model.txt
fi

mkdir -p data/models
cat > data/models/consent.json <<JSON
{
  "diarization": true,
  "voxtral": true,
  "formatter": true
}
JSON

echo "Mock models prepared in data/models"
