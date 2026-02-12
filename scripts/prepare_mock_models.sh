#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/models/diarization data/models/voxtral data/models/formatter

if [[ ! -f data/models/diarization/model.bin ]]; then
  printf 'mock-diarization' > data/models/diarization/model.bin
fi
if [[ ! -f data/models/voxtral/model.gguf ]]; then
  printf 'mock-voxtral' > data/models/voxtral/model.gguf
fi
if [[ ! -f data/models/formatter/model.gguf ]]; then
  printf 'mock-formatter' > data/models/formatter/model.gguf
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
