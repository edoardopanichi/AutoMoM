#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/models/diarization data/models/voxtral data/models/formatter
printf 'mock-diarization' > data/models/diarization/model.bin
printf 'mock-voxtral' > data/models/voxtral/model.gguf
printf 'mock-formatter' > data/models/formatter/model.gguf

mkdir -p data/models
cat > data/models/consent.json <<JSON
{
  "diarization": true,
  "voxtral": true,
  "formatter": true
}
JSON

echo "Mock models prepared in data/models"
