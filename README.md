# AutoMoM

AutoMoM is a local/offline web tool that converts meeting audio into English Minutes of Meeting (MoM) in Markdown.

## Features
- Local job pipeline with 9 stages:
  1. Validate/Normalize
  2. VAD
  3. Diarization
  4. Snippet extraction
  5. Speaker naming (blocking user step)
  6. Transcription
  7. Transcript assembly
  8. MoM formatting
  9. Export
- Speaker count detection and speaker snippet playback for naming
- Local voice profiles for speaker auto-identification
- Model manager with per-model consent and permission-gated downloads
- Multiple templates with default template included
- Stage progress reporting (overall, stage, segment-level)
- Markdown export with collapsible transcript section

## Repository Layout
- `backend/app`: FastAPI app, API routes, state schemas, job store
- `backend/pipeline`: pipeline stages and orchestrator
- `backend/models`: model manager (presence, consent, download)
- `backend/profiles`: voice profile embedding/matching manager
- `backend/tests`: unit + integration tests
- `scripts`: run/start helper scripts
- `data`: local runtime storage (jobs/models/templates/profiles/uploads)

## Requirements
- Python 3.11+
- `ffmpeg` installed and available in PATH

## Quick Start (One Command)
```bash
./scripts/run_automom.sh
```

Then open:
- `http://127.0.0.1:8000`

## First Run and Model Checks
At startup, required models are listed in Settings:
- Diarization model
- Voxtral weights
- Formatter LLM

Before a job starts, missing model consents/downloads must be handled.

### Dev shortcut (mock model placeholders)
For local development only:
```bash
./scripts/prepare_mock_models.sh
```

## Tests
Run:
```bash
source .venv/bin/activate
pytest backend/tests -q
```

Covers:
- Unit tests for audio normalization, VAD, diarization, snippets, voice profiles, transcript merging, voxtral wrapper, template rendering/prompting, and consent-gated model downloads
- Integration test for end-to-end pipeline run with golden transcript and Markdown structure checks

## Notes
- Real diarization/ASR/formatter model inference is pluggable. If local model binaries are not configured, the app falls back to deterministic offline-safe behavior for development/testing.
- Diarization backends:
  - `AUTOMOM_DIARIZATION_BACKEND=heuristic`: built-in heuristic clustering (default fallback)
  - `AUTOMOM_DIARIZATION_BACKEND=embedding`: uses speaker embeddings (`AUTOMOM_DIARIZATION_EMBEDDING_MODEL`, default `pyannote/wespeaker-voxceleb-resnet34-LM`) and clustering
  - `AUTOMOM_DIARIZATION_BACKEND=pyannote`: forces full pyannote pipeline mode; falls back to heuristic if pipeline is unavailable
  - `AUTOMOM_DIARIZATION_BACKEND=auto`: tries full pyannote pipeline, then embedding backend, then heuristic fallback
  - For full pyannote pipeline mode, set `AUTOMOM_DIARIZATION_PIPELINE` (or `AUTOMOM_DIARIZATION_MODEL`) to a local pipeline config path. Most official pyannote pipelines are gated and require `HF_TOKEN`.
- Runtime artifacts are stored under `data/jobs/<job_id>/`.
