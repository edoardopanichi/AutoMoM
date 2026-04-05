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
- Model manager with local model checks and web-triggered downloads
- Optional OpenAI API execution per stage with one API key across diarization, transcription, and MoM generation
- Multiple templates with default template included
- Stage progress reporting (overall, stage, segment-level)
- Markdown export

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

`run_automom.sh` now checks Ollama, starts `ollama serve` automatically when needed, and then launches the app.  
Set `AUTOMOM_OLLAMA_AUTOSTART=0` to disable automatic Ollama startup.

## First Run and Model Checks
At startup, required models are listed in Settings:
- Diarization model
- Voxtral weights
- Formatter LLM

Before a job starts, missing required models must be handled.
For formatter, the web UI downloads via Ollama (`/api/pull`) using the selected model tag.

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
- Unit tests for audio normalization, VAD, diarization, snippets, voice profiles, transcript merging, voxtral wrapper, template prompting, and model download flows
- Integration test for end-to-end pipeline run with golden transcript and Markdown structure checks

## Notes
- Real diarization/ASR/formatter model inference is mandatory for job execution. If required models/runtimes are missing, the job fails with an actionable configuration error.
- Compute selection:
  - `AUTOMOM_COMPUTE_DEVICE=auto|cpu|cuda` (default: `auto`)
  - `AUTOMOM_CUDA_DEVICE_ID` selects GPU index when CUDA is used
  - `AUTOMOM_DISABLE_CUDA=1` forces CPU even if CUDA is available
  - The pipeline auto-falls back to CPU if CUDA is unavailable or a model runtime rejects GPU flags
- Diarization backends:
  - `AUTOMOM_DIARIZATION_BACKEND=heuristic`: built-in heuristic clustering (explicit selection only)
  - `AUTOMOM_DIARIZATION_BACKEND=embedding`: uses speaker embeddings (`AUTOMOM_DIARIZATION_EMBEDDING_MODEL`, default `pyannote/wespeaker-voxceleb-resnet34-LM`) and clustering
  - `AUTOMOM_DIARIZATION_BACKEND=pyannote`: full pyannote pipeline mode (recommended for model-based diarization)
  - `AUTOMOM_DIARIZATION_BACKEND=auto`: strict alias of `pyannote` (no backend fallback)
  - `AUTOMOM_DIARIZATION_MIN_SPEAKERS` / `AUTOMOM_DIARIZATION_MAX_SPEAKERS`: optional speaker count bounds (`0` = auto/unbounded)
  - For full pyannote pipeline mode, set `AUTOMOM_DIARIZATION_PIPELINE` (or `AUTOMOM_DIARIZATION_MODEL`) to a local pipeline config path. Most official pyannote pipelines are gated and require `HF_TOKEN`.
  - In pyannote and embedding modes, device is selected dynamically from `AUTOMOM_COMPUTE_DEVICE`.
- Runtime acceleration flags:
  - `AUTOMOM_VOXTRAL_GPU_LAYERS` for whisper.cpp-based ASR binaries
  - `AUTOMOM_VOXTRAL_THREADS` / `AUTOMOM_VOXTRAL_PROCESSORS` tune whisper.cpp CPU execution
  - `AUTOMOM_TRANSCRIPTION_MERGE_GAP_S` / `AUTOMOM_TRANSCRIPTION_MAX_CHUNK_S` control same-speaker chunking before ASR
  - `AUTOMOM_TRANSCRIPTION_KEEP_SEGMENT_AUDIO=1` preserves extracted transcription chunk WAVs for debugging
  - Formatter default backend is Ollama (`AUTOMOM_FORMATTER_BACKEND=ollama`, `AUTOMOM_OLLAMA_HOST`, `AUTOMOM_FORMATTER_OLLAMA_MODEL`)
  - Legacy formatter command backend is optional (`AUTOMOM_FORMATTER_BACKEND=command`, `AUTOMOM_FORMATTER_COMMAND`)
- Cloud execution:
  - On the New Job page you can optionally provide an OpenAI API key and choose `local` or `api` independently for diarization, transcription, and MoM formatting
  - Recommended OpenAI models: `gpt-4o-transcribe-diarize`, `gpt-4o-transcribe`, and `gpt-5-mini`
- Audio denoise controls (applied during normalization):
  - `AUTOMOM_AUDIO_DENOISE=1|0` (default: `1`)
  - `AUTOMOM_AUDIO_DENOISE_FILTER` (default: `afftdn`; FFmpeg audio filter expression)
- Runtime artifacts are stored under `data/jobs/<job_id>/`.
  - Local ASR jobs now include `transcription_runtime.json` with requested vs active compute mode and binary capability details.
  - Job IDs are generated as `YYYY-MM-DD-HH:MM-meeting_title` (fallback: `...-meeting`).
