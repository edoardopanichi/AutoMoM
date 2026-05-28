# AutoMoM

AutoMoM is a local-first FastAPI web application that turns meeting audio into English Minutes of Meeting (MoM) in Markdown. The pipeline can run fully local, or switch diarization, transcription, and MoM generation independently to OpenAI on a per-job basis.

## Installation

AutoMoM runs on Windows and Ubuntu. The app can use local models for every stage, or use OpenAI per job for diarization, transcription, and/or formatting.

### What You Need

- Python 3.10+ with `venv`. Python 3.11 or 3.12 is the safest choice if a newer Python has missing ML wheels.
- FFmpeg in `PATH`, or set `AUTOMOM_FFMPEG_BIN`. On Windows, use a shared FFmpeg 4-7 build so `torchcodec` can load FFmpeg DLLs.
- Git.
- A shell launcher:
  - Windows: PowerShell is enough; Git Bash is also supported.
  - Ubuntu: Bash.
- Local formatter path:
  - Default: Ollama plus the `qwen2.5:3b-instruct-q5_K_M` tag.
  - Alternative: a custom command backend via `AUTOMOM_FORMATTER_BACKEND=command`.
- Local transcription path:
  - Default catalog entry: `whisper.cpp` with a `whisper-cli` binary and a `.gguf` model.
  - Alternative catalog entry: `faster-whisper` with a CTranslate2 model directory.
- Local diarization path:
  - A local pyannote pipeline directory containing `config.yaml`.
  - The default expected path is `data/models/diarization/pyannote-speaker-diarization-community-1/config.yaml`.
  - The speaker-profile embedding default is `pyannote/wespeaker-voxceleb-resnet34-LM`.
- Optional CUDA:
  - NVIDIA driver and a CUDA-capable PyTorch/whisper.cpp/faster-whisper stack.
  - Use `AUTOMOM_COMPUTE_DEVICE=cpu` if CUDA wheels or GPU memory are not suitable.
- Optional OpenAI API key:
  - Only needed when a job uses `api` for diarization, transcription, or formatter.
  - The UI asks for the key per job; there is no required `.env` key for the OpenAI path.

### Windows Setup

Install prerequisites. With `winget`, the usual commands are:

```powershell
winget install Git.Git
winget install BtbN.FFmpeg.LGPL.Shared.7.1
winget install Ollama.Ollama
winget install Python.Python.3.12
```

Restart the terminal after installing PATH-based tools, then verify:

```powershell
git --version
ffmpeg -version
ollama --version
py -3.12 --version
# If the Python launcher is not installed:
python --version
```

The shared FFmpeg package matters on Windows. Generic static FFmpeg packages can provide `ffmpeg.exe` but still leave `torchcodec` unable to load because the FFmpeg DLLs are missing or too new.

Create and populate the virtual environment:

```powershell
py -3.12 -m venv .venv
# Or, when the Python launcher is not installed:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

Create local configuration:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least the paths for local diarization and local transcription if you are not using only OpenAI/remote stages:

```dotenv
AUTOMOM_COMPUTE_DEVICE="cpu"
AUTOMOM_DIARIZATION_PIPELINE="C:\absolute\path\to\pyannote-speaker-diarization-community-1\config.yaml"
AUTOMOM_TRANSCRIPTION_BIN="C:\absolute\path\to\whisper-cli.exe"
AUTOMOM_TRANSCRIPTION_MODEL="C:\absolute\path\to\model.gguf"
AUTOMOM_FORMATTER_OLLAMA_MODEL="qwen2.5:3b-instruct-q5_K_M"
```

If the Python ML stack and local binaries support CUDA, switch compute back to:

```dotenv
AUTOMOM_COMPUTE_DEVICE="auto"
```

Install the formatter model:

```powershell
ollama serve
ollama pull qwen2.5:3b-instruct-q5_K_M
```

Install transcription assets:

- Download or build `whisper.cpp`.
- Put the directory containing `whisper-cli.exe` in `PATH`, or set `AUTOMOM_TRANSCRIPTION_BIN`.
- Download a whisper.cpp-compatible `.gguf` model and set `AUTOMOM_TRANSCRIPTION_MODEL`.

Install diarization assets:

- Download or otherwise prepare a pyannote pipeline locally.
- Set `AUTOMOM_DIARIZATION_PIPELINE` or `AUTOMOM_DIARIZATION_MODEL` to that pipeline's `config.yaml`.
- If the selected pyannote model or embedding model is gated, authenticate/download it with Hugging Face and set `HF_TOKEN` or `HUGGINGFACE_TOKEN` as needed.

Start AutoMoM:

```powershell
.\.venv\Scripts\python.exe run_automom.py
```

The launcher delegates to `scripts/run_automom.ps1` when Git Bash is not available. It installs Python requirements, starts Ollama when `AUTOMOM_FORMATTER_BACKEND=ollama`, and serves the UI at `http://127.0.0.1:8000`.

### Ubuntu Setup

Install system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg curl git build-essential
```

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b-instruct-q5_K_M
```

Create and populate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Create local configuration:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```dotenv
AUTOMOM_COMPUTE_DEVICE="cpu"
AUTOMOM_DIARIZATION_PIPELINE="/absolute/path/to/pyannote-speaker-diarization-community-1/config.yaml"
AUTOMOM_TRANSCRIPTION_BIN="/absolute/path/to/whisper-cli"
AUTOMOM_TRANSCRIPTION_MODEL="/absolute/path/to/model.gguf"
AUTOMOM_FORMATTER_OLLAMA_MODEL="qwen2.5:3b-instruct-q5_K_M"
```

For CUDA, install a compatible NVIDIA driver and CUDA-capable Python/binary stack, verify `nvidia-smi`, then use:

```dotenv
AUTOMOM_COMPUTE_DEVICE="auto"
```

Start AutoMoM:

```bash
python run_automom.py
```

### Verification

After startup, open:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/api/health`
- `http://127.0.0.1:8000/api/system/startup-check`

In the UI, open Settings and verify that each selected local model is installed. Register local or remote model entries if your paths differ from `.env`.

Run tests:

```bash
pytest backend/tests -q --basetemp .pytest-tmp
```

For the longer cross-platform guide and troubleshooting notes, see `INSTALL.md`.

Quick start after setup:

```bash
python run_automom.py
```

## Verified Feature Set

The items below were verified against the current source tree and test suite.

- Nine-stage job pipeline:
  1. Validate/Normalize
  2. VAD
  3. Diarization
  4. Snippet extraction
  5. Speaker naming
  6. Transcription
  7. Transcript assembly
  8. MoM formatting
  9. Export
- Local-first browser UI with:
  - audio upload
  - template selection and inline template creation
  - persisted New Job defaults for template, routing, and model selectors
  - per-stage local/OpenAI execution toggles
  - OpenAI API key field shown only when needed
  - progress KPIs, progress bars, logs, and SSE updates
  - speaker naming with snippet playback
  - model manager, template inventory, and voice profile inventory
- Per-stage local/OpenAI routing:
  - diarization: local pyannote or OpenAI diarized transcription
  - transcription: local whisper.cpp runtime or OpenAI transcription
  - formatter: local Ollama model, legacy command backend, or OpenAI Responses API
- Local diarization backends:
  - `auto` which resolves to `pyannote`
  - `pyannote`
  - `embedding`
  - `heuristic`
- Long-recording local diarization support:
  - recordings longer than 20 minutes are chunked
  - chunk boundaries snap toward silence when possible
  - chunk-local speaker ids are stitched into global speaker ids via embedding similarity
  - pyannote GPU runs fall back to CPU on CUDA OOM
- Speaker naming and voice profiles:
  - snippet extraction chooses representative clips per speaker
  - existing local profiles are matched by cosine similarity on embeddings
  - ambiguous matches are surfaced separately from confident matches
  - confirmed names can be saved back as shared voice profiles
  - saved profiles can be refreshed for the currently selected local diarization model
- Transcription runtime behavior:
  - whisper.cpp binary probing and preferred binary selection
  - multiple local transcription runtimes via persisted model catalog
  - per-job local transcription model override
  - optional GPU enablement based on available backend support
  - one-shot GPU fallback to CPU if runtime invocation fails
  - transcription chunk planning merges adjacent same-speaker spans within configured limits
  - optional cap on chunk count
  - optional preservation of intermediate transcription WAVs
  - runtime summary written per job
- Formatter behavior:
  - prompt assembly from template definitions
  - multiple local formatter runtimes via persisted model catalog
  - per-job local formatter override
  - template heading/order validation
  - retry loop with corrective feedback when structured output is invalid
  - long-input rolling chunk summaries when strict templates exceed the token budget
  - long-input summaries separate formal decisions, adopted actions, public requests, open items, risks, and tentative states
  - one-shot final MoM rendering from the rolling summaries so all template sections share the same context
  - raw stdout/stderr/prompt/validation artifacts persisted per job
- Model management:
  - local model presence checks before job start
  - register existing local stage models and make them selectable immediately
  - persisted local model catalog
  - resumable file downloads for non-formatter models when URLs are configured
  - checksum verification when SHA256 is configured
  - Ollama tag selection for formatter models
  - formatter model pull via Ollama `/api/pull`
- API features:
  - health and startup readiness endpoints
  - New Job default get/save endpoints
  - job creation/listing/status/SSE/cancel
  - artifact and snippet download endpoints
  - template CRUD endpoints
  - profile CRUD plus embedding refresh task endpoints
  - model status/download plus local model catalog endpoints
- Utility scripts:
  - `run_automom.py`
  - `scripts/run_automom.sh`
  - `scripts/run_automom.ps1`
  - `scripts/prepare_mock_models.sh`
  - `scripts/prepare_mock_models.ps1`
  - `scripts/check_faster_whisper_env.py`
- Experiment and comparison scripts:
  - `scripts/experiments/benchmark_local_transcription.py`
  - `scripts/experiments/benchmark_transcription_runtime_compare.py`
  - `scripts/experiments/run_long_audio_test.py`

## Repository Layout

- `backend/app`
  - FastAPI entrypoint, configuration, schemas, job store, static frontend
- `backend/pipeline`
  - audio normalization, VAD, diarization, snippets, transcription, formatter, orchestration, subprocess helpers
- `backend/models`
  - model registry and download manager
- `backend/profiles`
  - saved voice profiles, embedding persistence, matching, refresh tasks
- `backend/tests`
  - unit and integration coverage
- `scripts`
  - day-to-day developer helpers
- `scripts/experiments`
  - experiments, benchmarks, and comparison scripts
- `data`
  - runtime jobs, uploads, templates, models, profiles

## Runtime Artifacts

Each job writes to `data/jobs/<job_id>/`. Depending on execution path, the job may contain:

- `job_state.json`
- `job_summary.json`
- `audio_normalized.wav`
- `audio_metadata.json`
- `vad_regions.json`
- `diarization.json`
- `diarization_chunks.json`
- `diarization_stitching.json`
- `snippets.json`
- `snippets/*.wav`
- `speaker_mapping.json`
- `segments_transcript.json`
- `transcript.json`
- `transcription_runtime.json`
- `mom.md`
- `mom_structured.json`
- `formatter_system_prompt.txt`
- `formatter_user_prompt.txt`
- `formatter_stdout.txt`
- `formatter_stderr.txt`
- `formatter_raw_output.txt`
- `formatter_validation.json`
- `formatter_reduced_notes.json` for long-input rolling chunk summaries
- `export/mom.md`

Artifact keys are exposed through the job API and are part of the runtime contract between the backend and frontend.

## Key Configuration

- Compute:
  - `AUTOMOM_COMPUTE_DEVICE=auto|cpu|cuda`
  - `AUTOMOM_CUDA_DEVICE_ID=<index>`
  - `AUTOMOM_DISABLE_CUDA=1` to force CPU
- Diarization:
  - `AUTOMOM_DIARIZATION_BACKEND=auto|pyannote|embedding|heuristic`
  - `AUTOMOM_DIARIZATION_MODEL`
  - `AUTOMOM_DIARIZATION_PIPELINE`
  - `AUTOMOM_DIARIZATION_EMBEDDING_MODEL`
  - `AUTOMOM_DIARIZATION_MIN_SPEAKERS`
  - `AUTOMOM_DIARIZATION_MAX_SPEAKERS`
  - `AUTOMOM_DIARIZATION_SUBPROCESS`
- Transcription:
  - `AUTOMOM_TRANSCRIPTION_BIN`
  - `AUTOMOM_TRANSCRIPTION_MODEL`
  - `AUTOMOM_TRANSCRIPTION_URL`
  - `AUTOMOM_TRANSCRIPTION_SHA256`
  - `AUTOMOM_TRANSCRIPTION_THREADS`
  - `AUTOMOM_TRANSCRIPTION_PROCESSORS`
  - `AUTOMOM_TRANSCRIPTION_GPU_LAYERS`
  - `AUTOMOM_TRANSCRIPTION_MERGE_GAP_S`
  - `AUTOMOM_TRANSCRIPTION_MAX_CHUNK_S`
  - `AUTOMOM_TRANSCRIPTION_MAX_SEGMENTS`
  - `AUTOMOM_TRANSCRIPTION_KEEP_SEGMENT_AUDIO`
- Formatter:
  - `AUTOMOM_FORMATTER_BACKEND=ollama|command`
  - `AUTOMOM_FORMATTER_COMMAND`
  - `AUTOMOM_OLLAMA_HOST`
  - `AUTOMOM_FORMATTER_OLLAMA_MODEL`
  - `AUTOMOM_FORMATTER_OLLAMA_THINK=false|true|low|medium|high|omit` (default: `false`)
  - `AUTOMOM_FORMATTER_OLLAMA_NUM_CTX` (default: `8192`)
  - `AUTOMOM_FORMATTER_OLLAMA_NUM_PREDICT` (default: `1600`)
  - `AUTOMOM_FORMATTER_OLLAMA_TEMPERATURE` (default: `0.1`)
  - `AUTOMOM_FORMATTER_TIMEOUT_S`
- Audio normalization:
  - `AUTOMOM_AUDIO_DENOISE=1|0`
  - `AUTOMOM_AUDIO_DENOISE_FILTER=<ffmpeg filter>`
- Server:
  - `AUTOMOM_HOST`
  - `AUTOMOM_PORT`
  - `AUTOMOM_MAX_WORKERS`
  - `AUTOMOM_CORS_ORIGINS`

## API Surface

- `GET /`
  - frontend entrypoint
- `GET /api/health`
- `GET /api/system/startup-check`
- `GET /api/job-defaults`
- `POST /api/job-defaults`
- `GET /api/models`
- `GET /api/models/local`
- `GET /api/models/local/{stage}`
- `POST /api/models/local`
- `DELETE /api/models/local/{model_id}`
- `GET /api/models/diarization`
- `POST /api/models/consent`
- `POST /api/models/download`
- `GET /api/models/downloads`
- `GET /api/models/downloads/{model_id}`
- `GET /api/templates`
- `GET /api/templates/{template_id}`
- `POST /api/templates`
- `DELETE /api/templates/{template_id}`
- `GET /api/profiles`
- `POST /api/profiles`
- `DELETE /api/profiles/{profile_id}`
- `POST /api/profiles/rebuild`
- `GET /api/profiles/rebuild/{task_id}`
- `POST /api/jobs`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/events`
- `POST /api/jobs/{job_id}/cancel`
- `POST /api/jobs/{job_id}/speaker-mapping`
- `GET /api/jobs/{job_id}/artifacts/{artifact_name}`
- `GET /api/jobs/{job_id}/snippets/{snippet_name}`
- `GET /api/jobs/{job_id}/mom`
- `GET /api/jobs/{job_id}/download/mom`

## OpenAI Path Notes

- OpenAI audio uploads are limited to 25 MB per request in the current implementation.
- The backend prefers the original uploaded file for OpenAI audio when it is in a supported format and under the limit; otherwise it falls back to the normalized WAV if that file also fits.
- If OpenAI diarization already returned text-bearing segments, the pipeline reuses that transcript instead of uploading per-speaker chunks for a second cloud transcription pass.

## Formatter Pipeline

The formatter has two paths:

- Short or medium transcripts use the selected template directly. AutoMoM builds one prompt from the template, title, speakers, and transcript, then validates the returned Markdown against the template heading order.
- Long strict-template transcripts use a rolling chunk-summary pipeline before final rendering. The formatter splits the transcript into roughly 20-minute chunks, also respecting an approximate token cap. Each chunk is summarized by the selected formatter model with the previous accumulated summary as context.

The long-input chunk summaries are intentionally structured around meeting semantics rather than a fixed final template:

- formal decisions and outcomes
- adopted actions, conditions, and TODOs
- speaker requests or concerns that were not adopted
- open or pending items
- risks and concerns
- superseded or tentative states

This keeps the general MoM problem explicit: discussion, public objections, and tentative proposals should not become adopted actions unless the transcript later shows a motion, vote, chair ruling, staff requirement, applicant commitment, or explicit procedural requirement. After all chunks are summarized, the formatter performs one final model call that fills the selected template in a single pass from the combined summaries. The existing template validation and corrective retry loop still applies to the final Markdown.

For long-input jobs, `formatter_reduced_notes.json` stores the rolling chunk summaries and `formatter_validation.json` records `long_input_strategy=rolling_chunk_summary` plus the chunk count.

## Development

### One-command start

```bash
python run_automom.py
```

The helper checks Ollama, starts `ollama serve` automatically when needed, and then launches the app.

### Mock model placeholders

For local development only:

```bash
python run_automom.py --prepare-mock-models
```

### Tests

Run:

```bash
source .venv/bin/activate
pytest backend/tests -q --basetemp .pytest-tmp
```

Verified on May 28, 2026:

- `148` tests passed

## Dependency Notes

- `requirements.txt` now reflects the full current runtime surface, including the local diarization and voice-profile stack:
  - `torch`
  - `pyannote.audio`
- `jinja2` was removed because it is not used by the current codebase.
- Installing `torch` may still require platform-specific wheels depending on your CPU/CUDA setup.

## Documentation Standard

- Python functions and methods must carry Doxygen-style docstrings.
- Frontend JavaScript functions must carry JSDoc/Doxygen-style comments.
- Complex, fragile, or high-impact logic must also include inline comments that explain why the code is written that way.
- When behavior, artifacts, endpoints, or configuration change, update both `README.md` and `AGENT.md` in the same change.
