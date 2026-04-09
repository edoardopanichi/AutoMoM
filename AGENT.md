# AGENT.md

## Purpose

AutoMoM is a local-first meeting-audio pipeline that produces English Minutes of Meeting (MoM) in Markdown. It is a FastAPI backend with a static frontend, a multi-stage orchestration pipeline, optional OpenAI execution for selected stages, and persistent runtime artifacts under `data/`.

## Runtime Architecture

- FastAPI entrypoint: `backend/app/main.py`
- Configuration and required model specs: `backend/app/config.py`
- Runtime schema contract: `backend/app/schemas.py`
- In-memory job state plus on-disk persistence: `backend/app/job_store.py`
- Pipeline orchestration: `backend/pipeline/orchestrator.py`
- Stage modules:
  - `backend/pipeline/audio.py`
  - `backend/pipeline/vad.py`
  - `backend/pipeline/diarization.py`
  - `backend/pipeline/snippets.py`
  - `backend/pipeline/transcription.py`
  - `backend/pipeline/formatter.py`
- Support modules:
  - `backend/pipeline/compute.py`
  - `backend/pipeline/openai_client.py`
  - `backend/pipeline/subprocess_utils.py`
  - `backend/pipeline/template_manager.py`
  - `backend/pipeline/diarization_worker.py`
- Model/runtime management:
  - `backend/models/manager.py`
  - `backend/models/diarization_registry.py`
- Voice profile management:
  - `backend/profiles/manager.py`

## Verified Job Flow

The orchestrator currently runs these nine stages:

1. Validate/Normalize
2. VAD
3. Diarization
4. Snippet extraction
5. Speaker naming
6. Transcription
7. Transcript assembly
8. MoM formatting
9. Export

Important execution behavior:

- The pipeline is local-first.
- Diarization, transcription, and formatter execution can each be `local` or `api` on a per-job basis.
- Local diarization currently exposes one registry entry in the API: `pyannote-community-1`.
- `AUTOMOM_DIARIZATION_BACKEND=auto` resolves to the pyannote path, not to a best-effort fallback chain.
- Long local pyannote diarization is chunked above 20 minutes, with overlap and speaker stitching by embeddings.
- OpenAI diarization can short-circuit later cloud transcription by reusing the diarized transcript segments directly.
- Local transcription uses whisper.cpp-style binaries and contains GPU probing plus CPU fallback logic.
- Formatter supports three effective modes:
  - OpenAI Responses API
  - Ollama `/api/generate`
  - legacy command backend

## Artifact Contract

Per-job artifacts are written under `data/jobs/<job_id>/`. The backend and frontend both rely on artifact keys remaining stable.

Common artifact keys:

- `audio_normalized`
- `vad_regions`
- `diarization`
- `diarization_chunks`
- `diarization_stitching`
- `snippets`
- `speaker_mapping`
- `segments_transcript`
- `transcript`
- `transcription_runtime`
- `mom_markdown`
- `mom_structured`
- `formatter_system_prompt`
- `formatter_user_prompt`
- `formatter_stdout`
- `formatter_stderr`
- `formatter_raw_output`
- `formatter_validation`
- `formatter_reduced_notes`
- `export_markdown`
- `job_summary`

If you rename, remove, or stop writing an artifact, update both the API behavior and the frontend assumptions.

## API Surface

Main endpoint groups in `backend/app/main.py`:

- Root and health
- System startup readiness
- Model status, download flow, and formatter model selection
- Local diarization model listing
- Template CRUD
- Voice profile CRUD and refresh tasks
- Job creation, list, detail, cancel, speaker mapping
- SSE job events
- Artifact, snippet, and MoM download/preview endpoints

Security-sensitive behavior to preserve:

- snippet download sanitizes file names against traversal
- OpenAI API key is required only when at least one stage uses API execution
- job creation validates the template and only requires local models for stages that actually run locally

## Frontend Contract

Static frontend lives in `backend/app/static`:

- `backend/app/static/index.html`
- `backend/app/static/css/styles.css`
- `backend/app/static/js/app.js`

Frontend expectations that matter when changing backend behavior:

- tabs are `new-job`, `progress`, `result`, `settings`
- SSE payloads come from `/api/jobs/{job_id}/events`
- speaker snippet playback uses `/api/jobs/{job_id}/snippets/{snippet_name}`
- settings panels expect model/template/profile collections
- form submission depends on stage execution selectors, local model selectors, and optional OpenAI API key visibility

## Voice Profiles

- Profiles are stored as JSON manifests plus sample audio under `data/profiles/`
- Matching is local-only and keyed by diarization model id plus embedding model ref
- Refresh tasks only generate missing local embedding entries for saved samples
- OpenAI diarization does not currently create reusable local embedding vectors

## Model Management

- Non-formatter local models are represented by `ModelSpec`
- Download URLs and checksums are optional and come from environment variables
- Formatter installation state is checked through Ollama tags, not a local file
- Formatter model selection is persisted in `data/models/formatter/selected_model.txt`
- Full local diarization/profile functionality requires the Python runtime to include `torch` and `pyannote.audio`

## Tests And Scripts

- Test root: `backend/tests/`
- Unit and integration tests currently pass with `pytest backend/tests -q`
- Benchmark and validation scripts live in `scripts/`
- `scripts/benchmark_local_transcription.py` exercises both end-to-end jobs and isolated stage-6 benchmarking
- `scripts/run_long_audio_test.py` is a long-audio validation helper with automatic speaker submission

## Documentation And Comment Policy

This repository now follows these rules:

- Every Python function and method must have a Doxygen-style docstring.
- Every JavaScript function in the frontend must have a JSDoc/Doxygen-style comment.
- Complex, fragile, or high-impact logic must also have inline comments explaining why it works the way it does.
- Update `README.md` and `AGENT.md` whenever behavior, artifacts, public endpoints, scripts, or configuration materially change.
- Do not leave speculative documentation in place. If code and docs disagree, fix the docs in the same change.

## Change Guidance

When editing complex areas, be conservative:

- `backend/pipeline/orchestrator.py`
  - preserve stage names, progress semantics, and artifact writes unless intentionally changing the contract
- `backend/pipeline/diarization.py`
  - preserve chunk ownership, overlap, and stitching behavior unless tests and docs are updated together
- `backend/pipeline/transcription.py`
  - preserve runtime reporting, binary probing, and GPU fallback semantics
- `backend/pipeline/formatter.py`
  - preserve validation, retry, and reduced-notes behavior unless the template contract changes intentionally
- `backend/app/main.py`
  - preserve endpoint payload shapes unless the frontend and docs are updated in the same change
