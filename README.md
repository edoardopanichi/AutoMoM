# AutoMoM

AutoMoM is a local-first FastAPI web application that turns meeting audio into English Minutes of Meeting (MoM) in Markdown. The pipeline can run fully local, or switch diarization, transcription, and MoM generation independently to OpenAI on a per-job basis.

## Installation

- Linux step-by-step install guide (CPU and CUDA): `INSTALL.md`
- Quick start after setup:

```bash
./scripts/run_automom.sh
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
  - `scripts/run_automom.sh`
  - `scripts/prepare_mock_models.sh`
  - `scripts/benchmark_local_transcription.py`
  - `scripts/run_long_audio_test.py`
  - `scripts/check_faster_whisper_env.py`

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
  - local validation and benchmark helpers
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
./scripts/run_automom.sh
```

The helper checks Ollama, starts `ollama serve` automatically when needed, and then launches the app.

### Mock model placeholders

For local development only:

```bash
./scripts/prepare_mock_models.sh
```

### Tests

Run:

```bash
source .venv/bin/activate
pytest backend/tests -q
```

Verified on April 9, 2026:

- `80` tests passed

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
