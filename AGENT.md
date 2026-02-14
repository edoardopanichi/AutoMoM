# AGENT.md

## Purpose
AutoMoM is a local pipeline app that turns meeting audio into English MoM Markdown with diarized transcript support.

## Main Runtime Flow
- FastAPI entrypoint: `backend/app/main.py`
- Job orchestration: `backend/pipeline/orchestrator.py`
- Job persistence/state: `backend/app/job_store.py`
- Stage implementation modules:
  - `backend/pipeline/audio.py`
  - `backend/pipeline/vad.py`
  - `backend/pipeline/diarization.py`
  - `backend/pipeline/snippets.py`
  - `backend/pipeline/transcription.py`
  - `backend/pipeline/formatter.py`

## Data and Artifacts
- Local storage root: `data/`
- Key folders:
  - `data/jobs/`
  - `data/models/`
  - `data/templates/`
  - `data/profiles/`
  - `data/uploads/`
- Per-job artifacts are written into `data/jobs/<job_id>/`

## Models
- Model checks and consent/download flow: `backend/models/manager.py`
- Required model specs are defined in `backend/app/config.py`
- Downloads are permission-gated and resume-capable

## Templates
- Template manager: `backend/pipeline/template_manager.py`
- Default template auto-created in `data/templates/default.{json,md.j2}`
- Formatter prompt assembly happens in template manager; final structuring/rendering in `backend/pipeline/formatter.py`

## Voice Profiles
- Manager: `backend/profiles/manager.py`
- Profiles are stored as JSON files in `data/profiles/`
- Matching uses cosine similarity over locally computed embeddings

## Frontend
- Served by FastAPI static mount from `backend/app/static`
- Files:
  - `backend/app/static/index.html`
  - `backend/app/static/css/styles.css`
  - `backend/app/static/js/app.js`

## Testing
- Test root: `backend/tests/`
- Unit tests: `backend/tests/unit/`
- Integration tests + golden files: `backend/tests/integration/`
- Run with `pytest backend/tests -q`

## One-command Start
- `./scripts/run_automom.sh`

## Development Shortcut
- Create mock model placeholders for local testing:
  - `./scripts/prepare_mock_models.sh`

## Code Commenting Policy
- Write very clear, human-oriented comments in code so non-authors can read and maintain it quickly.
- Prefer one extra explanatory comment over one missing comment when logic is non-trivial.
