# AGENT.md - AutoMoM

## Project Snapshot

AutoMoM is a local-first FastAPI app with a static browser UI that turns meeting audio into English Minutes of Meeting (MoM). Diarization, transcription, and formatting can run locally or through OpenAI independently per job.

Use this file for operational guidance. Use `README.md` only when you need the complete API/config/artifact inventory, and use `auto_MoM_specification.md` only when changing product behavior or checking original intent.
For Linux public setup steps (CPU and CUDA), use `INSTALL.md`.

## Common Commands

- Start app: `./scripts/run_automom.sh` or `make dev`
- Run all tests: `pytest backend/tests -q` or `make test`
- Run one test file: `pytest backend/tests/unit/test_orchestrator.py -q`
- Run one test by name: `pytest backend/tests -q -k "name_fragment"`
- Install dev tooling: `pip install -r requirements.txt -r requirements-dev.txt`

## Repo Map

- `backend/app/main.py` - FastAPI routes, validation, static frontend mount
- `backend/app/config.py` - environment settings and built-in model specs
- `backend/app/job_defaults.py` - persisted New Job form defaults
- `backend/app/job_store.py` - in-memory job runtime plus persisted job state
- `backend/app/schemas.py` - API and persisted payload schemas
- `backend/pipeline/orchestrator.py` - nine-stage job flow and artifact writes
- `backend/pipeline/` - audio, VAD, diarization, snippets, transcription, formatter helpers
- `backend/models/local_catalog.py` - user-registered local model catalog
- `backend/profiles/manager.py` - voice profiles, embeddings, matching, refresh tasks
- `backend/app/static/` - single-page frontend
- `backend/tests/` - unit and integration coverage
- `data/templates/` - checked-in template examples; other `data/` subdirs are runtime state

## Task Workflow

1. Identify the owning module from the map above and read the smallest relevant files first.
2. If changing job behavior, trace `backend/pipeline/orchestrator.py` from the affected stage to the artifact/API/frontend consumer.
3. If changing request or response shape, update `backend/app/schemas.py`, route logic, frontend assumptions, and tests in the same change.
4. If changing artifacts, update artifact writes, download/preview endpoints, frontend links, and `README.md`.
5. Add or update focused tests near the changed behavior. Prefer unit tests unless the change crosses API, pipeline, and artifact boundaries.
6. Run the narrowest useful pytest command, then run `pytest backend/tests -q` for cross-cutting pipeline/API changes.

## Decision Tables

### Where to make a change

| Change needed | Start here | Also check |
| --- | --- | --- |
| Job stage order, progress, cancellation, artifacts | `backend/pipeline/orchestrator.py` | `backend/tests/unit/test_orchestrator.py`, integration tests |
| API validation or endpoint payload | `backend/app/main.py`, `backend/app/schemas.py` | `backend/app/static/js/app.js` |
| New Job default persistence | `backend/app/job_defaults.py` | `backend/app/static/js/app.js`, `backend/tests/unit/test_job_defaults.py` |
| Local model registration/defaults/discovery | `backend/models/local_catalog.py` | model routes, settings UI |
| Built-in required model specs or env defaults | `backend/app/config.py` | model manager tests, README config |
| Speaker profile matching or refresh | `backend/profiles/manager.py` | snippets, diarization model ids |
| Formatter output quality or validation | `backend/pipeline/formatter.py` | template manager, formatter artifacts |
| Browser workflow or visual state | `backend/app/static/js/app.js` | matching API route payloads |

### Local vs API execution

| Situation | Do this |
| --- | --- |
| A stage runs locally | Require and resolve only that stage's local model selection. |
| Any stage uses OpenAI | Require an OpenAI API key and create `OpenAIJobConfig`. |
| OpenAI diarization returns text segments | Reuse those segments; do not upload the same audio again for cloud transcription unless behavior intentionally changes. |
| Local diarization/profile code needs embeddings | Key profile data by local diarization model id plus embedding model ref. |
| Formatter has an OpenAI API key | Use the Responses API path before local Ollama/command fallback. |

## Stable Contracts

- The orchestrator has nine user-visible stages: Validate/Normalize, VAD, Diarization, Snippet extraction, Speaker naming, Transcription, Transcript assembly, MoM formatting, Export.
- Keep stage names, progress semantics, and cancellation checks stable unless the UI and tests are updated together.
- Per-job files live under `data/jobs/<job_id>/`. Do not commit generated jobs, uploads, local models, profiles, or `.env` files.
- Artifact keys are API/UI contracts. Before renaming or removing one, search for `set_artifact`, `artifact_paths`, and frontend URL builders.
- Common artifact keys include `audio_normalized`, `vad_regions`, `diarization`, `diarization_chunks`, `diarization_stitching`, `snippets`, `speaker_mapping`, `segments_transcript`, `transcript`, `transcription_runtime`, `full_meeting_transcript`, `mom_markdown`, `mom_structured`, `formatter_system_prompt`, `formatter_user_prompt`, `formatter_stdout`, `formatter_stderr`, `formatter_raw_output`, `formatter_validation`, `formatter_reduced_notes`, `openai_audio_chunks`, `export_markdown`, and `job_summary`.
- The frontend expects tabs named `new-job`, `progress`, `result`, and `settings`, plus SSE events from `/api/jobs/{job_id}/events`.
- New Job defaults are persisted by `/api/job-defaults`; do not store meeting titles, uploaded file paths, or OpenAI API keys there.

## Code Patterns

Use Doxygen-style docstrings for Python functions and methods:

```python
def set_artifact(self, job_id: str, key: str, path: Path) -> None:
    """! @brief Set artifact.
    @param job_id Identifier of the job being processed.
    @param key Artifact key exposed through job state.
    @param path Filesystem path for the artifact.
    """
```

Resolve per-stage execution before creating a job:

```python
required_local_selections = {
    stage: model_id
    for stage, model_id in selected_local_models.items()
    if execution_values[f"{stage}_execution"] == "local"
}
```

Write artifacts through the job store so API state, SSE updates, and download endpoints can see them:

```python
write_json(job_dir / "transcript.json", transcript_payload)
JOB_STORE.set_artifact(job_id, "transcript", job_dir / "transcript.json")
```

## Gotchas

- Do not add warning-only rules. When documenting a prohibition, include the replacement pattern.
- Do not instantiate new ad hoc local model stores. Use `LOCAL_MODEL_CATALOG` for user-registered model runtimes.
- Do not bypass `JOB_STORE` when a generated file should appear in job state. Write the file, then call `JOB_STORE.set_artifact`.
- Do not require OpenAI credentials for fully local jobs. Require them only when at least one execution selector is `api`.
- Do not treat README endpoint/config lists as automatically current. If code changes them, update the README in the same change.
- Keep inline comments for fragile logic such as diarization chunk stitching, GPU fallback, formatter retry/validation, and long-input reduction; skip comments that restate obvious code.

## Documentation Rules

- Python functions and methods must keep Doxygen-style docstrings.
- Frontend JavaScript functions must keep JSDoc/Doxygen-style comments.
- Update `README.md` and this file when behavior, artifacts, public endpoints, scripts, or configuration materially change.
- Keep this file concise. Move expanding detail to a focused reference file and link it here only when agents need to load it on demand.
