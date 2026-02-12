from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import SETTINGS, ensure_directories
from backend.app.job_store import JOB_STORE
from backend.app.schemas import (
    CreateVoiceProfileRequest,
    JobListResponse,
    ModelConsentRequest,
    ModelDownloadRequest,
    SubmitSpeakerMappingRequest,
    TemplateDefinition,
)
from backend.models.manager import MODEL_MANAGER
from backend.pipeline.orchestrator import ORCHESTRATOR
from backend.pipeline.template_manager import TEMPLATE_MANAGER
from backend.profiles.manager import VOICE_PROFILE_MANAGER


ensure_directories()

app = FastAPI(title=SETTINGS.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> list[dict[str, object]]:
    return [item.model_dump() for item in MODEL_MANAGER.statuses()]


@app.post("/api/models/consent")
def model_consent(request: ModelConsentRequest) -> dict[str, str]:
    try:
        MODEL_MANAGER.set_consent(request.model_id, request.approved)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown model") from exc
    return {"status": "ok"}


@app.post("/api/models/download")
def model_download(request: ModelDownloadRequest) -> dict[str, object]:
    try:
        result = MODEL_MANAGER.start_download(request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown model") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.get("/api/models/downloads")
def model_downloads() -> list[dict[str, object]]:
    return MODEL_MANAGER.all_download_statuses()


@app.get("/api/models/downloads/{model_id}")
def model_download_status(model_id: str) -> dict[str, object]:
    try:
        return MODEL_MANAGER.download_status(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown model") from exc


@app.get("/api/templates")
def list_templates() -> list[dict[str, object]]:
    return [item.model_dump() for item in TEMPLATE_MANAGER.list_templates()]


@app.get("/api/templates/{template_id}")
def get_template(template_id: str) -> dict[str, object]:
    try:
        template = TEMPLATE_MANAGER.load(template_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Template not found") from exc
    return template.model_dump()


@app.post("/api/templates")
def save_template(template: TemplateDefinition) -> dict[str, str]:
    TEMPLATE_MANAGER.save(template)
    return {"status": "ok"}


@app.delete("/api/templates/{template_id}")
def delete_template(template_id: str) -> dict[str, str]:
    try:
        TEMPLATE_MANAGER.delete(template_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@app.get("/api/profiles")
def list_profiles() -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in VOICE_PROFILE_MANAGER.list_profiles()]


@app.post("/api/profiles")
def create_profile(request: CreateVoiceProfileRequest) -> dict[str, object]:
    if not request.audio_path:
        raise HTTPException(status_code=400, detail="audio_path is required")

    audio_path = Path(request.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="audio_path not found")

    embedding = VOICE_PROFILE_MANAGER.compute_embedding(audio_path)
    profile = VOICE_PROFILE_MANAGER.upsert_profile(request.name, embedding)
    return profile.model_dump(mode="json")


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str) -> dict[str, str]:
    VOICE_PROFILE_MANAGER.delete(profile_id)
    return {"status": "ok"}


@app.post("/api/jobs")
async def create_job(
    audio_file: UploadFile = File(...),
    template_id: str = Form("default"),
    language_mode: str = Form("auto"),
    title: str | None = Form(None),
) -> dict[str, object]:
    validation_ok, message = MODEL_MANAGER.validate_for_job_start()
    if not validation_ok:
        raise HTTPException(status_code=400, detail=message)

    try:
        TEMPLATE_MANAGER.load(template_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found") from exc

    suffix = Path(audio_file.filename or "input.wav").suffix.lower() or ".wav"
    target_path = SETTINGS.uploads_dir / f"{uuid4()}{suffix}"
    payload = await audio_file.read()
    target_path.write_bytes(payload)

    runtime = JOB_STORE.create_job(
        audio_path=target_path,
        template_id=template_id,
        language_mode=language_mode,
        title=title,
    )
    ORCHESTRATOR.submit(runtime.state.job_id)

    return {
        "job_id": runtime.state.job_id,
        "status": runtime.state.status,
        "created_at": runtime.state.created_at,
    }


@app.get("/api/jobs")
def list_jobs() -> JobListResponse:
    return JobListResponse(jobs=JOB_STORE.list_states())


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        state = JOB_STORE.get_state(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return state.model_dump(mode="json")


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    try:
        JOB_STORE.get_state(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    async def event_stream():
        terminal = {"completed", "failed", "cancelled"}
        while True:
            state = JOB_STORE.get_state(job_id)
            yield f"data: {json.dumps(state.model_dump(mode='json'))}\n\n"
            if state.status in terminal:
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    try:
        ORCHESTRATOR.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return {"status": "cancelled"}


@app.post("/api/jobs/{job_id}/speaker-mapping")
def submit_speaker_mapping(job_id: str, request: SubmitSpeakerMappingRequest) -> dict[str, str]:
    try:
        JOB_STORE.submit_speaker_mapping(job_id, request.mappings)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return {"status": "ok"}


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
def get_artifact(job_id: str, artifact_name: str) -> FileResponse:
    try:
        state = JOB_STORE.get_state(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if artifact_name not in state.artifact_paths:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(state.artifact_paths[artifact_name])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact path missing")

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=content_type, filename=path.name)


@app.get("/api/jobs/{job_id}/snippets/{snippet_name}")
def get_snippet(job_id: str, snippet_name: str) -> FileResponse:
    path = SETTINGS.jobs_dir / job_id / "snippets" / snippet_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snippet not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/api/jobs/{job_id}/download/mom")
def download_mom(job_id: str) -> FileResponse:
    try:
        state = JOB_STORE.get_state(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    export_path = state.artifact_paths.get("export_markdown") or state.artifact_paths.get("mom_markdown")
    if not export_path:
        raise HTTPException(status_code=404, detail="MoM export not available")

    path = Path(export_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file missing")

    return FileResponse(path, media_type="text/markdown", filename=f"{job_id}_mom.md")


@app.get("/api/system/startup-check")
def startup_check() -> dict[str, object]:
    statuses = [item.model_dump() for item in MODEL_MANAGER.statuses()]
    all_ready = all(item["installed"] for item in statuses)
    return {"all_models_ready": all_ready, "models": statuses}
