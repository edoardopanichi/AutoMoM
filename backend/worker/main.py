from __future__ import annotations

import json
import os
import socket
import tempfile
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from backend.app.config import SETTINGS
from backend.pipeline.audio import normalize_audio
from backend.pipeline.diarization import (
    DiarizationSegment,
    compute_profile_embedding,
    diarize,
    pyannote_audio_version,
    resolve_profile_embedding_model_ref,
)
from backend.pipeline.transcription import WhisperCppTranscriber
from backend.pipeline.vad import detect_speech_regions


APP_VERSION = "0.1.0"
AUTH_TOKEN = os.getenv("AUTOMOM_REMOTE_AUTH_TOKEN", "").strip()
ENABLED_STAGES = {
    item.strip().lower()
    for item in os.getenv("AUTOMOM_REMOTE_WORKER_ENABLED_STAGES", "diarization,transcription").split(",")
    if item.strip()
}
DIARIZATION_PIPELINE = os.getenv("AUTOMOM_REMOTE_DIARIZATION_PIPELINE", SETTINGS.diarization_pipeline_path or SETTINGS.diarization_model_path).strip()
DIARIZATION_MODEL_NAME = os.getenv("AUTOMOM_REMOTE_DIARIZATION_MODEL_NAME", "pyannote-community-1").strip()
PROFILE_MODEL_REF = os.getenv("AUTOMOM_REMOTE_PROFILE_MODEL_REF", DIARIZATION_MODEL_NAME).strip() or DIARIZATION_MODEL_NAME
DIARIZATION_EMBEDDING_MODEL = os.getenv(
    "AUTOMOM_REMOTE_DIARIZATION_EMBEDDING_MODEL",
    resolve_profile_embedding_model_ref(pipeline_path=DIARIZATION_PIPELINE, embedding_model=SETTINGS.diarization_embedding_model),
).strip()
TRANSCRIPTION_BINARY = os.getenv("AUTOMOM_REMOTE_TRANSCRIPTION_BIN", SETTINGS.transcription_binary).strip()
TRANSCRIPTION_MODEL = os.getenv("AUTOMOM_REMOTE_TRANSCRIPTION_MODEL", SETTINGS.transcription_model_path).strip()
TRANSCRIPTION_MODEL_NAME = os.getenv("AUTOMOM_REMOTE_TRANSCRIPTION_MODEL_NAME", Path(TRANSCRIPTION_MODEL).stem or "whisper.cpp").strip()

app = FastAPI(title="AutoMoM Remote Worker", version=APP_VERSION)


def _require_auth(authorization: str | None) -> None:
    if not AUTH_TOKEN:
        return
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    finally:
        handle.close()
        upload.file.close()
    return Path(handle.name)


def _load_normalized_audio(upload: UploadFile) -> tuple[Path, Path]:
    uploaded_path = _save_upload(upload)
    normalized_path = uploaded_path.with_name(f"{uploaded_path.stem}-normalized.wav")
    try:
        normalize_audio(uploaded_path, normalized_path, ffmpeg_bin=SETTINGS.ffmpeg_bin)
    except Exception:
        uploaded_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)
        raise
    return uploaded_path, normalized_path


def _speaker_ranges(segments: list[DiarizationSegment]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in segments:
        grouped[item.speaker_id].append((float(item.start_s), float(item.end_s)))
    return grouped


@app.get("/health")
def health(authorization: str | None = Header(default=None)) -> dict[str, object]:
    _require_auth(authorization)
    payload: dict[str, object] = {
        "status": "ok",
        "version": APP_VERSION,
        "hostname": socket.gethostname(),
        "enabled_stages": sorted(ENABLED_STAGES),
        "auth_required": bool(AUTH_TOKEN),
    }
    if "diarization" in ENABLED_STAGES:
        payload["diarization"] = {
            "runtime": "pyannote",
            "model_name": DIARIZATION_MODEL_NAME,
            "profile_model_ref": PROFILE_MODEL_REF,
            "embedding_model_ref": DIARIZATION_EMBEDDING_MODEL,
        }
    if "transcription" in ENABLED_STAGES:
        payload["transcription"] = {
            "runtime": "whisper.cpp",
            "model_name": TRANSCRIPTION_MODEL_NAME,
        }
    return payload


@app.post("/diarize")
def diarize_audio(
    audio_file: UploadFile = File(...),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _require_auth(authorization)
    if "diarization" not in ENABLED_STAGES:
        raise HTTPException(status_code=404, detail="Diarization is disabled")
    uploaded_path, normalized_path = _load_normalized_audio(audio_file)
    try:
        speech_regions = detect_speech_regions(normalized_path, ffmpeg_bin=SETTINGS.ffmpeg_bin)
        result = diarize(
            normalized_path,
            speech_regions,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            backend="pyannote",
            model_path=Path(DIARIZATION_PIPELINE),
            pipeline_path=DIARIZATION_PIPELINE,
            embedding_model=DIARIZATION_EMBEDDING_MODEL,
            compute_device=SETTINGS.compute_device,
            cuda_device_id=SETTINGS.cuda_device_id,
        )
        speaker_embeddings = {
            speaker_id: compute_profile_embedding(
                normalized_path,
                model_ref=DIARIZATION_EMBEDDING_MODEL,
                compute_device=SETTINGS.compute_device,
                cuda_device_id=SETTINGS.cuda_device_id,
                segments=ranges,
            ).astype(float).tolist()
            for speaker_id, ranges in _speaker_ranges(result.segments).items()
        }
        return {
            "segments": result.to_json(),
            "speaker_count": result.speaker_count,
            "mode": "remote-pyannote",
            "details": socket.gethostname(),
            "profile_model_ref": PROFILE_MODEL_REF,
            "embedding_model_ref": DIARIZATION_EMBEDDING_MODEL,
            "library_version": pyannote_audio_version(),
            "engine_kind": "remote_pyannote",
            "speaker_embeddings": speaker_embeddings,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        uploaded_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)


@app.post("/embed")
def embed_audio(
    audio_file: UploadFile = File(...),
    clip_ranges: str = Form(default="[]"),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _require_auth(authorization)
    if "diarization" not in ENABLED_STAGES:
        raise HTTPException(status_code=404, detail="Diarization is disabled")
    uploaded_path, normalized_path = _load_normalized_audio(audio_file)
    try:
        ranges = [
            (float(item.get("start_s", 0.0)), float(item.get("end_s", 0.0)))
            for item in json.loads(clip_ranges or "[]")
        ]
        vector = compute_profile_embedding(
            normalized_path,
            model_ref=DIARIZATION_EMBEDDING_MODEL,
            compute_device=SETTINGS.compute_device,
            cuda_device_id=SETTINGS.cuda_device_id,
            segments=ranges or None,
        )
        return {
            "vector": vector.astype(float).tolist(),
            "threshold": 0.82,
            "library_version": pyannote_audio_version(),
            "profile_model_ref": PROFILE_MODEL_REF,
            "embedding_model_ref": DIARIZATION_EMBEDDING_MODEL,
            "engine_kind": "remote_pyannote",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        uploaded_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)


@app.post("/transcribe")
def transcribe_audio(
    audio_file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _require_auth(authorization)
    if "transcription" not in ENABLED_STAGES:
        raise HTTPException(status_code=404, detail="Transcription is disabled")
    uploaded_path, normalized_path = _load_normalized_audio(audio_file)
    try:
        transcriber = WhisperCppTranscriber(
            TRANSCRIPTION_BINARY,
            TRANSCRIPTION_MODEL,
            compute_device=SETTINGS.compute_device,
            cuda_device_id=SETTINGS.cuda_device_id,
            gpu_layers=SETTINGS.transcription_gpu_layers,
            threads=SETTINGS.transcription_threads,
            processors=SETTINGS.transcription_processors,
        )
        text = transcriber.transcribe(normalized_path)
        runtime = transcriber.runtime_report()
        return {
            "text": text,
            "runtime": {
                "backend": "whisper.cpp",
                "model": TRANSCRIPTION_MODEL_NAME,
                "compute_active": runtime.get("active_mode", "unknown"),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        uploaded_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)
