from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    size_mb: int
    source: str
    required_disk_mb: int
    file_path: Path
    checksum_sha256: str | None = None
    download_url: str | None = None


@dataclass(frozen=True)
class Settings:
    app_name: str = "AutoMoM"
    host: str = os.getenv("AUTOMOM_HOST", "127.0.0.1")
    port: int = int(os.getenv("AUTOMOM_PORT", "8000"))
    max_workers: int = int(os.getenv("AUTOMOM_MAX_WORKERS", "1"))

    data_dir: Path = DATA_DIR
    jobs_dir: Path = DATA_DIR / "jobs"
    models_dir: Path = DATA_DIR / "models"
    templates_dir: Path = DATA_DIR / "templates"
    profiles_dir: Path = DATA_DIR / "profiles"
    uploads_dir: Path = DATA_DIR / "uploads"

    ffmpeg_bin: str = os.getenv("AUTOMOM_FFMPEG_BIN", "ffmpeg")
    diarization_backend: str = os.getenv("AUTOMOM_DIARIZATION_BACKEND", "auto")
    diarization_model_path: str = os.getenv(
        "AUTOMOM_DIARIZATION_MODEL",
        str(DATA_DIR / "models" / "diarization" / "model.bin"),
    )
    diarization_embedding_model: str = os.getenv(
        "AUTOMOM_DIARIZATION_EMBEDDING_MODEL",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
    )
    voxtral_binary: str = os.getenv("AUTOMOM_VOXTRAL_BIN", "")
    voxtral_model_path: str = os.getenv(
        "AUTOMOM_VOXTRAL_MODEL",
        str(DATA_DIR / "models" / "voxtral" / "model.gguf"),
    )
    formatter_command: str = os.getenv("AUTOMOM_FORMATTER_COMMAND", "")
    formatter_model_path: str = os.getenv(
        "AUTOMOM_FORMATTER_MODEL",
        str(DATA_DIR / "models" / "formatter" / "model.gguf"),
    )
    formatter_timeout_s: int = int(os.getenv("AUTOMOM_FORMATTER_TIMEOUT_S", "120"))
    diarization_max_chunk_s: float = float(os.getenv("AUTOMOM_DIARIZATION_MAX_CHUNK_S", "18.0"))
    transcription_max_segments: int = int(os.getenv("AUTOMOM_TRANSCRIPTION_MAX_SEGMENTS", "0"))


SETTINGS = Settings()


def ensure_directories() -> None:
    for path in [
        SETTINGS.data_dir,
        SETTINGS.jobs_dir,
        SETTINGS.models_dir,
        SETTINGS.templates_dir,
        SETTINGS.profiles_dir,
        SETTINGS.uploads_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def required_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_id="diarization",
            name="Offline Diarization Model",
            size_mb=120,
            source="Local package / pyannote-compatible weights",
            required_disk_mb=150,
            file_path=SETTINGS.models_dir / "diarization" / "model.bin",
            download_url=os.getenv("AUTOMOM_DIARIZATION_URL"),
            checksum_sha256=os.getenv("AUTOMOM_DIARIZATION_SHA256"),
        ),
        ModelSpec(
            model_id="voxtral",
            name="Voxtral ASR Weights",
            size_mb=3800,
            source="Mistral Voxtral weights (local)",
            required_disk_mb=4200,
            file_path=SETTINGS.models_dir / "voxtral" / "model.gguf",
            download_url=os.getenv("AUTOMOM_VOXTRAL_URL"),
            checksum_sha256=os.getenv("AUTOMOM_VOXTRAL_SHA256"),
        ),
        ModelSpec(
            model_id="formatter",
            name="Formatter LLM (llama.cpp)",
            size_mb=4200,
            source="Local quantized instruction model",
            required_disk_mb=5000,
            file_path=SETTINGS.models_dir / "formatter" / "model.gguf",
            download_url=os.getenv("AUTOMOM_FORMATTER_URL"),
            checksum_sha256=os.getenv("AUTOMOM_FORMATTER_SHA256"),
        ),
    ]
