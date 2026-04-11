from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def _resolve_model_path(raw_path: str) -> Path:
    """! @brief Resolve model path.
    @param raw_path Value for raw path.
    @return Path result produced by the operation.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _env_first(*names: str, default: str = "") -> str:
    """! @brief Return the first non-empty environment variable value.
    @param names Environment variable names to inspect in order.
    @param default Value used when no environment variable is set.
    @return Selected environment value or default.
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _default_transcription_model_path() -> str:
    """! @brief Default transcription model path.
    @return Path string for the local transcription model.
    """
    current_path = DATA_DIR / "models" / "transcription" / "model.gguf"
    legacy_path = DATA_DIR / "models" / "voxtral" / "model.gguf"
    if not current_path.exists() and legacy_path.exists():
        return str(legacy_path)
    return str(current_path)


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
    max_workers: int = max(2, int(os.getenv("AUTOMOM_MAX_WORKERS", "2")))

    data_dir: Path = DATA_DIR
    jobs_dir: Path = DATA_DIR / "jobs"
    models_dir: Path = DATA_DIR / "models"
    templates_dir: Path = DATA_DIR / "templates"
    profiles_dir: Path = DATA_DIR / "profiles"
    uploads_dir: Path = DATA_DIR / "uploads"

    ffmpeg_bin: str = os.getenv("AUTOMOM_FFMPEG_BIN", "ffmpeg")
    compute_device: str = os.getenv("AUTOMOM_COMPUTE_DEVICE", "auto")
    cuda_device_id: int = int(os.getenv("AUTOMOM_CUDA_DEVICE_ID", "0"))
    diarization_backend: str = os.getenv("AUTOMOM_DIARIZATION_BACKEND", "auto")
    diarization_model_path: str = os.getenv(
        "AUTOMOM_DIARIZATION_MODEL",
        str(DATA_DIR / "models" / "diarization" / "pyannote-speaker-diarization-community-1" / "config.yaml"),
    )
    diarization_pipeline_path: str = os.getenv("AUTOMOM_DIARIZATION_PIPELINE", "")
    diarization_min_speakers: int = int(os.getenv("AUTOMOM_DIARIZATION_MIN_SPEAKERS", "0"))
    diarization_max_speakers: int = int(os.getenv("AUTOMOM_DIARIZATION_MAX_SPEAKERS", "0"))
    diarization_embedding_model: str = os.getenv(
        "AUTOMOM_DIARIZATION_EMBEDDING_MODEL",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
    )
    diarization_pyannote_chunk_s: float = float(os.getenv("AUTOMOM_DIARIZATION_PYANNOTE_CHUNK_S", "300.0"))
    transcription_binary: str = _env_first("AUTOMOM_TRANSCRIPTION_BIN", "AUTOMOM_VOXTRAL_BIN")
    transcription_model_path: str = _env_first(
        "AUTOMOM_TRANSCRIPTION_MODEL",
        "AUTOMOM_VOXTRAL_MODEL",
        default=_default_transcription_model_path(),
    )
    transcription_threads: int = max(
        1,
        int(
            _env_first(
                "AUTOMOM_TRANSCRIPTION_THREADS",
                "AUTOMOM_VOXTRAL_THREADS",
                default=str(min(os.cpu_count() or 4, 8)),
            )
        ),
    )
    transcription_processors: int = max(
        1,
        int(_env_first("AUTOMOM_TRANSCRIPTION_PROCESSORS", "AUTOMOM_VOXTRAL_PROCESSORS", default="1")),
    )
    transcription_gpu_layers: int = int(
        _env_first("AUTOMOM_TRANSCRIPTION_GPU_LAYERS", "AUTOMOM_VOXTRAL_GPU_LAYERS", default="99")
    )
    formatter_backend: str = os.getenv("AUTOMOM_FORMATTER_BACKEND", "ollama").strip().lower()
    formatter_command: str = os.getenv("AUTOMOM_FORMATTER_COMMAND", "")
    formatter_model_path: str = os.getenv(
        "AUTOMOM_FORMATTER_MODEL",
        str(DATA_DIR / "models" / "formatter" / "model.gguf"),
    )
    formatter_gpu_layers: int = int(os.getenv("AUTOMOM_FORMATTER_GPU_LAYERS", "99"))
    ollama_host: str = os.getenv("AUTOMOM_OLLAMA_HOST", "http://127.0.0.1:11434")
    formatter_ollama_model: str = os.getenv(
        "AUTOMOM_FORMATTER_OLLAMA_MODEL", "qwen2.5:3b-instruct-q5_K_M")
    formatter_timeout_s: int = int(os.getenv("AUTOMOM_FORMATTER_TIMEOUT_S", "300"))
    diarization_max_chunk_s: float = float(os.getenv("AUTOMOM_DIARIZATION_MAX_CHUNK_S", "18.0"))
    transcription_max_segments: int = int(os.getenv("AUTOMOM_TRANSCRIPTION_MAX_SEGMENTS", "0"))
    transcription_merge_gap_s: float = float(os.getenv("AUTOMOM_TRANSCRIPTION_MERGE_GAP_S", "1.5"))
    transcription_max_chunk_s: float = float(os.getenv("AUTOMOM_TRANSCRIPTION_MAX_CHUNK_S", "20.0"))
    transcription_keep_segment_audio: bool = os.getenv(
        "AUTOMOM_TRANSCRIPTION_KEEP_SEGMENT_AUDIO",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}


SETTINGS = Settings()


def ensure_directories() -> None:
    """! @brief Ensure directories.
    """
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
    """! @brief Required models.
    @return List produced by the operation.
    """
    return [
        ModelSpec(
            model_id="diarization",
            name="Offline Diarization Model",
            size_mb=900,
            source="Local pyannote speaker diarization pipeline",
            required_disk_mb=1200,
            file_path=_resolve_model_path(SETTINGS.diarization_model_path),
            download_url=os.getenv("AUTOMOM_DIARIZATION_URL"),
            checksum_sha256=os.getenv("AUTOMOM_DIARIZATION_SHA256"),
        ),
        ModelSpec(
            model_id="transcription",
            name="Transcription ASR Weights",
            size_mb=3800,
            source="Local whisper.cpp-compatible ASR model",
            required_disk_mb=4200,
            file_path=_resolve_model_path(SETTINGS.transcription_model_path),
            download_url=_env_first("AUTOMOM_TRANSCRIPTION_URL", "AUTOMOM_VOXTRAL_URL") or None,
            checksum_sha256=_env_first("AUTOMOM_TRANSCRIPTION_SHA256", "AUTOMOM_VOXTRAL_SHA256") or None,
        ),
        ModelSpec(
            model_id="formatter",
            name="Formatter LLM (Ollama)",
            size_mb=4200,
            source="Ollama local model registry",
            required_disk_mb=5000,
            file_path=SETTINGS.models_dir / "formatter" / "selected_model.txt",
        ),
    ]
