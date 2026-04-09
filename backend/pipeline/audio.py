from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.pipeline.subprocess_utils import run_cancellable_subprocess


class AudioError(RuntimeError):
    pass


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac"}
DENOISE_ENV_FLAG = "AUTOMOM_AUDIO_DENOISE"
DENOISE_FILTER_ENV = "AUTOMOM_AUDIO_DENOISE_FILTER"
DEFAULT_DENOISE_FILTER = "afftdn"


def validate_audio_input(path: Path) -> None:
    """! @brief Validate audio input.
    @param path Filesystem path used by the operation.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(f"Unsupported file format: {path.suffix}")


def _env_bool(name: str, default: bool) -> bool:
    """! @brief Env bool.
    @param name Value for name.
    @param default Value for default.
    @return True when the requested condition is satisfied; otherwise False.
    """
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_denoise_settings(
    denoise_enabled: bool | None,
    denoise_filter: str | None,
) -> tuple[bool, str]:
    """! @brief Resolve denoise settings.
    @param denoise_enabled Value for denoise enabled.
    @param denoise_filter Value for denoise filter.
    @return Tuple produced by the operation.
    """
    enabled = _env_bool(DENOISE_ENV_FLAG, True) if denoise_enabled is None else denoise_enabled
    filter_expr = (denoise_filter or os.getenv(DENOISE_FILTER_ENV, DEFAULT_DENOISE_FILTER)).strip()
    if enabled and not filter_expr:
        filter_expr = DEFAULT_DENOISE_FILTER
    return enabled, filter_expr



def normalize_audio(
    input_path: Path,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    *,
    job_id: str | None = None,
    denoise_enabled: bool | None = None,
    denoise_filter: str | None = None,
) -> dict[str, float | int | str]:
    """! @brief Normalize audio.
    @param input_path Path to the input file.
    @param output_path Path to the output file.
    @param ffmpeg_bin Value for ffmpeg bin.
    @param job_id Identifier of the job being processed.
    @param denoise_enabled Value for denoise enabled.
    @param denoise_filter Value for denoise filter.
    @return Dictionary produced by the operation.
    """
    enabled, filter_expr = _resolve_denoise_settings(denoise_enabled, denoise_filter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if enabled:
        command.extend(["-af", filter_expr])
    command.append(str(output_path))
    process = run_cancellable_subprocess(command, job_id=job_id)
    if process.returncode != 0:
        raise AudioError(process.stderr.strip() or "Audio normalization failed")

    metadata = audio_metadata(output_path)
    metadata["denoise_enabled"] = int(enabled)
    metadata["denoise_filter"] = filter_expr if enabled else ""
    return metadata



def audio_metadata(audio_path: Path) -> dict[str, float | int | str]:
    """! @brief Audio metadata.
    @param audio_path Path to the audio file.
    @return Dictionary produced by the operation.
    """
    data, sample_rate = sf.read(str(audio_path), always_2d=False)
    if isinstance(data, np.ndarray) and data.ndim > 1:
        channels = data.shape[1]
        duration = data.shape[0] / sample_rate
    else:
        channels = 1
        duration = len(data) / sample_rate

    return {
        "path": str(audio_path),
        "duration_s": float(duration),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
    }



def extract_segment(
    input_path: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    ffmpeg_bin: str = "ffmpeg",
    *,
    job_id: str | None = None,
) -> None:
    """! @brief Extract segment.
    @param input_path Path to the input file.
    @param output_path Path to the output file.
    @param start_s Value for start s.
    @param end_s Value for end s.
    @param ffmpeg_bin Value for ffmpeg bin.
    @param job_id Identifier of the job being processed.
    """
    duration = max(0.0, end_s - start_s)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    process = run_cancellable_subprocess(command, job_id=job_id)
    if process.returncode != 0:
        raise AudioError(process.stderr.strip() or f"Segment extraction failed for {output_path}")
