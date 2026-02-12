from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioError(RuntimeError):
    pass


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def validate_audio_input(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(f"Unsupported file format: {path.suffix}")



def normalize_audio(input_path: Path, output_path: Path, ffmpeg_bin: str = "ffmpeg") -> dict[str, float | int | str]:
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
        str(output_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise AudioError(process.stderr.strip() or "Audio normalization failed")

    return audio_metadata(output_path)



def audio_metadata(audio_path: Path) -> dict[str, float | int | str]:
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
) -> None:
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
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise AudioError(process.stderr.strip() or f"Segment extraction failed for {output_path}")
