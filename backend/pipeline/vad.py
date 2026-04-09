from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class SpeechRegion:
    start_s: float
    end_s: float



def detect_speech_regions(
    audio_path: Path,
    frame_ms: int = 30,
    energy_threshold_ratio: float = 0.6,
    min_region_ms: int = 250,
) -> list[SpeechRegion]:
    """! @brief Detect speech regions.
    @param audio_path Path to the audio file.
    @param frame_ms Value for frame ms.
    @param energy_threshold_ratio Value for energy threshold ratio.
    @param min_region_ms Value for min region ms.
    @return List produced by the operation.
    """
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return []

    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    num_frames = int(np.ceil(len(audio) / frame_size))
    energies = []
    for idx in range(num_frames):
        frame = audio[idx * frame_size : (idx + 1) * frame_size]
        if len(frame) == 0:
            energies.append(0.0)
        else:
            energies.append(float(np.sqrt(np.mean(np.square(frame)))))

    median_energy = float(np.median(energies))
    threshold = max(1e-4, median_energy * energy_threshold_ratio)
    active = np.array([energy >= threshold for energy in energies], dtype=bool)

    regions: list[SpeechRegion] = []
    start_idx: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        if not is_active and start_idx is not None:
            start_s = start_idx * frame_size / sample_rate
            end_s = idx * frame_size / sample_rate
            if (end_s - start_s) * 1000 >= min_region_ms:
                regions.append(SpeechRegion(start_s=start_s, end_s=end_s))
            start_idx = None

    if start_idx is not None:
        start_s = start_idx * frame_size / sample_rate
        end_s = len(audio) / sample_rate
        if (end_s - start_s) * 1000 >= min_region_ms:
            regions.append(SpeechRegion(start_s=start_s, end_s=end_s))

    return regions
