from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from backend.pipeline.vad import detect_speech_regions


def test_detect_speech_regions_finds_active_interval(tmp_path: Path) -> None:
    """! @brief Test detect speech regions finds active interval.
    @param tmp_path Value for tmp path.
    """
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    speech = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    path = tmp_path / "vad.wav"
    sf.write(path, audio, sample_rate)

    regions = detect_speech_regions(path)

    assert regions
    assert regions[0].start_s < 1.2
    assert regions[0].end_s > 1.7
