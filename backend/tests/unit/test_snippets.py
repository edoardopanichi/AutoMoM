from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from backend.pipeline.diarization import DiarizationSegment
from backend.pipeline.snippets import extract_snippets, pick_snippet_ranges


def test_pick_snippet_ranges_groups_by_speaker() -> None:
    segments = [
        DiarizationSegment("SPEAKER_0", 0.0, 6.0),
        DiarizationSegment("SPEAKER_0", 7.0, 11.0),
        DiarizationSegment("SPEAKER_1", 12.0, 19.0),
    ]

    selected = pick_snippet_ranges(segments, per_speaker=2)

    assert "SPEAKER_0" in selected
    assert "SPEAKER_1" in selected
    assert len(selected["SPEAKER_0"]) <= 2


def test_extract_snippets_invokes_segment_extraction(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_extract_segment(input_path, output_path, start_s, end_s, ffmpeg_bin):
        calls.append((input_path, output_path, start_s, end_s, ffmpeg_bin))
        output_path.write_bytes(b"wav")

    monkeypatch.setattr("backend.pipeline.snippets.extract_segment", fake_extract_segment)

    ranges = {"SPEAKER_0": [(0.0, 4.0), (6.0, 9.0)]}
    out = extract_snippets(tmp_path / "src.wav", tmp_path / "snippets", ranges, ffmpeg_bin="ffmpeg")

    assert len(out) == 2
    assert calls[0][4] == "ffmpeg"
    assert out[0].path.exists()


def test_pick_snippet_ranges_prefers_clearer_audio_regions(tmp_path: Path) -> None:
    sample_rate = 16000
    quiet = np.full(sample_rate * 6, 0.005, dtype=np.float32)
    clear = np.full(sample_rate * 6, 0.2, dtype=np.float32)
    audio = np.concatenate([quiet, clear])
    audio_path = tmp_path / "speaker.wav"
    sf.write(audio_path, audio, sample_rate)

    segments = [
        DiarizationSegment("SPEAKER_0", 0.0, 6.0, confidence=0.9),
        DiarizationSegment("SPEAKER_0", 6.0, 12.0, confidence=0.9),
    ]

    selected = pick_snippet_ranges(segments, per_speaker=1, audio_path=audio_path)

    assert selected["SPEAKER_0"] == [(6.0, 12.0)]
