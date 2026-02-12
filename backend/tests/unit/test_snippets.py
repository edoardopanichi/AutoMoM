from __future__ import annotations

from pathlib import Path

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
