from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import backend.pipeline.diarization as diarization_module
from backend.pipeline.diarization import diarize, merge_transcript_segments
from backend.pipeline.vad import SpeechRegion


def test_diarize_returns_segments(tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0, 1.5, int(sample_rate * 1.5), endpoint=False)
    speaker_a = 0.25 * np.sin(2 * np.pi * 220 * t)
    speaker_b = 0.25 * np.sin(2 * np.pi * 420 * t)
    audio = np.concatenate([speaker_a, speaker_b]).astype(np.float32)

    path = tmp_path / "diar.wav"
    sf.write(path, audio, sample_rate)

    regions = [SpeechRegion(start_s=0.0, end_s=len(audio) / sample_rate)]
    result = diarize(path, regions, max_speakers=4, max_chunk_s=0.8)

    assert result.speaker_count >= 1
    assert result.segments
    assert result.segments[0].speaker_id.startswith("SPEAKER_")


def test_diarize_forced_pyannote_falls_back_when_not_configured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AUTOMOM_DIARIZATION_PIPELINE", raising=False)
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    path = tmp_path / "diar_fallback.wav"
    sf.write(path, audio, sample_rate)

    regions = [SpeechRegion(start_s=0.0, end_s=1.0)]
    result = diarize(
        path,
        regions,
        max_speakers=2,
        max_chunk_s=0.5,
        backend="pyannote",
        model_path=tmp_path / "missing_model.yaml",
    )

    assert result.mode == "heuristic"
    assert result.details is not None
    assert "pyannote_forced_fallback" in result.details


def test_diarize_auto_uses_embedding_backend_when_pyannote_unavailable(monkeypatch, tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    path = tmp_path / "diar_auto_embedding.wav"
    sf.write(path, audio, sample_rate)

    monkeypatch.setattr(diarization_module, "_diarize_with_pyannote", lambda **_: (None, "no_pipeline"))
    monkeypatch.setattr(
        diarization_module,
        "_diarize_with_embeddings",
        lambda **_: (
            diarization_module.DiarizationResult(
                segments=[diarization_module.DiarizationSegment("SPEAKER_0", 0.0, 1.0, 0.9)],
                speaker_count=1,
                mode="embedding",
                details="embedding_model:test",
            ),
            None,
        ),
    )

    regions = [SpeechRegion(start_s=0.0, end_s=1.0)]
    result = diarization_module.diarize(path, regions, max_speakers=2, max_chunk_s=0.5, backend="auto")

    assert result.mode == "embedding"
    assert result.details == "embedding_model:test"


def test_diarize_forced_embedding_falls_back_to_heuristic(monkeypatch, tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    path = tmp_path / "diar_embedding_fallback.wav"
    sf.write(path, audio, sample_rate)

    monkeypatch.setattr(diarization_module, "_diarize_with_embeddings", lambda **_: (None, "embedding_unavailable"))

    regions = [SpeechRegion(start_s=0.0, end_s=1.0)]
    result = diarization_module.diarize(path, regions, max_speakers=2, max_chunk_s=0.5, backend="embedding")

    assert result.mode == "heuristic"
    assert result.details is not None
    assert "embedding_forced_fallback" in result.details


def test_merge_transcript_segments_merges_adjacent() -> None:
    segments = [
        {"speaker_name": "Alice", "start_s": 0.0, "end_s": 1.0, "text": "Hello"},
        {"speaker_name": "Alice", "start_s": 1.2, "end_s": 2.0, "text": "there"},
        {"speaker_name": "Bob", "start_s": 3.0, "end_s": 4.0, "text": "Hi"},
    ]

    merged = merge_transcript_segments(segments, max_gap_s=0.3)

    assert len(merged) == 2
    assert merged[0]["text"] == "Hello there"
