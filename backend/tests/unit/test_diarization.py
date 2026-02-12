from __future__ import annotations

import builtins
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


def test_pyannote_import_skipped_when_pipeline_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AUTOMOM_DIARIZATION_PIPELINE", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("pyannote"):
            raise AssertionError("pyannote import should not happen without pipeline configuration")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result, error = diarization_module._diarize_with_pyannote(
        audio_path=tmp_path / "dummy.wav",
        min_speakers=1,
        max_speakers=2,
        model_path=tmp_path / "missing.bin",
        compute_device="auto",
        cuda_device_id=0,
    )

    assert result is None
    assert error == "pyannote_pipeline_not_configured"


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
    assert result.details is not None
    assert result.details.startswith("embedding_model:test")


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


def test_merge_transcript_segments_default_merges_same_speaker_with_large_gap() -> None:
    segments = [
        {"speaker_name": "Alice", "start_s": 0.0, "end_s": 1.0, "text": "Hello"},
        {"speaker_name": "Alice", "start_s": 6.5, "end_s": 7.2, "text": "again"},
        {"speaker_name": "Bob", "start_s": 7.3, "end_s": 8.0, "text": "Hi"},
    ]

    merged = merge_transcript_segments(segments)

    assert len(merged) == 2
    assert merged[0]["speaker_name"] == "Alice"
    assert merged[0]["text"] == "Hello again"


def test_estimate_speaker_count_penalizes_singleton_heavy_clusterings(monkeypatch) -> None:
    features = np.zeros((10, 4), dtype=np.float32)
    labels_by_k = {
        2: np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        3: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2]),
        4: np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3]),
        5: np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4]),
    }
    raw_scores = {2: 0.20, 3: 0.27, 4: 0.33, 5: 0.38}

    monkeypatch.setattr(diarization_module, "_cluster", lambda _features, k: labels_by_k[k])
    monkeypatch.setattr(
        diarization_module,
        "_silhouette",
        lambda labels, _distances: raw_scores[len(np.unique(labels))],
    )

    result = diarization_module._estimate_speaker_count(features, max_speakers=5)

    assert result == 2


def test_estimate_speaker_count_can_pick_higher_k_when_clusters_are_stable(monkeypatch) -> None:
    features = np.zeros((12, 4), dtype=np.float32)
    labels_by_k = {
        2: np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        3: np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        4: np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    }
    raw_scores = {2: 0.21, 3: 0.41, 4: 0.33}

    monkeypatch.setattr(diarization_module, "_cluster", lambda _features, k: labels_by_k[k])
    monkeypatch.setattr(
        diarization_module,
        "_silhouette",
        lambda labels, _distances: raw_scores[len(np.unique(labels))],
    )

    result = diarization_module._estimate_speaker_count(features, max_speakers=4)

    assert result == 3
