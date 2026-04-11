from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import backend.pipeline.diarization as diarization_module
from backend.pipeline.diarization import diarize, merge_transcript_segments
from backend.pipeline.vad import SpeechRegion


def test_diarize_returns_segments(tmp_path: Path) -> None:
    """! @brief Test diarize returns segments.
    @param tmp_path Value for tmp path.
    """
    sample_rate = 16000
    t = np.linspace(0, 1.5, int(sample_rate * 1.5), endpoint=False)
    speaker_a = 0.25 * np.sin(2 * np.pi * 220 * t)
    speaker_b = 0.25 * np.sin(2 * np.pi * 420 * t)
    audio = np.concatenate([speaker_a, speaker_b]).astype(np.float32)

    path = tmp_path / "diar.wav"
    sf.write(path, audio, sample_rate)

    regions = [SpeechRegion(start_s=0.0, end_s=len(audio) / sample_rate)]
    result = diarize(path, regions, max_speakers=4, max_chunk_s=0.8, backend="heuristic")

    assert result.speaker_count >= 1
    assert result.segments
    assert result.segments[0].speaker_id.startswith("SPEAKER_")


def test_diarize_forced_pyannote_raises_when_not_configured(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test diarize forced pyannote raises when not configured.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    monkeypatch.delenv("AUTOMOM_DIARIZATION_PIPELINE", raising=False)
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    path = tmp_path / "diar_fallback.wav"
    sf.write(path, audio, sample_rate)

    regions = [SpeechRegion(start_s=0.0, end_s=1.0)]
    with pytest.raises(RuntimeError) as exc_info:
        diarize(
            path,
            regions,
            max_speakers=2,
            max_chunk_s=0.5,
            backend="pyannote",
            model_path=tmp_path / "missing_model.yaml",
        )

    assert "Diarization model/pipeline is not configured" in str(exc_info.value)


def test_pyannote_import_skipped_when_pipeline_missing(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test pyannote import skipped when pipeline missing.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    monkeypatch.delenv("AUTOMOM_DIARIZATION_PIPELINE", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """! @brief Guarded import.
        @param name Value for name.
        @param args Value for args.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        if name.startswith("pyannote"):
            raise AssertionError("pyannote import should not happen without pipeline configuration")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result, error = diarization_module._diarize_with_pyannote(
        audio_path=tmp_path / "dummy.wav",
        speech_regions=[],
        min_speakers=1,
        max_speakers=2,
        model_path=tmp_path / "missing.bin",
        compute_device="auto",
        cuda_device_id=0,
    )

    assert result is None
    assert error == "pyannote_pipeline_not_configured"


def test_plan_chunked_diarization_uses_ceil_20_min_windows() -> None:
    """! @brief Test plan chunked diarization uses ceil 20 min windows.
    """
    chunks = diarization_module._plan_chunked_diarization(
        speech_regions=[],
        total_duration_s=57 * 60.0,
    )

    assert len(chunks) == 3
    assert max(float(item["own_end_s"]) - float(item["own_start_s"]) for item in chunks) <= 20 * 60.0
    assert chunks[0]["own_start_s"] == 0.0
    assert chunks[-1]["own_end_s"] == 57 * 60.0


def test_plan_chunked_diarization_honors_configured_chunk_size() -> None:
    """! @brief Test plan chunked diarization honors configured chunk size.
    """
    chunks = diarization_module._plan_chunked_diarization(
        speech_regions=[],
        total_duration_s=57 * 60.0,
        max_chunk_s=5 * 60.0,
    )

    assert len(chunks) == 12
    assert max(float(item["own_end_s"]) - float(item["own_start_s"]) for item in chunks) <= 5 * 60.0


def test_pyannote_memory_error_message_is_actionable(tmp_path: Path) -> None:
    """! @brief Test pyannote memory error message is actionable.
    @param tmp_path Path provided by pytest.
    """
    message = diarization_module._pyannote_error_message(
        "pyannote_runtime_error:MemoryError",
        model_path=tmp_path / "config.yaml",
        pipeline_path=str(tmp_path / "config.yaml"),
    )

    assert "ran out of memory" in message
    assert "shorter local diarization chunk" in message


def test_filter_segments_to_owned_window_clips_and_deduplicates_overlap() -> None:
    """! @brief Test filter segments to owned window clips and deduplicates overlap.
    """
    segments = [
        diarization_module.DiarizationSegment("A", 590.0, 605.0),
        diarization_module.DiarizationSegment("A", 605.0, 615.0),
        diarization_module.DiarizationSegment("B", 1185.0, 1210.0),
    ]

    filtered = diarization_module._filter_segments_to_owned_window(
        segments,
        own_start_s=600.0,
        own_end_s=1200.0,
    )

    assert [(item.speaker_id, item.start_s, item.end_s) for item in filtered] == [
        ("A", 605.0, 615.0),
        ("B", 1185.0, 1200.0),
    ]


def test_assign_chunk_speakers_to_global_reuses_matching_speaker() -> None:
    """! @brief Test assign chunk speakers to global reuses matching speaker.
    """
    representative = {
        "chunk_speaker_0": np.array([1.0, 0.0], dtype=np.float32),
        "chunk_speaker_1": np.array([0.0, 1.0], dtype=np.float32),
    }
    global_bank = {
        "GLOBAL_0": [np.array([1.0, 0.0], dtype=np.float32)],
    }
    speaker_order = ["GLOBAL_0"]

    mapping, debug_rows = diarization_module._assign_chunk_speakers_to_global(
        representative,
        global_bank,
        speaker_order,
    )

    assert mapping["chunk_speaker_0"] == "GLOBAL_0"
    assert mapping["chunk_speaker_1"] == "GLOBAL_1"
    assert any(row["assigned_global_speaker_id"] == "GLOBAL_0" for row in debug_rows)


def test_unrepresented_chunk_speakers_get_global_ids() -> None:
    """! @brief Test unrepresented chunk speakers get global ids.
    """
    mapping = {"chunk_speaker_0": "GLOBAL_0"}
    global_bank = {"GLOBAL_0": [np.array([1.0, 0.0], dtype=np.float32)]}
    speaker_order = ["GLOBAL_0"]
    debug_rows: list[dict[str, object]] = []

    diarization_module._assign_unrepresented_chunk_speakers_to_global(
        segments=[
            diarization_module.DiarizationSegment("chunk_speaker_0", 0.0, 1.0),
            diarization_module.DiarizationSegment("chunk_speaker_1", 1.0, 1.2),
        ],
        local_to_global=mapping,
        global_bank=global_bank,
        speaker_order=speaker_order,
        debug_rows=debug_rows,
    )

    assert mapping["chunk_speaker_1"] == "GLOBAL_1"
    assert "chunk_speaker_1" not in speaker_order
    assert speaker_order == ["GLOBAL_0", "GLOBAL_1"]
    assert debug_rows[0]["reason"] == "no_representative_embedding"


def test_diarize_auto_raises_when_pyannote_unavailable(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test diarize auto raises when pyannote unavailable.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    path = tmp_path / "diar_auto_embedding.wav"
    sf.write(path, audio, sample_rate)

    monkeypatch.setattr(diarization_module, "_diarize_with_pyannote", lambda **_: (None, "no_pipeline"))

    regions = [SpeechRegion(start_s=0.0, end_s=1.0)]
    with pytest.raises(RuntimeError) as exc_info:
        diarization_module.diarize(path, regions, max_speakers=2, max_chunk_s=0.5, backend="auto")

    assert "Pyannote diarization unavailable (no_pipeline)" in str(exc_info.value)


def test_diarize_pyannote_delegates_to_subprocess_helper(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test diarize pyannote delegates to subprocess helper.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    sample_rate = 16000
    path = tmp_path / "diar_subprocess.wav"
    sf.write(path, np.zeros(sample_rate, dtype=np.float32), sample_rate)

    monkeypatch.setattr(diarization_module, "_diarization_subprocess_enabled", lambda: True)
    monkeypatch.setattr(
        diarization_module,
        "_diarize_with_pyannote_subprocess",
        lambda **kwargs: (
            diarization_module.DiarizationResult(
                segments=[diarization_module.DiarizationSegment("SPEAKER_0", 0.0, 1.0)],
                speaker_count=1,
                mode="pyannote",
                details="subprocess",
            ),
            None,
        ),
    )

    result = diarization_module.diarize(
        path,
        [SpeechRegion(start_s=0.0, end_s=1.0)],
        backend="pyannote",
        model_path=tmp_path / "config.yaml",
        pipeline_path=str(tmp_path / "config.yaml"),
    )

    assert result.details == "subprocess"


def test_diarize_forced_embedding_raises_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test diarize forced embedding raises when unavailable.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    path = tmp_path / "diar_embedding_fallback.wav"
    sf.write(path, audio, sample_rate)

    monkeypatch.setattr(diarization_module, "_diarize_with_embeddings", lambda **_: (None, "embedding_unavailable"))

    regions = [SpeechRegion(start_s=0.0, end_s=1.0)]
    with pytest.raises(RuntimeError) as exc_info:
        diarization_module.diarize(path, regions, max_speakers=2, max_chunk_s=0.5, backend="embedding")

    assert "Embedding diarization unavailable (embedding_unavailable)" in str(exc_info.value)


def test_merge_transcript_segments_merges_adjacent() -> None:
    """! @brief Test merge transcript segments merges adjacent.
    """
    segments = [
        {"speaker_name": "Alice", "start_s": 0.0, "end_s": 1.0, "text": "Hello"},
        {"speaker_name": "Alice", "start_s": 1.2, "end_s": 2.0, "text": "there"},
        {"speaker_name": "Bob", "start_s": 3.0, "end_s": 4.0, "text": "Hi"},
    ]

    merged = merge_transcript_segments(segments, max_gap_s=0.3)

    assert len(merged) == 2
    assert merged[0]["text"] == "Hello there"


def test_merge_transcript_segments_default_merges_same_speaker_with_large_gap() -> None:
    """! @brief Test merge transcript segments default merges same speaker with large gap.
    """
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
    """! @brief Test estimate speaker count penalizes singleton heavy clusterings.
    @param monkeypatch Value for monkeypatch.
    """
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
    """! @brief Test estimate speaker count can pick higher k when clusters are stable.
    @param monkeypatch Value for monkeypatch.
    """
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


def test_run_pyannote_pipeline_retries_on_cpu_after_cuda_oom() -> None:
    """! @brief Test run pyannote pipeline retries on cpu after cuda oom.
    """
    class FakeTorch:
        class cuda:
            @staticmethod
            def empty_cache():
                """! @brief Empty cache.
                @return Result produced by the operation.
                """
                return None

        @staticmethod
        def device(name):
            """! @brief Device operation.
            @param name Value for name.
            @return Result produced by the operation.
            """
            return name

    class FakePipeline:
        def __init__(self):
            """! @brief Initialize the FakePipeline instance.
            @return Result produced by the operation.
            """
            self.current_device = "cpu"
            self.moves = []

        def to(self, device):
            """! @brief To operation.
            @param device Value for device.
            @return Result produced by the operation.
            """
            self.current_device = str(device)
            self.moves.append(self.current_device)

        def __call__(self, input_payload, **kwargs):
            """! @brief Call operation.
            @param input_payload Value for input payload.
            @param kwargs Value for kwargs.
            @return Result produced by the operation.
            """
            if self.current_device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return "ok"

    pipeline = FakePipeline()
    result, active_device = diarization_module._run_pyannote_pipeline(
        pipeline,
        {"waveform": np.zeros((1, 10), dtype=np.float32), "sample_rate": 16000},
        {},
        FakeTorch,
        target_device="cuda",
    )

    assert result == "ok"
    assert active_device == "cpu"
    assert pipeline.moves == ["cuda", "cpu"]
