from __future__ import annotations

from pathlib import Path

from backend.pipeline.diarization import DiarizationSegment
from backend.pipeline.openai_client import OpenAIDiarizationResult, OpenAIDiarizedSegment
from backend.pipeline.orchestrator import PipelineOrchestrator


def test_collapse_labeled_segments_merges_ids_with_same_label() -> None:
    segments = [
        DiarizationSegment("SPEAKER_0", 0.0, 1.0),
        DiarizationSegment("SPEAKER_2", 1.2, 2.4),
        DiarizationSegment("SPEAKER_1", 2.8, 3.3),
    ]
    speaker_map = {
        "SPEAKER_0": "Alice",
        "SPEAKER_2": "Alice",
        "SPEAKER_1": "Bob",
    }

    collapsed = PipelineOrchestrator._collapse_labeled_segments(segments, speaker_map, max_gap_s=0.4)

    assert len(collapsed) == 2
    assert collapsed[0]["speaker_name"] == "Alice"
    assert collapsed[0]["start_s"] == 0.0
    assert collapsed[0]["end_s"] == 2.4
    assert collapsed[1]["speaker_name"] == "Bob"


def test_collapse_labeled_segments_keeps_turn_boundaries_when_not_adjacent() -> None:
    segments = [
        DiarizationSegment("SPEAKER_0", 0.0, 1.0),
        DiarizationSegment("SPEAKER_1", 1.1, 2.0),
        DiarizationSegment("SPEAKER_2", 2.2, 3.0),
    ]
    speaker_map = {
        "SPEAKER_0": "Alice",
        "SPEAKER_2": "Alice",
        "SPEAKER_1": "Bob",
    }

    collapsed = PipelineOrchestrator._collapse_labeled_segments(segments, speaker_map, max_gap_s=0.5)

    assert len(collapsed) == 3
    assert [item["speaker_name"] for item in collapsed] == ["Alice", "Bob", "Alice"]


def test_pick_openai_audio_source_prefers_supported_original(tmp_path: Path) -> None:
    original = tmp_path / "meeting.m4a"
    normalized = tmp_path / "meeting.wav"
    original.write_bytes(b"x" * 1024)
    normalized.write_bytes(b"y" * 2048)

    chosen = PipelineOrchestrator._pick_openai_audio_source(original, normalized)

    assert chosen == original


def test_transcript_segments_from_openai_diarization_applies_speaker_mapping() -> None:
    result = OpenAIDiarizationResult(
        text="hello",
        segments=[
            OpenAIDiarizedSegment("SPEAKER_0", 0.0, 1.0, "First"),
            OpenAIDiarizedSegment("SPEAKER_1", 1.0, 2.0, "Second"),
        ],
    )

    transcript = PipelineOrchestrator._transcript_segments_from_openai_diarization(
        result,
        {"SPEAKER_0": "Alice", "SPEAKER_1": "Bob"},
    )

    assert transcript == [
        {"speaker_id": "SPEAKER_0", "speaker_name": "Alice", "start_s": 0.0, "end_s": 1.0, "text": "First"},
        {"speaker_id": "SPEAKER_1", "speaker_name": "Bob", "start_s": 1.0, "end_s": 2.0, "text": "Second"},
    ]


def test_plan_transcription_chunks_merges_short_same_speaker_gaps() -> None:
    chunks = PipelineOrchestrator._plan_transcription_chunks(
        [
            {"speaker_id": "SPEAKER_0", "speaker_name": "Alice", "start_s": 0.0, "end_s": 4.0},
            {"speaker_id": "SPEAKER_0", "speaker_name": "Alice", "start_s": 4.8, "end_s": 9.0},
            {"speaker_id": "SPEAKER_1", "speaker_name": "Bob", "start_s": 9.5, "end_s": 12.0},
        ],
        max_gap_s=1.0,
        max_chunk_s=20.0,
    )

    assert len(chunks) == 2
    assert chunks[0]["start_s"] == 0.0
    assert chunks[0]["end_s"] == 9.0
    assert chunks[1]["speaker_name"] == "Bob"


def test_plan_transcription_chunks_respects_max_chunk_duration() -> None:
    chunks = PipelineOrchestrator._plan_transcription_chunks(
        [
            {"speaker_id": "SPEAKER_0", "speaker_name": "Alice", "start_s": 0.0, "end_s": 12.0},
            {"speaker_id": "SPEAKER_0", "speaker_name": "Alice", "start_s": 12.2, "end_s": 25.0},
        ],
        max_gap_s=1.0,
        max_chunk_s=20.0,
    )

    assert len(chunks) == 2
