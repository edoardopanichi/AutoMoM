from __future__ import annotations

from backend.pipeline.diarization import DiarizationSegment
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
