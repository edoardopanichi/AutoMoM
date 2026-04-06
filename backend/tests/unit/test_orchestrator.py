from __future__ import annotations

from pathlib import Path

from datetime import datetime

from backend.app.schemas import VoiceProfile, VoiceProfileEmbedding, VoiceProfileSample
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


def test_select_profile_segments_prefers_longer_segments() -> None:
    selected = PipelineOrchestrator._select_profile_segments(
        [(0.0, 0.9), (1.0, 5.5), (6.0, 8.0), (8.5, 12.5)],
        max_segments=2,
        min_duration_s=1.5,
    )

    assert selected == [(1.0, 5.5), (8.5, 12.5)]


def test_build_speaker_info_includes_profile_match_metadata(monkeypatch, tmp_path: Path) -> None:
    orchestrator = PipelineOrchestrator()
    now = datetime.fromisoformat("2026-01-01T00:00:00+00:00")

    monkeypatch.setattr(
        "backend.pipeline.orchestrator.VOICE_PROFILE_MANAGER.list_profiles",
        lambda: [
            VoiceProfile(
                profile_id="p1",
                name="Alice",
                created_at=now,
                updated_at=now,
                sample_count=1,
                samples=[
                    VoiceProfileSample(
                        sample_id="s1",
                        created_at=now,
                        reference_audio_path=str(tmp_path / "sample.wav"),
                        embeddings=[
                            VoiceProfileEmbedding(
                                embedding_id="e1",
                                engine_kind="local_pyannote",
                                diarization_model_id="pyannote-community-1",
                                embedding_model_ref="local-embed-v1",
                                library_version="test",
                                threshold=0.82,
                                vector=[1.0, 0.0],
                                created_at=now,
                                model_key="local_pyannote::pyannote-community-1::local-embed-v1",
                            )
                        ],
                    )
                ],
            )
        ],
    )
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.VOICE_PROFILE_MANAGER.compute_embedding",
        lambda *args, **kwargs: [1.0] * 20,
    )
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.VOICE_PROFILE_MANAGER.match",
        lambda *args, **kwargs: type(
            "Match",
            (),
            {
                "best_match": type(
                    "Best",
                    (),
                    {
                        "profile_id": "p1",
                        "sample_id": "s1",
                        "name": "Alice",
                        "score": 0.93,
                        "model_key": "local_pyannote::pyannote-community-1::local-embed-v1",
                    },
                )(),
                "ambiguous_matches": [],
            },
        )(),
    )
    logs = []
    monkeypatch.setattr("backend.pipeline.orchestrator.JOB_STORE.append_log", lambda job_id, message: logs.append(message))
    runtime = type(
        "Runtime",
        (),
        {
            "api_config": None,
            "local_diarization_model_id": "pyannote-community-1",
        },
    )()

    info = orchestrator._build_speaker_info(
        runtime,
        "job-1",
        tmp_path / "audio.wav",
        {"SPEAKER_0": [(0.0, 4.0)]},
        [],
    )

    assert info.speakers[0].suggested_name == "Alice"
    assert info.speakers[0].matched_profile is not None
    assert info.speakers[0].matched_profile.sample_id == "s1"
    assert info.speakers[0].matched_profile.status == "matched"
    assert "Auto-identified SPEAKER_0 as Alice" in logs[0]
