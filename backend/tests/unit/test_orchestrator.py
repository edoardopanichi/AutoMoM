from __future__ import annotations

from pathlib import Path

from datetime import datetime

from backend.app.job_store import JOB_STORE, OpenAIJobConfig
from backend.app.schemas import JobSpeakerInfo, SpeakerMappingItem, SpeakerState, VoiceProfile, VoiceProfileEmbedding, VoiceProfileSample
from backend.pipeline.diarization import DiarizationResult, DiarizationSegment
from backend.pipeline.openai_client import OpenAIDiarizationResult, OpenAIDiarizedSegment
from backend.pipeline.orchestrator import PipelineOrchestrator


def test_collapse_labeled_segments_merges_ids_with_same_label() -> None:
    """! @brief Test collapse labeled segments merges ids with same label.
    """
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
    """! @brief Test collapse labeled segments keeps turn boundaries when not adjacent.
    """
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
    """! @brief Test pick openai audio source prefers supported original.
    @param tmp_path Value for tmp path.
    """
    original = tmp_path / "meeting.m4a"
    normalized = tmp_path / "meeting.wav"
    original.write_bytes(b"x" * 1024)
    normalized.write_bytes(b"y" * 2048)

    chosen = PipelineOrchestrator._pick_openai_audio_source(original, normalized)

    assert chosen == original


def test_transcript_segments_from_openai_diarization_applies_speaker_mapping() -> None:
    """! @brief Test transcript segments from openai diarization applies speaker mapping.
    """
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


def test_build_execution_summary_does_not_resolve_api_local_models(monkeypatch) -> None:
    """! @brief Test api execution summary avoids unused local model resolution.
    @param monkeypatch Value for monkeypatch.
    """
    runtime = type(
        "Runtime",
        (),
        {
            "api_config": OpenAIJobConfig(
                api_key="",
                diarization_execution="api",
                transcription_execution="api",
                formatter_execution="api",
                diarization_model="gpt-diarize",
                transcription_model="gpt-transcribe",
                formatter_model="gpt-format",
            ),
            "local_diarization_model_id": None,
            "local_transcription_model_id": None,
            "local_formatter_model_id": None,
        },
    )()
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.resolve_local_diarization_model",
        lambda model_id: (_ for _ in ()).throw(AssertionError("resolved diarization")),
    )
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.LOCAL_MODEL_CATALOG.resolve_model",
        lambda stage, model_id: (_ for _ in ()).throw(AssertionError(f"resolved {stage}")),
    )

    summary = PipelineOrchestrator._build_execution_summary(runtime, None, None)

    assert summary["diarization"]["model"] == "gpt-diarize"
    assert summary["transcription"]["model"] == "gpt-transcribe"
    assert summary["formatter"]["model"] == "gpt-format"
    assert summary["formatter"]["backend"] == "openai"


def test_plan_openai_audio_chunks_respects_upload_budget(tmp_path: Path) -> None:
    """! @brief Test plan openai audio chunks respects upload budget.
    @param tmp_path Value for tmp path.
    """
    audio = tmp_path / "long.wav"
    with audio.open("wb") as handle:
        handle.truncate(60 * 1024 * 1024)

    chunks = PipelineOrchestrator._plan_openai_audio_chunks(
        audio,
        total_duration_s=3600.0,
        span_start_s=0.0,
        span_end_s=3600.0,
        overlap_s=2.0,
    )

    assert len(chunks) > 1
    bytes_per_second = audio.stat().st_size / 3600.0
    assert all(
        (float(item["audio_end_s"]) - float(item["audio_start_s"])) * bytes_per_second < 25 * 1024 * 1024
        for item in chunks
    )


def test_globalize_openai_chunk_segments_uses_overlap_for_speaker_stitching() -> None:
    """! @brief Test globalize openai chunk segments uses overlap for speaker stitching.
    """
    existing = [OpenAIDiarizedSegment("SPEAKER_0", 0.0, 10.0, "first")]
    owned, mapping, next_index = PipelineOrchestrator._globalize_openai_chunk_segments(
        [OpenAIDiarizedSegment("local_a", 0.0, 5.0, "continued")],
        {"audio_start_s": 9.0, "own_start_s": 10.0, "own_end_s": 20.0},
        existing_segments=existing,
        previous_mapping={},
        next_speaker_index=1,
    )

    assert mapping["local_a"] == "SPEAKER_0"
    assert next_index == 1
    assert owned[0].speaker_id == "SPEAKER_0"
    assert owned[0].start_s == 10.0


def test_run_job_marks_failed_when_summary_generation_fails(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    """! @brief Test run job marks failed when summary generation fails.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    source_audio = tmp_path / "meeting.wav"
    source_audio.write_bytes(b"audio")
    runtime = JOB_STORE.create_job(
        audio_path=source_audio,
        original_filename=source_audio.name,
        template_id="default",
        language_mode="auto",
        title="Summary Failure",
        local_diarization_model_id="pyannote-community-1",
        local_transcription_model_id="whispercpp-local",
        local_formatter_model_id="formatter-command-default",
    )
    orchestrator = PipelineOrchestrator()

    def fake_normalize(input_path: Path, output_path: Path, **kwargs):
        output_path.write_bytes(b"normalized")
        return {"path": str(output_path), "duration_s": 2.0, "sample_rate": 16000, "channels": 1}

    def fake_extract_segment(input_path: Path, output_path: Path, *args, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")

    class FakeTranscriber:
        def available(self) -> bool:
            return True

        def runtime_report(self) -> dict[str, object]:
            return {
                "requested_mode": "cpu",
                "available_mode": "cpu",
                "active_mode": "cpu",
                "binary_path": "fake",
                "model_path": "fake",
            }

        def runtime_summary(self) -> str:
            return "fake"

    class FakeFormatter:
        last_mode = "fake"
        last_stdout = ""
        last_stderr = ""
        last_raw_output = ""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def write_model_output_to_mom(self, *args, output_path: Path, **kwargs):
            output_path.write_text("mom", encoding="utf-8")
            return type(
                "FormatterResult",
                (),
                {
                    "structured": {},
                    "system_prompt": "system",
                    "user_prompt": "user",
                    "validation": {"valid": True},
                    "reduced_notes": None,
                },
            )()

    fake_model = type(
        "Model",
        (),
        {
            "runtime": "command",
            "name": "fake",
            "config": {"command_template": "fake", "model_path": "fake"},
            "validation_error": None,
            "model_id": "fake",
        },
    )()

    monkeypatch.setattr("backend.pipeline.orchestrator.validate_audio_input", lambda path: None)
    monkeypatch.setattr("backend.pipeline.orchestrator.normalize_audio", fake_normalize)
    monkeypatch.setattr("backend.pipeline.orchestrator.detect_speech_regions", lambda path: [])
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.resolve_local_diarization_model",
        lambda model_id: type(
            "DiarizationModel",
            (),
            {"pipeline_path": "fake", "model_id": "pyannote-community-1", "embedding_model_ref": "embed"},
        )(),
    )
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.diarize",
        lambda *args, **kwargs: DiarizationResult(
            segments=[DiarizationSegment("SPEAKER_0", 0.0, 1.0)],
            speaker_count=1,
        ),
    )
    monkeypatch.setattr("backend.pipeline.orchestrator.pick_snippet_ranges", lambda *args, **kwargs: [])
    monkeypatch.setattr("backend.pipeline.orchestrator.extract_snippets", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_build_speaker_info",
        lambda *args, **kwargs: JobSpeakerInfo(
            detected_speakers=1,
            speakers=[SpeakerState(speaker_id="SPEAKER_0")],
        ),
    )
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.JOB_STORE.wait_for_mapping",
        lambda job_id: [SpeakerMappingItem(speaker_id="SPEAKER_0", name="Alice")],
    )
    monkeypatch.setattr("backend.pipeline.orchestrator.extract_segment", fake_extract_segment)
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.transcribe_segments",
        lambda *args, **kwargs: [
            {"speaker_id": "SPEAKER_0", "speaker_name": "Alice", "start_s": 0.0, "end_s": 1.0, "text": "hello"}
        ],
    )
    monkeypatch.setattr("backend.pipeline.orchestrator.LOCAL_MODEL_CATALOG.resolve_model", lambda *args: fake_model)
    monkeypatch.setattr(PipelineOrchestrator, "_build_local_transcriber", lambda *args: FakeTranscriber())
    monkeypatch.setattr("backend.pipeline.orchestrator.Formatter", FakeFormatter)
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_write_job_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("summary exploded")),
    )

    orchestrator._run_job(runtime.state.job_id)

    state = JOB_STORE.get_state(runtime.state.job_id)
    assert state.status == "failed"
    assert state.error == "summary exploded"


def test_plan_transcription_chunks_merges_short_same_speaker_gaps() -> None:
    """! @brief Test plan transcription chunks merges short same speaker gaps.
    """
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
    """! @brief Test plan transcription chunks respects max chunk duration.
    """
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
    """! @brief Test select profile segments prefers longer segments.
    """
    selected = PipelineOrchestrator._select_profile_segments(
        [(0.0, 0.9), (1.0, 5.5), (6.0, 8.0), (8.5, 12.5)],
        max_segments=2,
        min_duration_s=1.5,
    )

    assert selected == [(1.0, 5.5), (8.5, 12.5)]


def test_build_speaker_info_includes_profile_match_metadata(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test bUIld speaker info includes profile match metadata.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
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
