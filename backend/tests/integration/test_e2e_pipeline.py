from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from backend.app.config import SETTINGS
from backend.app.job_store import JOB_STORE
from backend.app.schemas import SpeakerMappingItem
from backend.pipeline.diarization import DiarizationResult, DiarizationSegment
from backend.pipeline.orchestrator import ORCHESTRATOR
from backend.pipeline.template_manager import TemplateManager


def _write_tone(path: Path, frequency: float = 220.0, duration_s: float = 12.0) -> None:
    sample_rate = 16000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_end_to_end_job_with_golden_outputs(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    # Ensure default template exists in isolated path
    TemplateManager()

    source_audio = tmp_path / "meeting.wav"
    _write_tone(source_audio)

    runtime = JOB_STORE.create_job(
        audio_path=source_audio,
        template_id="default",
        language_mode="auto",
        title="Integration Test Meeting",
    )
    job_id = runtime.state.job_id

    def fake_normalize(input_path: Path, output_path: Path, ffmpeg_bin: str):
        data, sr = sf.read(str(input_path), always_2d=False)
        sf.write(output_path, data, sr)
        return {"path": str(output_path), "duration_s": len(data) / sr, "sample_rate": sr, "channels": 1}

    def fake_extract_segment(input_path: Path, output_path: Path, start_s: float, end_s: float, ffmpeg_bin: str):
        sample_rate = 16000
        duration = max(0.2, end_s - start_s)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(output_path, audio, sample_rate)

    def fake_diarize(audio_path: Path, speech_regions: Any, **kwargs):
        return DiarizationResult(
            segments=[
                DiarizationSegment("SPEAKER_0", 0.0, 5.0),
                DiarizationSegment("SPEAKER_1", 5.2, 10.0),
            ],
            speaker_count=2,
        )

    def fake_transcribe(transcriber, segment_jobs, progress_callback=None):
        if progress_callback:
            progress_callback(1, 2)
            progress_callback(2, 2)
        return [
            {
                "speaker_id": "SPEAKER_0",
                "speaker_name": "Alice",
                "start_s": 0.0,
                "end_s": 5.0,
                "text": "We decided to prioritize customer onboarding.",
            },
            {
                "speaker_id": "SPEAKER_1",
                "speaker_name": "Bob",
                "start_s": 5.2,
                "end_s": 10.0,
                "text": "Alice will share the rollout plan by next week.",
            },
        ]

    monkeypatch.setattr("backend.pipeline.orchestrator.normalize_audio", fake_normalize)
    monkeypatch.setattr("backend.pipeline.orchestrator.extract_segment", fake_extract_segment)
    monkeypatch.setattr("backend.pipeline.snippets.extract_segment", fake_extract_segment)
    monkeypatch.setattr("backend.pipeline.orchestrator.diarize", fake_diarize)
    monkeypatch.setattr("backend.pipeline.orchestrator.transcribe_segments", fake_transcribe)
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.JOB_STORE.wait_for_mapping",
        lambda jid: [
            SpeakerMappingItem(speaker_id="SPEAKER_0", name="Alice", save_voice_profile=False),
            SpeakerMappingItem(speaker_id="SPEAKER_1", name="Bob", save_voice_profile=False),
        ],
    )

    ORCHESTRATOR._run_job(job_id)

    state = JOB_STORE.get_state(job_id)
    assert state.status == "completed"

    transcript_path = Path(state.artifact_paths["transcript"])
    mom_path = Path(state.artifact_paths["mom_markdown"])

    generated_transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    expected_transcript = json.loads(
        (Path(__file__).parent / "golden" / "expected_transcript.json").read_text(encoding="utf-8")
    )
    assert generated_transcript == expected_transcript

    mom_text = mom_path.read_text(encoding="utf-8")
    required_sections = (Path(__file__).parent / "golden" / "required_mom_sections.txt").read_text(encoding="utf-8").splitlines()
    for section in required_sections:
        assert section in mom_text


def test_end_to_end_passthrough_uses_raw_formatter_output(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    TemplateManager()

    source_audio = tmp_path / "meeting.wav"
    _write_tone(source_audio, duration_s=8.0)

    runtime = JOB_STORE.create_job(
        audio_path=source_audio,
        template_id="default",
        language_mode="auto",
        title="Passthrough Meeting",
    )
    job_id = runtime.state.job_id

    def fake_normalize(input_path: Path, output_path: Path, ffmpeg_bin: str):
        data, sr = sf.read(str(input_path), always_2d=False)
        sf.write(output_path, data, sr)
        return {"path": str(output_path), "duration_s": len(data) / sr, "sample_rate": sr, "channels": 1}

    def fake_extract_segment(input_path: Path, output_path: Path, start_s: float, end_s: float, ffmpeg_bin: str):
        sample_rate = 16000
        duration = max(0.2, end_s - start_s)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(output_path, audio, sample_rate)

    def fake_diarize(audio_path: Path, speech_regions: Any, **kwargs):
        return DiarizationResult(
            segments=[
                DiarizationSegment("SPEAKER_0", 0.0, 3.5),
                DiarizationSegment("SPEAKER_1", 3.7, 7.0),
            ],
            speaker_count=2,
        )

    def fake_transcribe(transcriber, segment_jobs, progress_callback=None):
        if progress_callback:
            progress_callback(1, 2)
            progress_callback(2, 2)
        return [
            {
                "speaker_id": "SPEAKER_0",
                "speaker_name": "MAN",
                "start_s": 0.0,
                "end_s": 3.5,
                "text": "We should remove item 6.3 from today's meeting agenda.",
            },
            {
                "speaker_id": "SPEAKER_1",
                "speaker_name": "WOMAN",
                "start_s": 3.7,
                "end_s": 7.0,
                "text": "Mayor Terry and Councillor Kaczynski were mentioned for follow-up.",
            },
        ]

    monkeypatch.setattr("backend.pipeline.orchestrator.normalize_audio", fake_normalize)
    monkeypatch.setattr("backend.pipeline.orchestrator.extract_segment", fake_extract_segment)
    monkeypatch.setattr("backend.pipeline.snippets.extract_segment", fake_extract_segment)
    monkeypatch.setattr("backend.pipeline.orchestrator.diarize", fake_diarize)
    monkeypatch.setattr("backend.pipeline.orchestrator.transcribe_segments", fake_transcribe)
    monkeypatch.setattr(
        "backend.pipeline.orchestrator.JOB_STORE.wait_for_mapping",
        lambda jid: [
            SpeakerMappingItem(speaker_id="SPEAKER_0", name="MAN", save_voice_profile=False),
            SpeakerMappingItem(speaker_id="SPEAKER_1", name="WOMAN", save_voice_profile=False),
        ],
    )

    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("mock-model", encoding="utf-8")
    raw_formatter_output = (
        "# Minutes of Meeting\n"
        "## Participants\n"
        "- MAN\n"
        "- WOMAN\n"
        "- Mayor Terry\n"
        "- Councillor Kaczynski\n"
        "## Agenda\n"
        "- Remove item 6.3\n"
        "## Decisions\n"
        "- Postpone development agreement\n"
    )
    script = f"import sys; sys.stdin.read(); sys.stdout.write({raw_formatter_output!r})"

    old_command = SETTINGS.formatter_command
    old_model_path = SETTINGS.formatter_model_path
    try:
        object.__setattr__(SETTINGS, "formatter_command", f"python -c {shlex.quote(script)}")
        object.__setattr__(SETTINGS, "formatter_model_path", str(model_path))
        ORCHESTRATOR._run_job(job_id)
    finally:
        object.__setattr__(SETTINGS, "formatter_command", old_command)
        object.__setattr__(SETTINGS, "formatter_model_path", old_model_path)

    state = JOB_STORE.get_state(job_id)
    assert state.status == "completed"

    mom_path = Path(state.artifact_paths["mom_markdown"])
    assert mom_path.read_text(encoding="utf-8") == raw_formatter_output
    raw_output_path = Path(state.artifact_paths["formatter_raw_output"])
    assert raw_output_path.read_text(encoding="utf-8") == raw_formatter_output
