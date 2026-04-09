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
    """! @brief Write tone.
    @param path Filesystem path used by the operation.
    @param frequency Value for frequency.
    @param duration_s Value for duration s.
    """
    sample_rate = 16000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_end_to_end_job_with_golden_outputs(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    """! @brief Test end to end job with golden outputs.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    TemplateManager()

    source_audio = tmp_path / "meeting.wav"
    _write_tone(source_audio)

    runtime = JOB_STORE.create_job(
        audio_path=source_audio,
        original_filename=source_audio.name,
        template_id="default",
        language_mode="auto",
        title="Integration Test Meeting",
        local_diarization_model_id="pyannote-community-1",
    )
    job_id = runtime.state.job_id

    def fake_normalize(input_path: Path, output_path: Path, ffmpeg_bin: str, job_id=None):
        """! @brief Fake normalize.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        data, sr = sf.read(str(input_path), always_2d=False)
        sf.write(output_path, data, sr)
        return {"path": str(output_path), "duration_s": len(data) / sr, "sample_rate": sr, "channels": 1}

    def fake_extract_segment(input_path: Path, output_path: Path, start_s: float, end_s: float, ffmpeg_bin: str, job_id=None):
        """! @brief Fake extract segment.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param start_s Value for start s.
        @param end_s Value for end s.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        sample_rate = 16000
        duration = max(0.2, end_s - start_s)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(output_path, audio, sample_rate)

    def fake_diarize(audio_path: Path, speech_regions: Any, **kwargs):
        """! @brief Fake diarize.
        @param audio_path Path to the audio file.
        @param speech_regions Detected speech regions used as input.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        return DiarizationResult(
            segments=[
                DiarizationSegment("SPEAKER_0", 0.0, 5.0),
                DiarizationSegment("SPEAKER_1", 5.2, 10.0),
            ],
            speaker_count=2,
        )

    def fake_transcribe(transcriber, segment_jobs, progress_callback=None):
        """! @brief Fake transcribe.
        @param transcriber Value for transcriber.
        @param segment_jobs Value for segment jobs.
        @param progress_callback Optional callback invoked with progress updates.
        @return Result produced by the operation.
        """
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

    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("mock-model", encoding="utf-8")
    raw_formatter_output = (
        "### Title: Minutes of Meeting - Integration Test Meeting\n"
        "#### Participants:\n"
        "- Alice\n"
        "- Bob\n"
        "#### Concise Overview:\n"
        "Customer onboarding was prioritized.\n"
        "#### TODO's:\n"
        "- Alice to share the rollout plan.\n"
        "#### CONCLUSIONS:\n"
        "- Prioritize customer onboarding\n"
        "#### DECISION/OPEN POINTS:\n"
        "None\n"
        "#### RISKS:\n"
        "None\n"
    )
    script = f"import sys; sys.stdin.read(); sys.stdout.write({raw_formatter_output!r})"
    asr_binary = tmp_path / "whisper-cli"
    asr_model = tmp_path / "voxtral.gguf"
    asr_binary.write_text("bin", encoding="utf-8")
    asr_model.write_text("model", encoding="utf-8")
    old_backend = SETTINGS.formatter_backend
    old_command = SETTINGS.formatter_command
    old_model_path = SETTINGS.formatter_model_path
    old_voxtral_bin = SETTINGS.voxtral_binary
    old_voxtral_model_path = SETTINGS.voxtral_model_path
    old_voxtral_threads = SETTINGS.voxtral_threads
    old_voxtral_processors = SETTINGS.voxtral_processors
    old_keep_segment_audio = SETTINGS.transcription_keep_segment_audio
    try:
        object.__setattr__(SETTINGS, "formatter_backend", "command")
        object.__setattr__(SETTINGS, "formatter_command", f"python -c {shlex.quote(script)}")
        object.__setattr__(SETTINGS, "formatter_model_path", str(model_path))
        object.__setattr__(SETTINGS, "voxtral_binary", str(asr_binary))
        object.__setattr__(SETTINGS, "voxtral_model_path", str(asr_model))
        object.__setattr__(SETTINGS, "voxtral_threads", 2)
        object.__setattr__(SETTINGS, "voxtral_processors", 1)
        object.__setattr__(SETTINGS, "transcription_keep_segment_audio", False)
        ORCHESTRATOR._run_job(job_id)
    finally:
        object.__setattr__(SETTINGS, "formatter_backend", old_backend)
        object.__setattr__(SETTINGS, "formatter_command", old_command)
        object.__setattr__(SETTINGS, "formatter_model_path", old_model_path)
        object.__setattr__(SETTINGS, "voxtral_binary", old_voxtral_bin)
        object.__setattr__(SETTINGS, "voxtral_model_path", old_voxtral_model_path)
        object.__setattr__(SETTINGS, "voxtral_threads", old_voxtral_threads)
        object.__setattr__(SETTINGS, "voxtral_processors", old_voxtral_processors)
        object.__setattr__(SETTINGS, "transcription_keep_segment_audio", old_keep_segment_audio)

    state = JOB_STORE.get_state(job_id)
    assert state.status == "completed"

    transcript_path = Path(state.artifact_paths["transcript"])
    mom_path = Path(state.artifact_paths["mom_markdown"])
    runtime_path = Path(state.artifact_paths["transcription_runtime"])
    summary_path = Path(state.artifact_paths["job_summary"])

    generated_transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    expected_transcript = json.loads(
        (Path(__file__).parent / "golden" / "expected_transcript.json").read_text(encoding="utf-8")
    )
    assert generated_transcript == expected_transcript

    assert mom_path.read_text(encoding="utf-8").rstrip() == raw_formatter_output.rstrip()
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert runtime_payload["thread_count"] == 2
    assert runtime_payload["active_mode"] == "cpu"
    assert summary_payload["meeting_title"] == "Integration Test Meeting"
    assert summary_payload["template_id"] == "default"
    assert summary_payload["audio"]["original_filename"] == "meeting.wav"
    assert summary_payload["speakers"]["count"] == 2
    assert summary_payload["execution"]["transcription"]["compute_active"] == "cpu"
    assert "transcription" in summary_payload["timings"]["stages"]
    assert not any((transcript_path.parent / "transcription_segments").glob("*.wav"))


def test_end_to_end_passthrough_uses_raw_formatter_output(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    """! @brief Test end to end passthrough uses raw formatter output.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    TemplateManager()

    source_audio = tmp_path / "meeting.wav"
    _write_tone(source_audio, duration_s=8.0)

    runtime = JOB_STORE.create_job(
        audio_path=source_audio,
        original_filename=source_audio.name,
        template_id="default",
        language_mode="auto",
        title="Passthrough Meeting",
        local_diarization_model_id="pyannote-community-1",
    )
    job_id = runtime.state.job_id

    def fake_normalize(input_path: Path, output_path: Path, ffmpeg_bin: str, job_id=None):
        """! @brief Fake normalize.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        data, sr = sf.read(str(input_path), always_2d=False)
        sf.write(output_path, data, sr)
        return {"path": str(output_path), "duration_s": len(data) / sr, "sample_rate": sr, "channels": 1}

    def fake_extract_segment(input_path: Path, output_path: Path, start_s: float, end_s: float, ffmpeg_bin: str, job_id=None):
        """! @brief Fake extract segment.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param start_s Value for start s.
        @param end_s Value for end s.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        sample_rate = 16000
        duration = max(0.2, end_s - start_s)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(output_path, audio, sample_rate)

    def fake_diarize(audio_path: Path, speech_regions: Any, **kwargs):
        """! @brief Fake diarize.
        @param audio_path Path to the audio file.
        @param speech_regions Detected speech regions used as input.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        return DiarizationResult(
            segments=[
                DiarizationSegment("SPEAKER_0", 0.0, 3.5),
                DiarizationSegment("SPEAKER_1", 3.7, 7.0),
            ],
            speaker_count=2,
        )

    def fake_transcribe(transcriber, segment_jobs, progress_callback=None):
        """! @brief Fake transcribe.
        @param transcriber Value for transcriber.
        @param segment_jobs Value for segment jobs.
        @param progress_callback Optional callback invoked with progress updates.
        @return Result produced by the operation.
        """
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
        "### Title: Minutes of Meeting - Passthrough Meeting\n"
        "#### Participants:\n"
        "- MAN\n"
        "- WOMAN\n"
        "- Mayor Terry\n"
        "- Councillor Kaczynski\n"
        "#### Concise Overview:\n"
        "The meeting reviewed agenda items and follow-up references.\n"
        "#### TODO's:\n"
        "None\n"
        "#### CONCLUSIONS:\n"
        "- Postpone development agreement\n"
        "#### DECISION/OPEN POINTS:\n"
        "- Remove item 6.3 from today's meeting agenda.\n"
        "#### RISKS:\n"
        "None\n"
    )
    script = f"import sys; sys.stdin.read(); sys.stdout.write({raw_formatter_output!r})"
    asr_binary = tmp_path / "whisper-cli"
    asr_model = tmp_path / "voxtral.gguf"
    asr_binary.write_text("bin", encoding="utf-8")
    asr_model.write_text("model", encoding="utf-8")

    old_backend = SETTINGS.formatter_backend
    old_command = SETTINGS.formatter_command
    old_model_path = SETTINGS.formatter_model_path
    old_voxtral_bin = SETTINGS.voxtral_binary
    old_voxtral_model_path = SETTINGS.voxtral_model_path
    try:
        object.__setattr__(SETTINGS, "formatter_backend", "command")
        object.__setattr__(SETTINGS, "formatter_command", f"python -c {shlex.quote(script)}")
        object.__setattr__(SETTINGS, "formatter_model_path", str(model_path))
        object.__setattr__(SETTINGS, "voxtral_binary", str(asr_binary))
        object.__setattr__(SETTINGS, "voxtral_model_path", str(asr_model))
        ORCHESTRATOR._run_job(job_id)
    finally:
        object.__setattr__(SETTINGS, "formatter_backend", old_backend)
        object.__setattr__(SETTINGS, "formatter_command", old_command)
        object.__setattr__(SETTINGS, "formatter_model_path", old_model_path)
        object.__setattr__(SETTINGS, "voxtral_binary", old_voxtral_bin)
        object.__setattr__(SETTINGS, "voxtral_model_path", old_voxtral_model_path)

    state = JOB_STORE.get_state(job_id)
    assert state.status == "completed"

    mom_path = Path(state.artifact_paths["mom_markdown"])
    assert mom_path.read_text(encoding="utf-8").rstrip() == raw_formatter_output.rstrip()
    raw_output_path = Path(state.artifact_paths["formatter_raw_output"])
    assert raw_output_path.read_text(encoding="utf-8").rstrip() == raw_formatter_output.rstrip()
    stdout_path = Path(state.artifact_paths["formatter_stdout"])
    assert stdout_path.read_text(encoding="utf-8").rstrip() == raw_formatter_output.rstrip()


def test_end_to_end_stderr_prefixed_output_passthrough(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    """! @brief Test end to end stderr prefixed output passthrough.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    TemplateManager()

    source_audio = tmp_path / "meeting.wav"
    _write_tone(source_audio, duration_s=8.0)

    runtime = JOB_STORE.create_job(
        audio_path=source_audio,
        original_filename=source_audio.name,
        template_id="default",
        language_mode="auto",
        title="Stderr Passthrough Meeting",
        local_diarization_model_id="pyannote-community-1",
    )
    job_id = runtime.state.job_id

    def fake_normalize(input_path: Path, output_path: Path, ffmpeg_bin: str, job_id=None):
        """! @brief Fake normalize.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        data, sr = sf.read(str(input_path), always_2d=False)
        sf.write(output_path, data, sr)
        return {"path": str(output_path), "duration_s": len(data) / sr, "sample_rate": sr, "channels": 1}

    def fake_extract_segment(input_path: Path, output_path: Path, start_s: float, end_s: float, ffmpeg_bin: str, job_id=None):
        """! @brief Fake extract segment.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param start_s Value for start s.
        @param end_s Value for end s.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        sample_rate = 16000
        duration = max(0.2, end_s - start_s)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(output_path, audio, sample_rate)

    def fake_diarize(audio_path: Path, speech_regions: Any, **kwargs):
        """! @brief Fake diarize.
        @param audio_path Path to the audio file.
        @param speech_regions Detected speech regions used as input.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        return DiarizationResult(
            segments=[
                DiarizationSegment("SPEAKER_0", 0.0, 3.5),
                DiarizationSegment("SPEAKER_1", 3.7, 7.0),
            ],
            speaker_count=2,
        )

    def fake_transcribe(transcriber, segment_jobs, progress_callback=None):
        """! @brief Fake transcribe.
        @param transcriber Value for transcriber.
        @param segment_jobs Value for segment jobs.
        @param progress_callback Optional callback invoked with progress updates.
        @return Result produced by the operation.
        """
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
    raw_stderr_output = (
        "main: ### Title: Minutes of Meeting - Stderr Passthrough Meeting\n"
        "main: #### Participants:\n"
        "main: - MAN\n"
        "main: - WOMAN\n"
        "main: - Mayor Terry\n"
        "main: - Councillor Kaczynski\n"
        "main: #### Concise Overview:\n"
        "main: The meeting reviewed agenda items and follow-up references.\n"
        "main: #### TODO's:\n"
        "main: None\n"
        "main: #### CONCLUSIONS:\n"
        "main: - Postpone development agreement\n"
        "main: #### DECISION/OPEN POINTS:\n"
        "main: - Remove item 6.3\n"
        "main: #### RISKS:\n"
        "main: None\n"
    )
    script = f"import sys; sys.stdin.read(); sys.stderr.write({raw_stderr_output!r})"
    asr_binary = tmp_path / "whisper-cli"
    asr_model = tmp_path / "voxtral.gguf"
    asr_binary.write_text("bin", encoding="utf-8")
    asr_model.write_text("model", encoding="utf-8")

    old_backend = SETTINGS.formatter_backend
    old_command = SETTINGS.formatter_command
    old_model_path = SETTINGS.formatter_model_path
    old_voxtral_bin = SETTINGS.voxtral_binary
    old_voxtral_model_path = SETTINGS.voxtral_model_path
    try:
        object.__setattr__(SETTINGS, "formatter_backend", "command")
        object.__setattr__(SETTINGS, "formatter_command", f"python -c {shlex.quote(script)}")
        object.__setattr__(SETTINGS, "formatter_model_path", str(model_path))
        object.__setattr__(SETTINGS, "voxtral_binary", str(asr_binary))
        object.__setattr__(SETTINGS, "voxtral_model_path", str(asr_model))
        ORCHESTRATOR._run_job(job_id)
    finally:
        object.__setattr__(SETTINGS, "formatter_backend", old_backend)
        object.__setattr__(SETTINGS, "formatter_command", old_command)
        object.__setattr__(SETTINGS, "formatter_model_path", old_model_path)
        object.__setattr__(SETTINGS, "voxtral_binary", old_voxtral_bin)
        object.__setattr__(SETTINGS, "voxtral_model_path", old_voxtral_model_path)

    state = JOB_STORE.get_state(job_id)
    assert state.status == "completed"

    normalized_stderr_output = raw_stderr_output.replace("main: ", "")
    mom_path = Path(state.artifact_paths["mom_markdown"])
    assert mom_path.read_text(encoding="utf-8").rstrip() == normalized_stderr_output.rstrip()
    raw_output_path = Path(state.artifact_paths["formatter_raw_output"])
    assert raw_output_path.read_text(encoding="utf-8").rstrip() == normalized_stderr_output.rstrip()
    stderr_path = Path(state.artifact_paths["formatter_stderr"])
    assert stderr_path.read_text(encoding="utf-8") == raw_stderr_output


def test_end_to_end_nonzero_formatter_exit_with_stdout_still_passthrough(
    isolated_settings, monkeypatch, tmp_path: Path
) -> None:
    """! @brief Test end to end nonzero formatter exit with stdout still passthrough.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    TemplateManager()

    source_audio = tmp_path / "meeting.wav"
    _write_tone(source_audio, duration_s=8.0)

    runtime = JOB_STORE.create_job(
        audio_path=source_audio,
        original_filename=source_audio.name,
        template_id="default",
        language_mode="auto",
        title="Nonzero Passthrough Meeting",
        local_diarization_model_id="pyannote-community-1",
    )
    job_id = runtime.state.job_id

    def fake_normalize(input_path: Path, output_path: Path, ffmpeg_bin: str, job_id=None):
        """! @brief Fake normalize.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        data, sr = sf.read(str(input_path), always_2d=False)
        sf.write(output_path, data, sr)
        return {"path": str(output_path), "duration_s": len(data) / sr, "sample_rate": sr, "channels": 1}

    def fake_extract_segment(input_path: Path, output_path: Path, start_s: float, end_s: float, ffmpeg_bin: str, job_id=None):
        """! @brief Fake extract segment.
        @param input_path Path to the input file.
        @param output_path Path to the output file.
        @param start_s Value for start s.
        @param end_s Value for end s.
        @param ffmpeg_bin Value for ffmpeg bin.
        @param job_id Identifier of the job being processed.
        @return Result produced by the operation.
        """
        sample_rate = 16000
        duration = max(0.2, end_s - start_s)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(output_path, audio, sample_rate)

    def fake_diarize(audio_path: Path, speech_regions: Any, **kwargs):
        """! @brief Fake diarize.
        @param audio_path Path to the audio file.
        @param speech_regions Detected speech regions used as input.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        return DiarizationResult(
            segments=[
                DiarizationSegment("SPEAKER_0", 0.0, 3.5),
                DiarizationSegment("SPEAKER_1", 3.7, 7.0),
            ],
            speaker_count=2,
        )

    def fake_transcribe(transcriber, segment_jobs, progress_callback=None):
        """! @brief Fake transcribe.
        @param transcriber Value for transcriber.
        @param segment_jobs Value for segment jobs.
        @param progress_callback Optional callback invoked with progress updates.
        @return Result produced by the operation.
        """
        if progress_callback:
            progress_callback(1, 2)
            progress_callback(2, 2)
        return [
            {
                "speaker_id": "SPEAKER_0",
                "speaker_name": "MAN",
                "start_s": 0.0,
                "end_s": 3.5,
                "text": "Agenda updates reviewed.",
            },
            {
                "speaker_id": "SPEAKER_1",
                "speaker_name": "WOMAN",
                "start_s": 3.7,
                "end_s": 7.0,
                "text": "Decision reached.",
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
    raw_stdout_output = (
        "### Title: Minutes of Meeting - Nonzero Passthrough Meeting\n"
        "#### Participants:\n"
        "- MAN\n"
        "- WOMAN\n"
        "#### Concise Overview:\n"
        "The meeting confirmed the rollout plan.\n"
        "#### TODO's:\n"
        "None\n"
        "#### CONCLUSIONS:\n"
        "- Proceed with rollout\n"
        "#### DECISION/OPEN POINTS:\n"
        "- Keep item 6.3\n"
        "#### RISKS:\n"
        "None\n"
    )
    script = (
        "import sys; "
        f"sys.stdin.read(); sys.stdout.write({raw_stdout_output!r}); "
        "sys.stderr.write('error: simulated nonzero exit\\n'); "
        "sys.exit(1)"
    )
    asr_binary = tmp_path / "whisper-cli"
    asr_model = tmp_path / "voxtral.gguf"
    asr_binary.write_text("bin", encoding="utf-8")
    asr_model.write_text("model", encoding="utf-8")

    old_backend = SETTINGS.formatter_backend
    old_command = SETTINGS.formatter_command
    old_model_path = SETTINGS.formatter_model_path
    old_voxtral_bin = SETTINGS.voxtral_binary
    old_voxtral_model_path = SETTINGS.voxtral_model_path
    try:
        object.__setattr__(SETTINGS, "formatter_backend", "command")
        object.__setattr__(SETTINGS, "formatter_command", f"python -c {shlex.quote(script)}")
        object.__setattr__(SETTINGS, "formatter_model_path", str(model_path))
        object.__setattr__(SETTINGS, "voxtral_binary", str(asr_binary))
        object.__setattr__(SETTINGS, "voxtral_model_path", str(asr_model))
        ORCHESTRATOR._run_job(job_id)
    finally:
        object.__setattr__(SETTINGS, "formatter_backend", old_backend)
        object.__setattr__(SETTINGS, "formatter_command", old_command)
        object.__setattr__(SETTINGS, "formatter_model_path", old_model_path)
        object.__setattr__(SETTINGS, "voxtral_binary", old_voxtral_bin)
        object.__setattr__(SETTINGS, "voxtral_model_path", old_voxtral_model_path)

    state = JOB_STORE.get_state(job_id)
    assert state.status == "completed"
    mom_path = Path(state.artifact_paths["mom_markdown"])
    assert mom_path.read_text(encoding="utf-8").rstrip() == raw_stdout_output.rstrip()
