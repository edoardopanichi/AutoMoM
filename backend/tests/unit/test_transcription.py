from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.pipeline.transcription import (
    ASRBinaryCapabilities,
    TranscriptionError,
    VoxtralTranscriber,
    _gpu_verified_active,
    clean_transcript_text,
    transcribe_segments,
)


def test_voxtral_invocation_wrapper_uses_subprocess(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.gguf"
    segment = tmp_path / "segment.wav"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    segment.write_text("wav", encoding="utf-8")

    def fake_run(command, capture_output, text):
        assert str(binary) in command
        assert str(model) in command
        assert "-t" in command
        assert "-p" in command
        return SimpleNamespace(returncode=0, stdout="hello world", stderr="")

    monkeypatch.setattr("backend.pipeline.transcription.should_enable_native_gpu", lambda *_: False)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda *_: ASRBinaryCapabilities(str(binary), True, ("-t", "-p"), ("cpu",), False),
    )
    monkeypatch.setattr("backend.pipeline.transcription._binary_supports_any_flag", lambda *_: True)
    monkeypatch.setattr("backend.pipeline.transcription.subprocess.run", fake_run)

    transcriber = VoxtralTranscriber(str(binary), str(model), threads=8, processors=2)
    text = transcriber.transcribe(segment)

    assert text == "hello world"


def test_transcribe_segments_reports_progress(tmp_path: Path) -> None:
    segment = tmp_path / "segment.wav"
    segment.write_text("wav", encoding="utf-8")

    class StubTranscriber:
        @staticmethod
        def transcribe(_segment_path: Path) -> str:
            return "ok"

    transcriber = StubTranscriber()
    progress = []

    result = transcribe_segments(
        transcriber,
        [
            {
                "segment_path": str(segment),
                "speaker_id": "SPEAKER_0",
                "speaker_name": "Alice",
                "start_s": 0.0,
                "end_s": 2.0,
            }
        ],
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert len(result) == 1
    assert progress == [(1, 1)]


def test_voxtral_raises_when_runtime_unavailable(tmp_path: Path) -> None:
    segment = tmp_path / "segment.wav"
    segment.write_text("wav", encoding="utf-8")

    transcriber = VoxtralTranscriber(binary_path="", model_path="")
    with pytest.raises(TranscriptionError):
        transcriber.transcribe(segment)


def test_gpu_verified_active_parses_runtime_output() -> None:
    assert _gpu_verified_active("whisper_backend_init_gpu: device 0: NVIDIA GeForce RTX 3050") is True
    assert _gpu_verified_active("whisper_backend_init_gpu: device 0: CPU\nwhisper_backend_init_gpu: no GPU found") is False


def test_clean_transcript_text_removes_timestamp_noise() -> None:
    raw = """
    [00:00:00.000 --> 00:00:01.200] Hello everyone
    <|0.00|><|0.42|> this is a test
    [00:00:03] 01:02:03
    """

    cleaned = clean_transcript_text(raw)

    assert cleaned == "Hello everyone this is a test"


def test_transcribe_segments_merges_consecutive_same_speaker_without_gap_limit(tmp_path: Path) -> None:
    class StubTranscriber:
        def __init__(self) -> None:
            self._items = ["first part", "second part", "other speaker"]
            self._index = 0

        def transcribe(self, _segment_path: Path) -> str:
            text = self._items[self._index]
            self._index += 1
            return text

    progress = []
    transcriber = StubTranscriber()
    result = transcribe_segments(
        transcriber,  # type: ignore[arg-type]
        [
            {
                "segment_path": str(tmp_path / "s1.wav"),
                "speaker_id": "SPEAKER_0",
                "speaker_name": "Alice",
                "start_s": 0.0,
                "end_s": 1.0,
            },
            {
                "segment_path": str(tmp_path / "s2.wav"),
                "speaker_id": "SPEAKER_0",
                "speaker_name": "Alice",
                "start_s": 8.0,
                "end_s": 9.0,
            },
            {
                "segment_path": str(tmp_path / "s3.wav"),
                "speaker_id": "SPEAKER_1",
                "speaker_name": "Bob",
                "start_s": 9.2,
                "end_s": 10.0,
            },
        ],
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert len(result) == 2
    assert result[0]["speaker_name"] == "Alice"
    assert result[0]["text"] == "first part second part"
    assert result[1]["speaker_name"] == "Bob"


def test_voxtral_gpu_retry_falls_back_to_cpu(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.gguf"
    segment = tmp_path / "segment.wav"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    segment.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.should_enable_native_gpu", lambda *_: True)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda *_: ASRBinaryCapabilities(str(binary), True, ("-ngl", "-dev", "-t", "-p"), ("cuda",), True),
    )
    monkeypatch.setattr("backend.pipeline.transcription._binary_supports_any_flag", lambda *_: True)
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        if "-ngl" in command:
            return SimpleNamespace(returncode=1, stdout="", stderr="unsupported flag")
        return SimpleNamespace(returncode=0, stdout="ok text", stderr="")

    monkeypatch.setattr("backend.pipeline.transcription.subprocess.run", fake_run)

    transcriber = VoxtralTranscriber(str(binary), str(model), compute_device="auto", gpu_layers=99)
    text = transcriber.transcribe(segment)

    assert text == "ok text"
    assert any("-ngl" in call for call in calls)
    assert transcriber.compute_mode() == "cpu(gpu_retry_disabled)"
    assert transcriber.runtime_report()["gpu_verified_active"] is False


def test_voxtral_gpu_retry_handles_rc0_with_stderr_error(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.gguf"
    segment = tmp_path / "segment.wav"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    segment.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.should_enable_native_gpu", lambda *_: True)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda *_: ASRBinaryCapabilities(str(binary), True, ("-ngl", "-dev", "-t", "-p"), ("cuda",), True),
    )
    monkeypatch.setattr("backend.pipeline.transcription._binary_supports_any_flag", lambda *_: True)
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        if "-ngl" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="error: unknown argument: -ngl")
        return SimpleNamespace(returncode=0, stdout="[00:00:00.000 --> 00:00:01.000] hello", stderr="")

    monkeypatch.setattr("backend.pipeline.transcription.subprocess.run", fake_run)

    transcriber = VoxtralTranscriber(str(binary), str(model), compute_device="auto", gpu_layers=99)
    text = transcriber.transcribe(segment)

    assert text == "hello"
    assert any("-ngl" in call for call in calls)
    assert transcriber.compute_mode() == "cpu(gpu_retry_disabled)"


def test_voxtral_does_not_add_gpu_layers_when_binary_does_not_support_them(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.gguf"
    segment = tmp_path / "segment.wav"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    segment.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.should_enable_native_gpu", lambda *_: True)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda *_: ASRBinaryCapabilities(str(binary), True, ("-dev", "-t", "-p"), ("cuda",), True),
    )
    monkeypatch.setattr(
        "backend.pipeline.transcription._binary_supports_any_flag",
        lambda _binary, flags: ("-dev" in flags),
    )
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="hello", stderr="")

    monkeypatch.setattr("backend.pipeline.transcription.subprocess.run", fake_run)

    transcriber = VoxtralTranscriber(str(binary), str(model), compute_device="auto", cuda_device_id=1, gpu_layers=99)
    text = transcriber.transcribe(segment)

    assert text == "hello"
    assert calls
    assert "-ngl" not in calls[0]
    assert "-dev" in calls[0]


def test_voxtral_runtime_summary_reports_cpu_when_gpu_backend_unavailable(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.gguf"
    segment = tmp_path / "segment.wav"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    segment.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.should_enable_native_gpu", lambda *_: True)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda *_: ASRBinaryCapabilities(str(binary), True, ("-dev", "-t", "-p"), ("cpu",), False),
    )
    monkeypatch.setattr(
        "backend.pipeline.transcription.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="hello",
            stderr="whisper_backend_init_gpu: no GPU found",
        ),
    )

    transcriber = VoxtralTranscriber(str(binary), str(model), compute_device="auto")
    assert transcriber.transcribe(segment) == "hello"
    assert transcriber.compute_mode() == "cpu(gpu_backend_unavailable)"
    assert transcriber.runtime_summary() == "compute=cpu (GPU backend unavailable in ASR binary)"


def test_voxtral_runtime_summary_reports_verified_cuda(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.gguf"
    segment = tmp_path / "segment.wav"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    segment.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.should_enable_native_gpu", lambda *_: True)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda *_: ASRBinaryCapabilities(str(binary), True, ("-dev", "-t", "-p"), ("cuda",), True),
    )
    monkeypatch.setattr(
        "backend.pipeline.transcription.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="hello",
            stderr="whisper_backend_init_gpu: device 0: NVIDIA GeForce RTX 3050",
        ),
    )

    transcriber = VoxtralTranscriber(str(binary), str(model), compute_device="auto")
    assert transcriber.transcribe(segment) == "hello"
    assert transcriber.compute_mode() == "cuda"
    assert transcriber.runtime_summary() == "compute=cuda (verified active)"


def test_binary_selection_prefers_configured_gpu_capable_binary(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    configured = tmp_path / "configured-whisper-cli"
    repo_cuda = repo_root / "tools" / "whisper.cpp" / "build-cuda" / "bin" / "whisper-cli"
    repo_cpu = repo_root / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    for path in (configured, repo_cuda, repo_cpu):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bin", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda path: ASRBinaryCapabilities(path, True, tuple(), ("cuda",) if path == str(configured) else ("cpu",), path == str(configured)),
    )

    resolved = VoxtralTranscriber._resolve_preferred_binary_path(str(configured))

    assert resolved == str(configured)


def test_binary_selection_prefers_repo_cuda_when_configured_binary_is_cpu_only(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    configured = tmp_path / "configured-whisper-cli"
    repo_cuda = repo_root / "tools" / "whisper.cpp" / "build-cuda" / "bin" / "whisper-cli"
    repo_cpu = repo_root / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    for path in (configured, repo_cuda, repo_cpu):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bin", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.REPO_ROOT", repo_root)

    def fake_probe(path: str) -> ASRBinaryCapabilities:
        if path == str(repo_cuda):
            return ASRBinaryCapabilities(path, True, tuple(), ("cpu", "cuda"), True)
        return ASRBinaryCapabilities(path, True, tuple(), ("cpu",), False)

    monkeypatch.setattr("backend.pipeline.transcription._probe_asr_binary", fake_probe)

    resolved = VoxtralTranscriber._resolve_preferred_binary_path(str(configured))

    assert resolved == str(repo_cuda)


def test_binary_selection_falls_back_to_configured_cpu_binary_when_no_gpu_binary_exists(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    configured = tmp_path / "configured-whisper-cli"
    configured.parent.mkdir(parents=True, exist_ok=True)
    configured.write_text("bin", encoding="utf-8")

    monkeypatch.setattr("backend.pipeline.transcription.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "backend.pipeline.transcription._probe_asr_binary",
        lambda path: ASRBinaryCapabilities(path, True, tuple(), ("cpu",), False),
    )

    resolved = VoxtralTranscriber._resolve_preferred_binary_path(str(configured))

    assert resolved == str(configured)
