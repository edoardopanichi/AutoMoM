from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend.pipeline.transcription import VoxtralTranscriber, transcribe_segments


def test_voxtral_invocation_wrapper_uses_subprocess(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "voxtral"
    model = tmp_path / "model.gguf"
    segment = tmp_path / "segment.wav"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    segment.write_text("wav", encoding="utf-8")

    def fake_run(command, capture_output, text):
        assert str(binary) in command
        assert str(model) in command
        return SimpleNamespace(returncode=0, stdout="hello world", stderr="")

    monkeypatch.setattr("backend.pipeline.transcription.subprocess.run", fake_run)

    transcriber = VoxtralTranscriber(str(binary), str(model))
    text = transcriber.transcribe(segment)

    assert text == "hello world"


def test_transcribe_segments_reports_progress(tmp_path: Path) -> None:
    segment = tmp_path / "segment.wav"
    segment.write_text("wav", encoding="utf-8")

    transcriber = VoxtralTranscriber(binary_path="", model_path="")
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
