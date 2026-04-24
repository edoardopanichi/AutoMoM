from __future__ import annotations

from pathlib import Path

from backend.pipeline.diarization import DiarizationResult
from backend.pipeline.vad import SpeechRegion
from backend.worker import main as worker


def test_remote_diarize_uses_vad_signature(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test remote diarize uses VAD signature.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    uploaded_path = tmp_path / "uploaded.wav"
    normalized_path = tmp_path / "normalized.wav"
    uploaded_path.write_bytes(b"uploaded")
    normalized_path.write_bytes(b"normalized")
    calls: dict[str, object] = {}

    def fake_load_normalized_audio(upload):
        calls["upload_name"] = upload.filename
        return uploaded_path, normalized_path

    def fake_detect_speech_regions(audio_path: Path) -> list[SpeechRegion]:
        calls["vad_path"] = audio_path
        return [SpeechRegion(start_s=0.0, end_s=1.0)]

    def fake_diarize(audio_path: Path, speech_regions: list[SpeechRegion], **kwargs) -> DiarizationResult:
        calls["diarize_path"] = audio_path
        calls["speech_regions"] = speech_regions
        return DiarizationResult(segments=[], speaker_count=0)

    monkeypatch.setattr(worker, "_load_normalized_audio", fake_load_normalized_audio)
    monkeypatch.setattr(worker, "detect_speech_regions", fake_detect_speech_regions)
    monkeypatch.setattr(worker, "diarize", fake_diarize)
    monkeypatch.setattr(worker, "ENABLED_STAGES", {"diarization"})

    response = worker.diarize_audio(
        audio_file=type("Upload", (), {"filename": "meeting.wav"})(),
    )

    assert response["mode"] == "remote-pyannote"
    assert calls["upload_name"] == "meeting.wav"
    assert calls["vad_path"] == normalized_path
    assert calls["diarize_path"] == normalized_path
    assert calls["speech_regions"] == [SpeechRegion(start_s=0.0, end_s=1.0)]
