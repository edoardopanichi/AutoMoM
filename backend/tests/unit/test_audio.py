from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from backend.pipeline import audio


def test_normalize_audio_invokes_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test normalize audio invokes ffmpeg.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    input_path = tmp_path / "input.mp3"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"fake")

    called = {}

    def fake_run(command, job_id=None, **kwargs):
        """! @brief Fake run.
        @param command Command arguments passed to the subprocess.
        @param job_id Identifier of the job being processed.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        called["command"] = command
        # create a valid output wav so metadata can be read
        samples = np.zeros(16000, dtype=np.float32)
        sf.write(output_path, samples, 16000)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio, "run_cancellable_subprocess", fake_run)

    metadata = audio.normalize_audio(input_path, output_path, ffmpeg_bin="ffmpeg")

    assert called["command"][0] == "ffmpeg"
    assert "-ac" in called["command"]
    assert "-af" in called["command"]
    denoise_idx = called["command"].index("-af")
    assert called["command"][denoise_idx + 1] == "afftdn"
    assert metadata["sample_rate"] == 16000
    assert metadata["channels"] == 1
    assert metadata["denoise_enabled"] == 1
    assert metadata["denoise_filter"] == "afftdn"


def test_normalize_audio_can_disable_denoise(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test normalize audio can disable denoise.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    input_path = tmp_path / "input.mp3"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"fake")

    called = {}

    def fake_run(command, job_id=None, **kwargs):
        """! @brief Fake run.
        @param command Command arguments passed to the subprocess.
        @param job_id Identifier of the job being processed.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        called["command"] = command
        samples = np.zeros(16000, dtype=np.float32)
        sf.write(output_path, samples, 16000)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio, "run_cancellable_subprocess", fake_run)

    metadata = audio.normalize_audio(input_path, output_path, ffmpeg_bin="ffmpeg", denoise_enabled=False)

    assert "-af" not in called["command"]
    assert metadata["denoise_enabled"] == 0
    assert metadata["denoise_filter"] == ""


def test_normalize_audio_uses_custom_denoise_filter(monkeypatch, tmp_path: Path) -> None:
    """! @brief Test normalize audio uses custom denoise filter.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    input_path = tmp_path / "input.mp3"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"fake")

    called = {}

    def fake_run(command, job_id=None, **kwargs):
        """! @brief Fake run.
        @param command Command arguments passed to the subprocess.
        @param job_id Identifier of the job being processed.
        @param kwargs Value for kwargs.
        @return Result produced by the operation.
        """
        called["command"] = command
        samples = np.zeros(16000, dtype=np.float32)
        sf.write(output_path, samples, 16000)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio, "run_cancellable_subprocess", fake_run)

    metadata = audio.normalize_audio(
        input_path,
        output_path,
        ffmpeg_bin="ffmpeg",
        denoise_filter="afftdn=nf=-28",
    )

    denoise_idx = called["command"].index("-af")
    assert called["command"][denoise_idx + 1] == "afftdn=nf=-28"
    assert metadata["denoise_filter"] == "afftdn=nf=-28"


def test_validate_audio_input_rejects_unknown_extension(tmp_path: Path) -> None:
    """! @brief Test validate audio input rejects unknown extension.
    @param tmp_path Value for tmp path.
    """
    bad_path = tmp_path / "meeting.txt"
    bad_path.write_text("not audio", encoding="utf-8")

    try:
        audio.validate_audio_input(bad_path)
    except audio.AudioError as exc:
        assert "Unsupported file format" in str(exc)
    else:
        raise AssertionError("Expected AudioError")


def test_validate_audio_input_accepts_aac(tmp_path: Path) -> None:
    """! @brief Test validate audio input accepts aac.
    @param tmp_path Value for tmp path.
    """
    aac_path = tmp_path / "meeting.aac"
    aac_path.write_bytes(b"fake")

    audio.validate_audio_input(aac_path)
