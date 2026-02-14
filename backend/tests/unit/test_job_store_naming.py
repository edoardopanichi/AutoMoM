from __future__ import annotations

from pathlib import Path

from backend.app.job_store import JOB_STORE


def test_job_id_uses_timestamp_and_title(isolated_settings, tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    runtime = JOB_STORE.create_job(
        audio_path=audio,
        template_id="default",
        language_mode="auto",
        title="Weekly Sync",
    )

    assert "-weekly_sync" in runtime.state.job_id
    assert (runtime.state.job_id[:16].count("-") == 3)  # YYYY-MM-DD-HH:MM


def test_job_id_falls_back_to_meeting_when_title_missing(isolated_settings, tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    runtime = JOB_STORE.create_job(
        audio_path=audio,
        template_id="default",
        language_mode="auto",
        title=None,
    )

    assert runtime.state.job_id.endswith("-meeting")
