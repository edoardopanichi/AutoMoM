from __future__ import annotations

from pathlib import Path

from backend.app.job_store import JOB_STORE, JobStore


def test_job_id_uses_timestamp_and_title(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test job id uses timestamp and title.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    runtime = JOB_STORE.create_job(
        audio_path=audio,
        original_filename=audio.name,
        template_id="default",
        language_mode="auto",
        title="Weekly Sync",
        local_diarization_model_id="pyannote-community-1",
        local_transcription_model_id="whispercpp-local",
        local_formatter_model_id="formatter-ollama-default",
    )

    assert "-weekly_sync" in runtime.state.job_id
    assert (runtime.state.job_id[:16].count("-") == 3)  # YYYY-MM-DD-HH:MM


def test_job_id_falls_back_to_meeting_when_title_missing(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test job id falls back to meeting when title missing.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    runtime = JOB_STORE.create_job(
        audio_path=audio,
        original_filename=audio.name,
        template_id="default",
        language_mode="auto",
        title=None,
        local_diarization_model_id="pyannote-community-1",
        local_transcription_model_id="whispercpp-local",
        local_formatter_model_id="formatter-ollama-default",
    )

    assert runtime.state.job_id.endswith("-meeting")


def test_submit_speaker_mapping_requires_waiting_state(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test submit speaker mapping reqUIres waiting state.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    runtime = JOB_STORE.create_job(
        audio_path=audio,
        original_filename=audio.name,
        template_id="default",
        language_mode="auto",
        title="Weekly Sync",
        local_diarization_model_id="pyannote-community-1",
        local_transcription_model_id="whispercpp-local",
        local_formatter_model_id="formatter-ollama-default",
    )

    try:
        JOB_STORE.submit_speaker_mapping(runtime.state.job_id, [])
    except ValueError as exc:
        assert "not waiting for speaker input" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected submit_speaker_mapping to reject non-waiting jobs")


def test_job_store_rehydrates_completed_jobs(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test job store rehydrates completed jobs.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    runtime = JOB_STORE.create_job(
        audio_path=audio,
        original_filename=audio.name,
        template_id="default",
        language_mode="auto",
        title="Persisted",
        local_diarization_model_id="pyannote-community-1",
        local_transcription_model_id="whispercpp-local",
        local_formatter_model_id="formatter-ollama-default",
    )
    JOB_STORE.mark_completed(runtime.state.job_id)

    loaded = JobStore()

    state = loaded.get_state(runtime.state.job_id)
    assert state.status == "completed"
    assert loaded.get_runtime(runtime.state.job_id).audio_path == audio


def test_job_store_marks_non_terminal_rehydrated_jobs_failed(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test job store marks non-terminal rehydrated jobs failed.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    runtime = JOB_STORE.create_job(
        audio_path=audio,
        original_filename=audio.name,
        template_id="default",
        language_mode="auto",
        title="Interrupted",
        local_diarization_model_id="pyannote-community-1",
        local_transcription_model_id="whispercpp-local",
        local_formatter_model_id="formatter-ollama-default",
    )
    JOB_STORE.mark_running(runtime.state.job_id)

    loaded = JobStore()

    state = loaded.get_state(runtime.state.job_id)
    assert state.status == "failed"
    assert state.error == JobStore.INTERRUPTED_RESTART_ERROR


def test_delete_job_removes_terminal_artifacts_and_upload(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test delete job removes terminal artifacts and upload.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    from backend.app.config import SETTINGS

    audio = SETTINGS.uploads_dir / "a.wav"
    audio.write_bytes(b"x")
    runtime = JOB_STORE.create_job(
        audio_path=audio,
        original_filename=audio.name,
        template_id="default",
        language_mode="auto",
        title="Delete",
        local_diarization_model_id="pyannote-community-1",
        local_transcription_model_id="whispercpp-local",
        local_formatter_model_id="formatter-ollama-default",
    )
    job_dir = SETTINGS.jobs_dir / runtime.state.job_id
    JOB_STORE.mark_failed(runtime.state.job_id, "failed")

    JOB_STORE.delete_job(runtime.state.job_id)

    assert not job_dir.exists()
    assert not audio.exists()
