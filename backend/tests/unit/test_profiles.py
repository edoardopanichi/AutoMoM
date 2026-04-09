from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import soundfile as sf

from backend.profiles.manager import VoiceProfileManager


def _tone(path: Path, freq: float, duration_s: float = 1.5) -> None:
    """! @brief Tone operation.
    @param path Filesystem path used by the operation.
    @param freq Value for freq.
    @param duration_s Value for duration s.
    """
    sample_rate = 16000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def _fake_embedding(audio_path: Path, *, model_ref: str, **_: object) -> np.ndarray:
    """! @brief Fake embedding.
    @param audio_path Path to the audio file.
    @param model_ref Value for model ref.
    @param _ Value for _.
    @return Result produced by the operation.
    """
    key = f"{audio_path.stem}:{model_ref}"
    if "alice" in key:
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif "bob" in key:
        base = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        base = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if "alt" in key:
        base = base + np.array([0.0, 0.0, 0.5], dtype=np.float32)
    return base / np.linalg.norm(base)


def test_save_profile_sample_creates_profile_directory_and_sample_audio(
    isolated_settings, monkeypatch, tmp_path: Path
) -> None:
    """! @brief Test save profile sample creates profile directory and sample audio.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    _tone(tmp_path / "alice.wav", 220.0, duration_s=4.0)
    monkeypatch.setattr("backend.profiles.manager.compute_profile_embedding", _fake_embedding)

    manager = VoiceProfileManager()
    profile = manager.save_profile_sample(
        name="Alice Smith",
        source_audio_path=tmp_path / "alice.wav",
        clip_ranges=[(0.0, 1.2), (2.0, 3.0)],
        diarization_model_id="pyannote-community-1",
        embedding_model_ref="local-embed-v1",
    )

    assert profile.sample_count == 1
    assert profile.samples[0].reference_audio_path.endswith(".wav")
    assert Path(profile.samples[0].reference_audio_path).exists()
    assert profile.samples[0].embeddings[0].diarization_model_id == "pyannote-community-1"
    assert profile.samples[0].embeddings[0].embedding_model_ref == "local-embed-v1"


def test_same_name_appends_new_sample_instead_of_overwriting(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    """! @brief Test same name appends new sample instead of overwriting.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    _tone(tmp_path / "alice_a.wav", 220.0, duration_s=4.0)
    _tone(tmp_path / "alice_b.wav", 240.0, duration_s=4.0)
    monkeypatch.setattr("backend.profiles.manager.compute_profile_embedding", _fake_embedding)

    manager = VoiceProfileManager()
    first = manager.save_profile_sample(
        name="Alice",
        source_audio_path=tmp_path / "alice_a.wav",
        clip_ranges=[(0.0, 1.0)],
        diarization_model_id="pyannote-community-1",
        embedding_model_ref="local-embed-v1",
    )
    second = manager.save_profile_sample(
        name="alice",
        source_audio_path=tmp_path / "alice_b.wav",
        clip_ranges=[(0.0, 1.0)],
        diarization_model_id="pyannote-community-1",
        embedding_model_ref="local-embed-v1",
    )

    assert first.profile_id == second.profile_id
    assert second.sample_count == 2
    assert len(second.samples) == 2


def test_match_ignores_embeddings_from_other_models(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    """! @brief Test match ignores embeddings from other models.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    _tone(tmp_path / "alice.wav", 220.0)
    monkeypatch.setattr("backend.profiles.manager.compute_profile_embedding", _fake_embedding)

    manager = VoiceProfileManager()
    manager.save_profile_sample(
        name="Alice",
        source_audio_path=tmp_path / "alice.wav",
        clip_ranges=[(0.0, 1.0)],
        diarization_model_id="pyannote-community-1",
        embedding_model_ref="local-embed-v1",
    )
    probe = manager.compute_embedding(
        tmp_path / "alice.wav",
        diarization_model_id="pyannote-community-1",
        embedding_model_ref="local-embed-alt",
    )

    result = manager.match(
        probe,
        diarization_model_id="pyannote-community-1",
        embedding_model_ref="local-embed-alt",
    )

    assert result.best_match is None


def test_refresh_task_adds_only_missing_embeddings(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    """! @brief Test refresh task adds only missing embeddings.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    @param tmp_path Value for tmp path.
    """
    _tone(tmp_path / "alice.wav", 220.0)
    calls: list[str] = []

    def tracked_embedding(audio_path: Path, *, model_ref: str, **_: object) -> np.ndarray:
        """! @brief Tracked embedding.
        @param audio_path Path to the audio file.
        @param model_ref Value for model ref.
        @param _ Value for _.
        @return Result produced by the operation.
        """
        calls.append(model_ref)
        return _fake_embedding(audio_path, model_ref=model_ref)

    monkeypatch.setattr("backend.profiles.manager.compute_profile_embedding", tracked_embedding)

    manager = VoiceProfileManager()
    manager.save_profile_sample(
        name="Alice",
        source_audio_path=tmp_path / "alice.wav",
        clip_ranges=[(0.0, 1.0)],
        diarization_model_id="pyannote-community-1",
        embedding_model_ref="local-embed-v1",
    )
    task = manager.start_refresh_task(
        diarization_execution="local",
        local_diarization_model_id="pyannote-community-1",
        openai_diarization_model=None,
        embedding_model_ref="local-embed-v2",
    )

    for _ in range(20):
        latest = manager.get_refresh_task(task.task_id)
        if latest.status in {"completed", "failed"}:
            break
        time.sleep(0.01)
    latest = manager.get_refresh_task(task.task_id)

    assert latest.status == "completed"
    assert calls.count("local-embed-v2") == 1

    task_2 = manager.start_refresh_task(
        diarization_execution="local",
        local_diarization_model_id="pyannote-community-1",
        openai_diarization_model=None,
        embedding_model_ref="local-embed-v2",
    )
    for _ in range(20):
        latest_2 = manager.get_refresh_task(task_2.task_id)
        if latest_2.status in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert calls.count("local-embed-v2") == 1
