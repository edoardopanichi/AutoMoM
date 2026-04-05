from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from backend.app.config import SETTINGS
from backend.profiles.manager import VoiceProfileManager


def _tone(path: Path, freq: float) -> None:
    sample_rate = 16000
    t = np.linspace(0, 1.5, int(sample_rate * 1.5), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_embedding_generation_and_matching(isolated_settings, tmp_path: Path) -> None:
    _tone(tmp_path / "alice.wav", 220.0)
    _tone(tmp_path / "bob.wav", 480.0)

    manager = VoiceProfileManager()
    alice_embedding = manager.compute_embedding(tmp_path / "alice.wav")
    manager.upsert_profile("Alice", alice_embedding)

    same_alice = manager.compute_embedding(tmp_path / "alice.wav")
    result = manager.match(same_alice, threshold=0.75)

    assert result.best_match is not None
    assert result.best_match.name == "Alice"
    assert result.best_match.score > 0.75


def test_upsert_profile_merges_samples_for_same_name(isolated_settings, tmp_path: Path) -> None:
    _tone(tmp_path / "alice_a.wav", 220.0)
    _tone(tmp_path / "alice_b.wav", 235.0)

    manager = VoiceProfileManager()
    first = manager.upsert_profile("Alice", manager.compute_embedding(tmp_path / "alice_a.wav"))
    second = manager.upsert_profile("alice", manager.compute_embedding(tmp_path / "alice_b.wav"))

    profiles = manager.list_profiles()
    assert len(profiles) == 1
    assert first.profile_id == second.profile_id
    assert second.sample_count == 2
    assert second.updated_at is not None


def test_profile_filename_includes_speaker_name(isolated_settings, tmp_path: Path) -> None:
    _tone(tmp_path / "alice.wav", 220.0)

    manager = VoiceProfileManager()
    profile = manager.upsert_profile("Alice Smith", manager.compute_embedding(tmp_path / "alice.wav"))

    paths = sorted(SETTINGS.profiles_dir.glob("*.json"))
    assert len(paths) == 1
    assert paths[0].name.startswith("alice_smith--")
    assert profile.profile_id in paths[0].name
