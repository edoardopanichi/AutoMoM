from __future__ import annotations

from pathlib import Path

from backend.app.config import _default_transcription_model_path, required_models


def test_config_source_has_no_legacy_voxtral_env_names() -> None:
    """! @brief Test config source has no legacy voxtral env names."""
    config_path = Path(__file__).resolve().parents[2] / "app" / "config.py"
    content = config_path.read_text(encoding="utf-8")

    assert "AUTOMOM_VOXTRAL_" not in content


def test_default_transcription_model_path_is_transcription_dir() -> None:
    """! @brief Test default transcription model path is in transcription dir."""
    assert _default_transcription_model_path().endswith("data/models/transcription/model.gguf")


def test_required_models_prefers_transcription_download_env(monkeypatch) -> None:
    """! @brief Test required models prefer transcription download env values.
    @param monkeypatch Value for monkeypatch.
    """
    monkeypatch.setenv("AUTOMOM_TRANSCRIPTION_URL", "https://example.invalid/model.gguf")
    monkeypatch.setenv("AUTOMOM_TRANSCRIPTION_SHA256", "abc123")
    monkeypatch.setenv("AUTOMOM_VOXTRAL_URL", "https://legacy.invalid/model.gguf")
    monkeypatch.setenv("AUTOMOM_VOXTRAL_SHA256", "legacy")

    specs = {item.model_id: item for item in required_models()}
    transcription = specs["transcription"]

    assert transcription.download_url == "https://example.invalid/model.gguf"
    assert transcription.checksum_sha256 == "abc123"
