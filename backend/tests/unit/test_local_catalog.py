from __future__ import annotations

from pathlib import Path

from backend.app.config import SETTINGS
from backend.app.schemas import LocalModelDefaultRequest, LocalModelRegistrationRequest
from backend.models.local_catalog import LocalModelCatalog


def test_local_catalog_seeds_from_settings(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test local catalog seeds from settings.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    diarization_dir = tmp_path / "pyannote"
    diarization_dir.mkdir()
    (diarization_dir / "config.yaml").write_text("pipeline", encoding="utf-8")
    asr_binary = tmp_path / "whisper-cli"
    asr_binary.write_text("binary", encoding="utf-8")
    asr_model = tmp_path / "model.gguf"
    asr_model.write_text("model", encoding="utf-8")

    object.__setattr__(SETTINGS, "diarization_model_path", str(diarization_dir / "config.yaml"))
    object.__setattr__(SETTINGS, "diarization_pipeline_path", str(diarization_dir / "config.yaml"))
    object.__setattr__(SETTINGS, "voxtral_binary", str(asr_binary))
    object.__setattr__(SETTINGS, "voxtral_model_path", str(asr_model))
    object.__setattr__(SETTINGS, "formatter_backend", "ollama")
    object.__setattr__(SETTINGS, "formatter_ollama_model", "qwen2.5:3b-instruct-q5_K_M")

    catalog = LocalModelCatalog()
    payload = catalog.list_all()

    assert payload.defaults["diarization"] == "pyannote-community-1"
    assert payload.defaults["transcription"] == "whispercpp-local"
    assert payload.defaults["formatter"] == "formatter-ollama-default"
    assert any(item.model_id == "whispercpp-local" for item in payload.models)


def test_register_ollama_model_updates_default(isolated_settings, monkeypatch) -> None:
    """! @brief Test register ollama model updates default.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    """
    catalog = LocalModelCatalog()
    monkeypatch.setattr(catalog, "_ollama_has_model", lambda tag: tag == "phi4-mini")

    record = catalog.register(
        LocalModelRegistrationRequest(
            stage="formatter",
            runtime="ollama",
            name="Phi-4 Mini",
            config={"tag": "phi4-mini"},
            set_as_default=True,
        )
    )

    stage_payload = catalog.list_stage("formatter")
    assert record.model_id == stage_payload.selected_model_id
    assert any(item.config.get("tag") == "phi4-mini" for item in stage_payload.models)


def test_register_whisper_cpp_requires_existing_paths(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test register whisper cpp requires existing paths.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    catalog = LocalModelCatalog()
    asr_binary = tmp_path / "whisper-cli"
    asr_binary.write_text("binary", encoding="utf-8")

    try:
        catalog.register(
            LocalModelRegistrationRequest(
                stage="transcription",
                runtime="whisper.cpp",
                name="Broken Whisper",
                config={"binary_path": str(asr_binary), "model_path": str(tmp_path / "missing.gguf")},
            )
        )
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected whisper.cpp registration to fail for missing model_path")


def test_set_default_rejects_uninstalled_model(isolated_settings, monkeypatch) -> None:
    """! @brief Test set default rejects uninstalled model.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    """
    catalog = LocalModelCatalog()
    monkeypatch.setattr(catalog, "_ollama_has_model", lambda _tag: False)
    payload = catalog._seed_payload()
    payload["models"].append(
        {
            "model_id": "formatter-phi4",
            "stage": "formatter",
            "runtime": "ollama",
            "name": "Phi-4 Mini",
            "installed": False,
            "languages": ["english"],
            "notes": "",
            "config": {"tag": "phi4-mini"},
            "validation_error": "Ollama tag is not installed locally: phi4-mini",
        }
    )
    catalog._write_payload(payload)

    try:
        catalog.set_default(LocalModelDefaultRequest(stage="formatter", model_id="formatter-phi4"))
    except ValueError as exc:
        assert "not installed locally" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected set_default to reject an uninstalled model")
