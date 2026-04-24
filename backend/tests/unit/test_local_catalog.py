from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from backend.app.config import SETTINGS
from backend.app.schemas import LocalModelRegistrationRequest
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
    object.__setattr__(SETTINGS, "transcription_binary", str(asr_binary))
    object.__setattr__(SETTINGS, "transcription_model_path", str(asr_model))
    object.__setattr__(SETTINGS, "formatter_backend", "ollama")
    object.__setattr__(SETTINGS, "formatter_ollama_model", "qwen2.5:3b-instruct-q5_K_M")

    catalog = LocalModelCatalog()
    payload = catalog.list_all()

    assert {item.model_id for item in payload.models} >= {
        "pyannote-community-1",
        "whispercpp-local",
        "formatter-ollama-default",
    }
    assert any(item.model_id == "whispercpp-local" for item in payload.models)


def test_register_ollama_model_makes_it_selectable(isolated_settings, monkeypatch) -> None:
    """! @brief Test register ollama model makes it selectable.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    """
    catalog = LocalModelCatalog()
    monkeypatch.setattr(catalog, "_ollama_has_model", lambda tag: tag == "phi4-mini")

    record = catalog.register(
        LocalModelRegistrationRequest(
            stage="formatter",
            location="local",
            runtime="ollama",
            name="Phi-4 Mini",
            config={"tag": "phi4-mini"},
        )
    )

    stage_payload = catalog.list_stage("formatter")
    assert record.model_id in {item.model_id for item in stage_payload.models}
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
                location="local",
                runtime="whisper.cpp",
                name="Broken Whisper",
                config={"binary_path": str(asr_binary), "model_path": str(tmp_path / "missing.gguf")},
            )
        )
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected whisper.cpp registration to fail for missing model_path")


def test_delete_model_is_not_blocked_by_legacy_defaults(isolated_settings, monkeypatch) -> None:
    """! @brief Test delete model is not blocked by legacy defaults.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    """
    catalog = LocalModelCatalog()
    monkeypatch.setattr(catalog, "_ollama_has_model", lambda tag: tag == "phi4-mini")
    record = catalog.register(
        LocalModelRegistrationRequest(
            stage="formatter",
            location="local",
            runtime="ollama",
            name="Phi-4 Mini",
            config={"tag": "phi4-mini"},
        )
    )

    catalog.delete(record.model_id)

    assert all(item.model_id != record.model_id for item in catalog.list_all().models)


def test_registration_rejects_set_as_default(isolated_settings) -> None:
    """! @brief Test registration rejects legacy default flag.
    @param isolated_settings Value for isolated settings.
    """
    try:
        LocalModelRegistrationRequest.model_validate(
            {
                "stage": "formatter",
                "location": "local",
                "runtime": "ollama",
                "name": "Phi-4 Mini",
                "config": {"tag": "phi4-mini"},
                "set_as_default": True,
            }
        )
    except ValidationError as exc:
        assert "Extra inputs are not permitted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected set_as_default to be rejected")


def test_local_catalog_repairs_seeded_default_paths(isolated_settings, tmp_path: Path, monkeypatch) -> None:
    """! @brief Test local catalog repairs seeded default paths.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    @param monkeypatch Value for monkeypatch.
    """
    real_pyannote = tmp_path / "real-pyannote" / "config.yaml"
    real_pyannote.parent.mkdir()
    real_pyannote.write_text("pipeline", encoding="utf-8")
    real_binary = tmp_path / "real-whisper-cli"
    real_binary.write_text("binary", encoding="utf-8")
    real_model = tmp_path / "real-model.gguf"
    real_model.write_text("model", encoding="utf-8")
    missing_root = tmp_path / "removed"

    object.__setattr__(SETTINGS, "diarization_pipeline_path", str(real_pyannote))
    object.__setattr__(SETTINGS, "transcription_binary", str(real_binary))
    object.__setattr__(SETTINGS, "transcription_model_path", str(real_model))

    catalog = LocalModelCatalog()
    monkeypatch.setattr(catalog, "_ollama_has_model", lambda _tag: False)
    payload = catalog._seed_payload()
    for item in payload["models"]:
        if item["model_id"] == "pyannote-community-1":
            item["config"]["pipeline_path"] = str(missing_root / "pyannote" / "config.yaml")
        if item["model_id"] == "whispercpp-local":
            item["config"]["binary_path"] = str(missing_root / "whisper-cli")
            item["config"]["model_path"] = str(missing_root / "model.gguf")
    catalog._write_payload(payload)

    repaired = catalog.list_all()
    by_id = {item.model_id: item for item in repaired.models}

    assert by_id["pyannote-community-1"].config["pipeline_path"] == str(real_pyannote)
    assert by_id["whispercpp-local"].config["binary_path"] == str(real_binary)
    assert by_id["whispercpp-local"].config["model_path"] == str(real_model)


def test_runtime_descriptors_are_stage_first_and_hide_command_as_advanced(isolated_settings) -> None:
    """! @brief Test model manager runtime descriptors.
    @param isolated_settings Value for isolated settings.
    """
    catalog = LocalModelCatalog()

    descriptors = catalog.runtime_descriptors()
    by_key = {(item.stage, item.location, item.runtime): item for item in descriptors}

    assert by_key[("diarization", "local", "pyannote")].stage == "diarization"
    assert by_key[("diarization", "remote", "pyannote")].location == "remote"
    assert by_key[("transcription", "local", "whisper.cpp")].stage == "transcription"
    assert by_key[("transcription", "remote", "whisper.cpp")].location == "remote"
    assert by_key[("transcription", "local", "faster-whisper")].stage == "transcription"
    assert by_key[("formatter", "local", "ollama")].supports_install is True
    assert by_key[("formatter", "remote", "ollama")].location == "remote"
    assert by_key[("formatter", "local", "command")].advanced is True
    assert any(field.key == "tag" and field.required for field in by_key[("formatter", "local", "ollama")].fields)


def test_discover_pyannote_uses_known_model_dir(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test pyannote discovery in known model directories.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    pipeline = tmp_path / "models" / "diarization" / "custom-pyannote" / "config.yaml"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("pipeline", encoding="utf-8")
    object.__setattr__(SETTINGS, "models_dir", tmp_path / "models")
    object.__setattr__(SETTINGS, "diarization_pipeline_path", "")
    object.__setattr__(SETTINGS, "diarization_model_path", "")
    object.__setattr__(SETTINGS, "diarization_embedding_model", "local-embedding")

    catalog = LocalModelCatalog()
    payload = catalog.discover("diarization", "local", "pyannote")

    assert any(item.config["pipeline_path"] == str(pipeline) for item in payload.suggestions)
    assert any(item.config["embedding_model_ref"] == "local-embedding" for item in payload.suggestions)


def test_discover_whisper_cpp_pairs_known_binary_and_model(isolated_settings, tmp_path: Path) -> None:
    """! @brief Test whisper.cpp discovery in known model directories.
    @param isolated_settings Value for isolated settings.
    @param tmp_path Value for tmp path.
    """
    binary = tmp_path / "whisper-cli"
    binary.write_text("binary", encoding="utf-8")
    model = tmp_path / "models" / "transcription" / "tiny.gguf"
    model.parent.mkdir(parents=True)
    model.write_text("model", encoding="utf-8")
    object.__setattr__(SETTINGS, "models_dir", tmp_path / "models")
    object.__setattr__(SETTINGS, "transcription_binary", str(binary))
    object.__setattr__(SETTINGS, "transcription_model_path", "")

    catalog = LocalModelCatalog()
    payload = catalog.discover("transcription", "local", "whisper.cpp")

    assert any(
        item.config["binary_path"] == str(binary) and item.config["model_path"] == str(model)
        for item in payload.suggestions
    )


def test_discover_ollama_uses_local_tags(isolated_settings, monkeypatch) -> None:
    """! @brief Test Ollama discovery from local tags.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    """
    catalog = LocalModelCatalog()
    monkeypatch.setattr(catalog, "_ollama_tags", lambda: ["qwen2.5:3b", "phi4-mini"])

    payload = catalog.discover("formatter", "local", "ollama")

    assert [item.config["tag"] for item in payload.suggestions] == ["qwen2.5:3b", "phi4-mini"]


def test_register_remote_whisper_cpp_validates_worker_health(isolated_settings, monkeypatch) -> None:
    catalog = LocalModelCatalog()
    monkeypatch.setattr(
        catalog,
        "_remote_worker_health",
        lambda base_url, auth_token, timeout_s, stage: (
            {"enabled_stages": ["transcription"], "transcription": {"runtime": "whisper.cpp", "model_name": "large-v3"}},
            None,
        ),
    )

    record = catalog.register(
        LocalModelRegistrationRequest(
            stage="transcription",
            location="remote",
            runtime="whisper.cpp",
            name="Office GPU",
            config={"base_url": "http://office-gpu:8011", "model_name": "large-v3"},
        )
    )

    assert record.location == "remote"
    assert record.installed is True


def test_register_remote_ollama_uses_remote_host(isolated_settings, monkeypatch) -> None:
    catalog = LocalModelCatalog()
    monkeypatch.setattr(catalog, "_ollama_has_model", lambda tag, **kwargs: kwargs.get("base_url") == "http://office-gpu:11434")

    record = catalog.register(
        LocalModelRegistrationRequest(
            stage="formatter",
            location="remote",
            runtime="ollama",
            name="Remote Qwen",
            config={"base_url": "http://office-gpu:11434", "tag": "qwen2.5:3b"},
        )
    )

    assert record.location == "remote"
    assert record.installed is True
