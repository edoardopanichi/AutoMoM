from __future__ import annotations

from types import SimpleNamespace

from pydantic import ValidationError

from backend.app.job_defaults import NewJobDefaultsManager
from backend.app.schemas import LocalModelRecord, NewJobDefaults


class FakeTemplateManager:
    def __init__(self, templates: set[str] | None = None) -> None:
        self.templates = templates or {"default", "board_sync"}

    def load(self, template_id: str) -> object:
        """! @brief Load fake template."""
        if template_id not in self.templates:
            raise FileNotFoundError(template_id)
        return object()


class FakeLocalCatalog:
    def __init__(self) -> None:
        self.records = {
            "diarization-local": LocalModelRecord(
                model_id="diarization-local",
                stage="diarization",
                location="local",
                runtime="pyannote",
                name="Local diarization",
                installed=True,
            ),
            "transcription-remote": LocalModelRecord(
                model_id="transcription-remote",
                stage="transcription",
                location="remote",
                runtime="whisper.cpp",
                name="Remote transcription",
                installed=True,
            ),
            "transcription-local": LocalModelRecord(
                model_id="transcription-local",
                stage="transcription",
                location="local",
                runtime="whisper.cpp",
                name="Local transcription",
                installed=True,
            ),
            "formatter-local": LocalModelRecord(
                model_id="formatter-local",
                stage="formatter",
                location="local",
                runtime="ollama",
                name="Local formatter",
                installed=True,
            ),
            "formatter-missing": LocalModelRecord(
                model_id="formatter-missing",
                stage="formatter",
                location="local",
                runtime="ollama",
                name="Missing formatter",
                installed=False,
                validation_error="Ollama tag is not installed locally: missing",
            ),
        }

    def list_all(self) -> object:
        """! @brief Return fake catalog records."""
        return SimpleNamespace(models=list(self.records.values()))

    def resolve_model(self, stage: str, model_id: str | None = None) -> LocalModelRecord:
        """! @brief Resolve a fake model by id."""
        record = self.records.get(model_id or "")
        if not record or record.stage != stage:
            raise ValueError(f"Unknown local {stage} model: {model_id}")
        return record


def test_new_job_defaults_persist_across_manager_instances(isolated_settings) -> None:
    """! @brief Test saved New Job defaults persist on disk.
    @param isolated_settings Value for isolated settings.
    """
    catalog = FakeLocalCatalog()
    templates = FakeTemplateManager()
    manager = NewJobDefaultsManager(catalog, templates)

    saved = manager.save(
        NewJobDefaults(
            template_id="board_sync",
            diarization_execution="local",
            transcription_execution="remote",
            formatter_execution="api",
            local_diarization_model_id="diarization-local",
            local_transcription_model_id="transcription-remote",
            local_formatter_model_id="formatter-local",
            openai_formatter_model="gpt-4.1",
        )
    )

    reloaded = NewJobDefaultsManager(catalog, templates).load()

    assert saved.template_id == "board_sync"
    assert reloaded.template_id == "board_sync"
    assert reloaded.transcription_execution == "remote"
    assert reloaded.openai_formatter_model == "gpt-4.1"


def test_new_job_defaults_initial_fallback_uses_first_installed_local_models(isolated_settings) -> None:
    """! @brief Test initial defaults use built-in local choices.
    @param isolated_settings Value for isolated settings.
    """
    defaults = NewJobDefaultsManager(FakeLocalCatalog(), FakeTemplateManager()).load()

    assert defaults.template_id == "default"
    assert defaults.diarization_execution == "local"
    assert defaults.transcription_execution == "local"
    assert defaults.formatter_execution == "local"
    assert defaults.local_diarization_model_id == "diarization-local"
    assert defaults.local_transcription_model_id == "transcription-local"
    assert defaults.local_formatter_model_id == "formatter-local"


def test_new_job_defaults_repairs_stale_saved_model(isolated_settings) -> None:
    """! @brief Test stale saved model selections repair to built-in local choices.
    @param isolated_settings Value for isolated settings.
    """
    catalog = FakeLocalCatalog()
    manager = NewJobDefaultsManager(catalog, FakeTemplateManager())
    manager.save(
        NewJobDefaults(
            diarization_execution="api",
            transcription_execution="api",
            formatter_execution="local",
            local_formatter_model_id="formatter-local",
        )
    )
    del catalog.records["formatter-local"]
    catalog.records["formatter-replacement"] = LocalModelRecord(
        model_id="formatter-replacement",
        stage="formatter",
        location="local",
        runtime="ollama",
        name="Replacement formatter",
        installed=True,
    )

    repaired = manager.load()

    assert repaired.formatter_execution == "local"
    assert repaired.local_formatter_model_id == "formatter-replacement"


def test_new_job_defaults_reject_invalid_template(isolated_settings) -> None:
    """! @brief Test invalid template ids are rejected on save.
    @param isolated_settings Value for isolated settings.
    """
    manager = NewJobDefaultsManager(FakeLocalCatalog(), FakeTemplateManager())

    try:
        manager.save(NewJobDefaults(template_id="missing"))
    except FileNotFoundError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid template to be rejected")


def test_new_job_defaults_reject_invalid_stage_model(isolated_settings) -> None:
    """! @brief Test invalid model defaults are rejected on save.
    @param isolated_settings Value for isolated settings.
    """
    manager = NewJobDefaultsManager(FakeLocalCatalog(), FakeTemplateManager())

    try:
        manager.save(
            NewJobDefaults(
                diarization_execution="api",
                transcription_execution="api",
                formatter_execution="local",
                local_formatter_model_id="formatter-missing",
            )
        )
    except ValueError as exc:
        assert "not installed locally" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unavailable model to be rejected")


def test_new_job_defaults_schema_excludes_title_audio_and_api_key() -> None:
    """! @brief Test unsafe or per-job-only fields cannot be persisted."""
    payload = NewJobDefaults().model_dump()
    assert "title" not in payload
    assert "audio_file" not in payload
    assert "openai_api_key" not in payload

    try:
        NewJobDefaults.model_validate(
            {
                **payload,
                "title": "Private title",
                "audio_file": "/tmp/audio.wav",
                "openai_api_key": "sk-secret",
            }
        )
    except ValidationError as exc:
        assert "Extra inputs are not permitted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected excluded fields to be rejected")
