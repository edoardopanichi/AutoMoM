from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from backend.app.config import SETTINGS
from backend.app.schemas import LocalModelRecord, NewJobDefaults


class TemplateResolver(Protocol):
    def get_default_template_id(self) -> str:
        """! @brief Return the selected default template id."""

    def load(self, template_id: str) -> object:
        """! @brief Load a template by id."""


class LocalModelResolver(Protocol):
    def list_all(self) -> object:
        """! @brief Return catalog models and defaults."""

    def resolve_model(self, stage: str, model_id: str | None = None) -> LocalModelRecord:
        """! @brief Resolve a model by stage and id."""


STAGE_FIELDS = {
    "diarization": ("diarization_execution", "local_diarization_model_id"),
    "transcription": ("transcription_execution", "local_transcription_model_id"),
    "formatter": ("formatter_execution", "local_formatter_model_id"),
}


class NewJobDefaultsManager:
    def __init__(self, local_catalog: LocalModelResolver, template_manager: TemplateResolver) -> None:
        """! @brief Initialize new job defaults persistence.
        @param local_catalog Model catalog used to validate saved selections.
        @param template_manager Template manager used to validate template ids.
        """
        self._local_catalog = local_catalog
        self._template_manager = template_manager

    def _path(self) -> Path:
        """! @brief Return persisted defaults path."""
        return SETTINGS.data_dir / "job_defaults.json"

    def load(self) -> NewJobDefaults:
        """! @brief Load saved defaults, repairing stale fields to current fallbacks.
        @return Defaults safe to apply to the New Job form.
        """
        raw = self._read_raw()
        try:
            defaults = NewJobDefaults.model_validate(raw)
        except ValidationError:
            defaults = self._fallback_defaults()
        return self._repair(defaults)

    def save(self, defaults: NewJobDefaults) -> NewJobDefaults:
        """! @brief Validate and persist defaults.
        @param defaults Defaults submitted by the frontend.
        @return Saved defaults.
        """
        self._validate_template(defaults.template_id)
        self._validate_stage_models(defaults)
        self._path().parent.mkdir(parents=True, exist_ok=True)
        self._path().write_text(json.dumps(defaults.model_dump(), indent=2), encoding="utf-8")
        return defaults

    def _read_raw(self) -> dict[str, object]:
        """! @brief Read raw persisted defaults JSON."""
        path = self._path()
        if not path.exists():
            return self._fallback_defaults().model_dump()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._fallback_defaults().model_dump()
        return payload if isinstance(payload, dict) else self._fallback_defaults().model_dump()

    def _fallback_defaults(self) -> NewJobDefaults:
        """! @brief Build defaults from current template/model defaults."""
        template_id = self._template_manager.get_default_template_id()
        catalog = self._local_catalog.list_all()
        defaults = getattr(catalog, "defaults", {})
        return NewJobDefaults(
            template_id=template_id,
            local_diarization_model_id=str(defaults.get("diarization", "")),
            local_transcription_model_id=str(defaults.get("transcription", "")),
            local_formatter_model_id=str(defaults.get("formatter", "")),
        )

    def _repair(self, defaults: NewJobDefaults) -> NewJobDefaults:
        """! @brief Replace stale persisted values with current safe fallbacks."""
        fallback = self._fallback_defaults()
        updates: dict[str, object] = {}
        try:
            self._validate_template(defaults.template_id)
        except (FileNotFoundError, ValueError):
            updates["template_id"] = fallback.template_id

        for stage, (execution_field, model_field) in STAGE_FIELDS.items():
            execution = getattr(defaults, execution_field)
            model_id = getattr(defaults, model_field)
            if execution not in {"local", "remote"}:
                continue
            if not self._model_is_usable(stage, model_id, execution):
                replacement = self._fallback_model_id(stage, execution)
                if replacement:
                    updates[model_field] = replacement
                else:
                    updates[execution_field] = getattr(fallback, execution_field)
                    updates[model_field] = getattr(fallback, model_field)

        return defaults.model_copy(update=updates)

    def _validate_template(self, template_id: str) -> None:
        """! @brief Validate that a template id exists.
        @param template_id Candidate template id.
        """
        self._template_manager.load(template_id)

    def _validate_stage_models(self, defaults: NewJobDefaults) -> None:
        """! @brief Validate saved local/remote stage model selections.
        @param defaults Submitted defaults.
        """
        for stage, (execution_field, model_field) in STAGE_FIELDS.items():
            execution = getattr(defaults, execution_field)
            if execution not in {"local", "remote"}:
                continue
            model_id = getattr(defaults, model_field)
            record = self._local_catalog.resolve_model(stage, model_id)
            if record.location != execution:
                raise ValueError(f"Selected {stage} model is not configured for {execution} execution")
            if not record.installed:
                raise ValueError(record.validation_error or f"Selected {stage} model is not installed")

    def _model_is_usable(self, stage: str, model_id: str, execution: str) -> bool:
        """! @brief Return whether a persisted model selection can still be applied."""
        try:
            record = self._local_catalog.resolve_model(stage, model_id)
        except ValueError:
            return False
        return record.location == execution and record.installed

    def _fallback_model_id(self, stage: str, execution: str) -> str:
        """! @brief Return an installed fallback model id for the requested stage/location."""
        catalog = self._local_catalog.list_all()
        defaults = getattr(catalog, "defaults", {})
        default_id = str(defaults.get(stage, ""))
        if self._model_is_usable(stage, default_id, execution):
            return default_id
        for record in getattr(catalog, "models", []):
            if record.stage == stage and record.location == execution and record.installed:
                return record.model_id
        return ""
