from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from backend.app.config import SETTINGS
from backend.app.schemas import (
    LocalModelCatalogResponse,
    LocalModelDefaultRequest,
    LocalModelRecord,
    LocalModelRegistrationRequest,
    LocalModelRuntime,
    LocalModelStage,
    LocalStageModelResponse,
)
from backend.pipeline.transcription import _probe_asr_binary


STAGES: tuple[LocalModelStage, ...] = ("diarization", "transcription", "formatter")
RUNTIMES_BY_STAGE: dict[LocalModelStage, tuple[LocalModelRuntime, ...]] = {
    "diarization": ("pyannote",),
    "transcription": ("whisper.cpp", "faster-whisper"),
    "formatter": ("ollama", "command"),
}


class LocalModelCatalog:
    def _catalog_path(self) -> Path:
        """! @brief Catalog path.
        @return Path result produced by the operation.
        """
        return SETTINGS.models_dir / "catalog.json"

    def _defaults_path(self) -> Path:
        """! @brief Defaults path.
        @return Path result produced by the operation.
        """
        return SETTINGS.models_dir / "defaults.json"

    def _legacy_formatter_selection_path(self) -> Path:
        """! @brief Legacy formatter selection path.
        @return Path result produced by the operation.
        """
        return SETTINGS.models_dir / "formatter" / "selected_model.txt"

    def list_all(self) -> LocalModelCatalogResponse:
        """! @brief List all operation.
        @return Result produced by the operation.
        """
        payload = self._load_payload()
        return LocalModelCatalogResponse(
            defaults=payload["defaults"],
            models=[self._validate_record(LocalModelRecord.model_validate(item)) for item in payload["models"]],
        )

    def list_stage(self, stage: LocalModelStage) -> LocalStageModelResponse:
        """! @brief List stage operation.
        @param stage Value for stage.
        @return Result produced by the operation.
        """
        self._validate_stage_name(stage)
        payload = self._load_payload()
        models = [
            self._validate_record(LocalModelRecord.model_validate(item))
            for item in payload["models"]
            if item.get("stage") == stage
        ]
        selected = payload["defaults"].get(stage) or (models[0].model_id if models else "")
        return LocalStageModelResponse(stage=stage, selected_model_id=selected, models=models)

    def resolve_model(self, stage: LocalModelStage, model_id: str | None = None) -> LocalModelRecord:
        """! @brief Resolve model.
        @param stage Value for stage.
        @param model_id Identifier of the target model.
        @return Result produced by the operation.
        """
        self._validate_stage_name(stage)
        payload = self._load_payload()
        wanted = (model_id or payload["defaults"].get(stage) or "").strip()
        models = [
            self._validate_record(LocalModelRecord.model_validate(item))
            for item in payload["models"]
            if item.get("stage") == stage
        ]
        if not models:
            raise ValueError(f"No local models are registered for stage '{stage}'")
        if not wanted:
            return models[0]
        for item in models:
            if item.model_id == wanted:
                return item
        raise ValueError(f"Unknown local {stage} model: {wanted}")

    def register(self, request: LocalModelRegistrationRequest) -> LocalModelRecord:
        """! @brief Register operation.
        @param request Request payload for the operation.
        @return Result produced by the operation.
        """
        self._validate_stage_runtime(request.stage, request.runtime)
        payload = self._load_payload()
        candidate_id = self._normalize_model_id(
            request.model_id or self._suggest_model_id(request.stage, request.runtime, request.name)
        )
        if any(item.get("model_id") == candidate_id for item in payload["models"]):
            raise ValueError(f"Model id '{candidate_id}' is already registered")

        record = LocalModelRecord(
            model_id=candidate_id,
            stage=request.stage,
            runtime=request.runtime,
            name=request.name.strip(),
            installed=False,
            languages=[item.strip() for item in request.languages if item.strip()],
            notes=request.notes.strip(),
            config={key: str(value).strip() for key, value in request.config.items() if str(value).strip()},
            validation_error=None,
        )
        validated = self._validate_record(record)
        if not validated.installed:
            raise ValueError(validated.validation_error or "Model validation failed")

        payload["models"].append(validated.model_dump())
        if request.set_as_default or not payload["defaults"].get(request.stage):
            payload["defaults"][request.stage] = validated.model_id
        self._write_payload(payload)
        self._sync_legacy_settings()
        return validated

    def delete(self, model_id: str) -> None:
        """! @brief Delete operation.
        @param model_id Identifier of the target model.
        """
        payload = self._load_payload()
        for stage, selected_model_id in payload["defaults"].items():
            if selected_model_id == model_id:
                raise ValueError("Cannot delete the selected default model for a stage")
        next_models = [item for item in payload["models"] if item.get("model_id") != model_id]
        if len(next_models) == len(payload["models"]):
            raise ValueError(f"Unknown local model: {model_id}")
        payload["models"] = next_models
        self._write_payload(payload)

    def set_default(self, request: LocalModelDefaultRequest) -> LocalStageModelResponse:
        """! @brief Set default operation.
        @param request Request payload for the operation.
        @return Result produced by the operation.
        """
        payload = self._load_payload()
        selected = self.resolve_model(request.stage, request.model_id)
        if not selected.installed:
            raise ValueError(selected.validation_error or "Selected model is not ready")
        payload["defaults"][request.stage] = selected.model_id
        self._write_payload(payload)
        self._sync_legacy_settings()
        return self.list_stage(request.stage)

    def validate_selection(self, selections: dict[LocalModelStage, str]) -> tuple[bool, str | None]:
        """! @brief Validate selection.
        @param selections Value for selections.
        @return Tuple produced by the operation.
        """
        for stage, model_id in selections.items():
            record = self.resolve_model(stage, model_id)
            if not record.installed:
                return False, record.validation_error or f"Local {stage} model '{record.name}' is not installed"
        return True, None

    def get_formatter_tag(self) -> str:
        """! @brief Get formatter tag.
        @return str result produced by the operation.
        """
        try:
            selected = self.resolve_model("formatter")
        except ValueError:
            return ""
        if selected.runtime != "ollama":
            return ""
        return selected.config.get("tag", "")

    def set_formatter_tag(self, model_tag: str) -> str:
        """! @brief Set formatter tag.
        @param model_tag Value for model tag.
        @return str result produced by the operation.
        """
        normalized = model_tag.strip()
        if not normalized:
            raise ValueError("Formatter model tag cannot be empty")
        stage_models = self.list_stage("formatter")
        for item in stage_models.models:
            if item.runtime == "ollama" and item.config.get("tag") == normalized:
                self.set_default(LocalModelDefaultRequest(stage="formatter", model_id=item.model_id))
                return normalized

        record = self.register(
            LocalModelRegistrationRequest(
                stage="formatter",
                runtime="ollama",
                name=normalized,
                config={"tag": normalized},
                set_as_default=True,
            )
        )
        return record.config.get("tag", normalized)

    def _load_payload(self) -> dict[str, object]:
        """! @brief Load payload.
        @return Dictionary produced by the operation.
        """
        self._ensure_seeded()
        try:
            models = json.loads(self._catalog_path().read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid local model catalog JSON: {exc}") from exc
        try:
            defaults = json.loads(self._defaults_path().read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid local model defaults JSON: {exc}") from exc

        normalized_defaults = {stage: str(defaults.get(stage, "")).strip() for stage in STAGES}
        normalized_models = [
            LocalModelRecord.model_validate(item).model_dump(mode="json")
            for item in models
        ]
        normalized_models, repaired = self._repair_seeded_default_paths(normalized_models)
        if any(stage not in normalized_defaults for stage in STAGES):
            for stage in STAGES:
                normalized_defaults.setdefault(stage, "")
            repaired = True
        if repaired:
            self._write_payload({"models": normalized_models, "defaults": normalized_defaults})
        return {"models": normalized_models, "defaults": normalized_defaults}

    def _write_payload(self, payload: dict[str, object]) -> None:
        """! @brief Write payload.
        @param payload Value for payload.
        """
        self._catalog_path().parent.mkdir(parents=True, exist_ok=True)
        self._catalog_path().write_text(json.dumps(payload["models"], indent=2), encoding="utf-8")
        self._defaults_path().write_text(json.dumps(payload["defaults"], indent=2), encoding="utf-8")

    def _ensure_seeded(self) -> None:
        """! @brief Ensure seeded.
        """
        if self._catalog_path().exists() and self._defaults_path().exists():
            return
        seeded = self._seed_payload()
        self._write_payload(seeded)

    def _seed_payload(self) -> dict[str, object]:
        """! @brief Seed payload.
        @return Dictionary produced by the operation.
        """
        formatter_runtime: LocalModelRuntime = "command" if SETTINGS.formatter_backend == "command" else "ollama"
        formatter_model_id = "formatter-command-default" if formatter_runtime == "command" else "formatter-ollama-default"
        formatter_tag = self._legacy_formatter_tag() or SETTINGS.formatter_ollama_model
        models = [
            LocalModelRecord(
                model_id="pyannote-community-1",
                stage="diarization",
                runtime="pyannote",
                name="Pyannote Community-1",
                installed=False,
                languages=["multilingual"],
                config={
                    "pipeline_path": SETTINGS.diarization_pipeline_path or SETTINGS.diarization_model_path,
                    "embedding_model_ref": SETTINGS.diarization_embedding_model,
                },
            ),
            LocalModelRecord(
                model_id="whispercpp-local",
                stage="transcription",
                runtime="whisper.cpp",
                name="whisper.cpp local ASR",
                installed=False,
                languages=["multilingual"],
                config={
                    "binary_path": SETTINGS.voxtral_binary,
                    "model_path": SETTINGS.voxtral_model_path,
                },
            ),
            LocalModelRecord(
                model_id=formatter_model_id,
                stage="formatter",
                runtime=formatter_runtime,
                name="Current local formatter",
                installed=False,
                languages=["english"],
                config=(
                    {
                        "command_template": SETTINGS.formatter_command,
                        "model_path": SETTINGS.formatter_model_path,
                    }
                    if formatter_runtime == "command"
                    else {"tag": formatter_tag}
                ),
            ),
        ]
        return {
            "models": [self._validate_record(item).model_dump(mode="json") for item in models],
            "defaults": {
                "diarization": "pyannote-community-1",
                "transcription": "whispercpp-local",
                "formatter": formatter_model_id,
            },
        }

    def _validate_record(self, record: LocalModelRecord) -> LocalModelRecord:
        """! @brief Validate record.
        @param record Value for record.
        @return Result produced by the operation.
        """
        validator = {
            "pyannote": self._validate_pyannote,
            "whisper.cpp": self._validate_whisper_cpp,
            "faster-whisper": self._validate_faster_whisper,
            "ollama": self._validate_ollama,
            "command": self._validate_command,
        }[record.runtime]
        installed, validation_error = validator(record.config)
        return record.model_copy(update={"installed": installed, "validation_error": validation_error})

    def _repair_seeded_default_paths(self, models: list[dict[str, object]]) -> tuple[list[dict[str, object]], bool]:
        """! @brief Repair invalid seeded model paths from current settings when possible.
        @param models Value for models.
        @return Tuple produced by the operation.
        """
        repaired = False
        next_models: list[dict[str, object]] = []
        for item in models:
            record = LocalModelRecord.model_validate(item)
            updated_config = dict(record.config)
            if record.model_id == "pyannote-community-1" and record.runtime == "pyannote":
                candidate = SETTINGS.diarization_pipeline_path or SETTINGS.diarization_model_path
                if self._should_repair_path(updated_config.get("pipeline_path", ""), candidate):
                    updated_config["pipeline_path"] = candidate
            elif record.model_id == "whispercpp-local" and record.runtime == "whisper.cpp":
                if self._should_repair_path(updated_config.get("binary_path", ""), SETTINGS.voxtral_binary):
                    updated_config["binary_path"] = SETTINGS.voxtral_binary
                if self._should_repair_path(updated_config.get("model_path", ""), SETTINGS.voxtral_model_path):
                    updated_config["model_path"] = SETTINGS.voxtral_model_path
            if updated_config != record.config:
                record = record.model_copy(update={"config": updated_config})
                repaired = True
            next_models.append(record.model_dump(mode="json"))
        return next_models, repaired

    @staticmethod
    def _should_repair_path(current_path: str, candidate_path: str) -> bool:
        """! @brief Check whether a stored path should be repaired from settings.
        @param current_path Value for current path.
        @param candidate_path Value for candidate path.
        @return True when the requested condition is satisfied; otherwise False.
        """
        current = current_path.strip()
        candidate = candidate_path.strip()
        if not candidate or not Path(candidate).expanduser().exists():
            return False
        return not current or not Path(current).expanduser().exists()

    def _validate_pyannote(self, config: dict[str, str]) -> tuple[bool, str | None]:
        """! @brief Validate pyannote.
        @param config Value for config.
        @return Tuple produced by the operation.
        """
        pipeline_path = config.get("pipeline_path", "").strip()
        embedding_ref = config.get("embedding_model_ref", "").strip()
        if not pipeline_path:
            return False, "Pyannote model requires pipeline_path"
        if not Path(pipeline_path).expanduser().exists():
            return False, f"Pyannote pipeline path does not exist: {pipeline_path}"
        if not embedding_ref:
            return False, "Pyannote model requires embedding_model_ref"
        return True, None

    def _validate_whisper_cpp(self, config: dict[str, str]) -> tuple[bool, str | None]:
        """! @brief Validate whisper cpp.
        @param config Value for config.
        @return Tuple produced by the operation.
        """
        binary_path = config.get("binary_path", "").strip()
        model_path = config.get("model_path", "").strip()
        resolved_binary = self._resolve_binary(binary_path)
        if not resolved_binary:
            return False, "whisper.cpp model requires a valid binary_path"
        if not model_path:
            return False, "whisper.cpp model requires model_path"
        if not Path(model_path).expanduser().exists():
            return False, f"whisper.cpp model file does not exist: {model_path}"
        capabilities = _probe_asr_binary(resolved_binary)
        if not capabilities.is_whisper_cli:
            return False, f"Binary is not a whisper.cpp CLI: {resolved_binary}"
        return True, None

    def _validate_faster_whisper(self, config: dict[str, str]) -> tuple[bool, str | None]:
        """! @brief Validate faster whisper.
        @param config Value for config.
        @return Tuple produced by the operation.
        """
        model_path = config.get("model_path", "").strip()
        if not model_path:
            return False, "faster-whisper model requires model_path"
        if not Path(model_path).expanduser().exists():
            return False, f"faster-whisper model path does not exist: {model_path}"
        try:
            import faster_whisper  # noqa: F401
        except Exception as exc:
            return False, f"faster-whisper is not installed: {exc}"
        return True, None

    def _validate_ollama(self, config: dict[str, str]) -> tuple[bool, str | None]:
        """! @brief Validate ollama.
        @param config Value for config.
        @return Tuple produced by the operation.
        """
        tag = config.get("tag", "").strip()
        if not tag:
            return False, "Ollama model requires tag"
        if not self._ollama_has_model(tag):
            return False, f"Ollama tag is not installed locally: {tag}"
        return True, None

    def _validate_command(self, config: dict[str, str]) -> tuple[bool, str | None]:
        """! @brief Validate command.
        @param config Value for config.
        @return Tuple produced by the operation.
        """
        command_template = config.get("command_template", "").strip()
        model_path = config.get("model_path", "").strip()
        if not command_template:
            return False, "Command formatter requires command_template"
        if not model_path:
            return False, "Command formatter requires model_path"
        if not Path(model_path).expanduser().exists():
            return False, f"Command formatter model path does not exist: {model_path}"
        return True, None

    def _ollama_has_model(self, model_tag: str) -> bool:
        """! @brief Ollama has model.
        @param model_tag Value for model tag.
        @return True when the requested condition is satisfied; otherwise False.
        """
        request = urllib.request.Request(url=f"{SETTINGS.ollama_host.rstrip('/')}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return False

        wanted = model_tag.strip().lower()
        for item in payload.get("models", []):
            name = str(item.get("name", "")).strip().lower()
            if name == wanted:
                return True
        return False

    def _legacy_formatter_tag(self) -> str:
        """! @brief Legacy formatter tag.
        @return str result produced by the operation.
        """
        path = self._legacy_formatter_selection_path()
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _sync_legacy_settings(self) -> None:
        """! @brief Sync legacy settings.
        """
        try:
            formatter = self.resolve_model("formatter")
        except ValueError:
            return
        if formatter.runtime == "ollama":
            tag = formatter.config.get("tag", "").strip()
            if tag:
                self._legacy_formatter_selection_path().parent.mkdir(parents=True, exist_ok=True)
                self._legacy_formatter_selection_path().write_text(tag, encoding="utf-8")
                object.__setattr__(SETTINGS, "formatter_ollama_model", tag)
        elif formatter.runtime == "command":
            object.__setattr__(SETTINGS, "formatter_command", formatter.config.get("command_template", ""))
            object.__setattr__(SETTINGS, "formatter_model_path", formatter.config.get("model_path", ""))

    def _resolve_binary(self, binary_path: str) -> str | None:
        """! @brief Resolve binary.
        @param binary_path Value for binary path.
        @return Result produced by the operation.
        """
        candidate = Path(binary_path).expanduser()
        if candidate.exists():
            return str(candidate)
        return shutil.which(binary_path)

    def _validate_stage_runtime(self, stage: LocalModelStage, runtime: LocalModelRuntime) -> None:
        """! @brief Validate stage runtime.
        @param stage Value for stage.
        @param runtime Value for runtime.
        """
        self._validate_stage_name(stage)
        if runtime not in RUNTIMES_BY_STAGE[stage]:
            raise ValueError(f"Runtime '{runtime}' is not supported for stage '{stage}'")

    @staticmethod
    def _validate_stage_name(stage: str) -> None:
        """! @brief Validate stage name.
        @param stage Value for stage.
        """
        if stage not in STAGES:
            raise KeyError(stage)

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        """! @brief Normalize model id.
        @param model_id Value for model id.
        @return str result produced by the operation.
        """
        normalized = re.sub(r"[^a-z0-9]+", "-", model_id.strip().lower()).strip("-")
        if not normalized:
            raise ValueError("Model id cannot be empty")
        return normalized[:80]

    @classmethod
    def _suggest_model_id(cls, stage: LocalModelStage, runtime: LocalModelRuntime, name: str) -> str:
        """! @brief Suggest model id.
        @param stage Value for stage.
        @param runtime Value for runtime.
        @param name Value for name.
        @return str result produced by the operation.
        """
        base = cls._normalize_model_id(f"{stage}-{runtime}-{name}")
        return base


LOCAL_MODEL_CATALOG = LocalModelCatalog()
