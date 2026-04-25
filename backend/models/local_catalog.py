from __future__ import annotations

import json
import os
import re
import shutil
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.app.config import SETTINGS
from backend.app.schemas import (
    LocalModelCatalogResponse,
    LocalModelDiscoveryResponse,
    LocalModelDiscoverySuggestion,
    LocalModelFieldDescriptor,
    LocalModelInstallRequest,
    LocalModelInstallTask,
    LocalModelLocation,
    LocalModelRecord,
    LocalModelRegistrationRequest,
    LocalModelRuntime,
    LocalModelRuntimeDescriptor,
    LocalModelStage,
    LocalStageModelResponse,
)
from backend.pipeline.transcription import _probe_asr_binary


STAGES: tuple[LocalModelStage, ...] = ("diarization", "transcription", "formatter")
RUNTIMES_BY_STAGE_LOCATION: dict[tuple[LocalModelStage, LocalModelLocation], tuple[LocalModelRuntime, ...]] = {
    ("diarization", "local"): ("pyannote",),
    ("diarization", "remote"): ("pyannote",),
    ("transcription", "local"): ("whisper.cpp", "faster-whisper"),
    ("transcription", "remote"): ("whisper.cpp",),
    ("formatter", "local"): ("ollama", "command"),
    ("formatter", "remote"): ("ollama",),
}

VALID_FASTER_WHISPER_COMPUTE_TYPES: set[str] = {
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int16",
    "float16",
    "float32",
    "bfloat16",
}
FASTER_WHISPER_REQUIRED_MODEL_FILES: tuple[str, ...] = (
    "model.bin",
    "model.safetensors",
    "model.bin.index.json",
    "model.safetensors.index.json",
)


def validate_faster_whisper_model_directory(model_path: Path) -> tuple[bool, str | None]:
    """! @brief Validate a faster-whisper/CTranslate2 model directory."""
    expanded = model_path.expanduser()
    if not expanded.exists():
        return False, f"faster-whisper model path does not exist: {model_path}"
    if not expanded.is_dir():
        return False, f"faster-whisper model path is not a directory: {model_path}"
    if not (expanded / "config.json").is_file():
        return False, f"faster-whisper model directory is missing config.json: {model_path}"
    if not any((expanded / name).is_file() for name in FASTER_WHISPER_REQUIRED_MODEL_FILES):
        required = ", ".join(FASTER_WHISPER_REQUIRED_MODEL_FILES)
        return False, (
            f"faster-whisper model directory is missing weight files ({required}): {model_path}"
        )
    return True, None


class LocalModelCatalog:
    def __init__(self) -> None:
        """! @brief Initialize local model catalog state."""
        self._install_lock = threading.RLock()
        self._install_tasks: dict[str, LocalModelInstallTask] = {}

    def _catalog_path(self) -> Path:
        """! @brief Catalog path.
        @return Path result produced by the operation.
        """
        return SETTINGS.models_dir / "catalog.json"

    def list_all(self) -> LocalModelCatalogResponse:
        """! @brief List all operation.
        @return Result produced by the operation.
        """
        payload = self._load_payload()
        return LocalModelCatalogResponse(
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
        selected = next((item.model_id for item in models if item.installed), models[0].model_id if models else "")
        return LocalStageModelResponse(stage=stage, selected_model_id=selected, models=models)

    def resolve_model(self, stage: LocalModelStage, model_id: str | None = None) -> LocalModelRecord:
        """! @brief Resolve model.
        @param stage Value for stage.
        @param model_id Identifier of the target model.
        @return Result produced by the operation.
        """
        self._validate_stage_name(stage)
        wanted = (model_id or "").strip()
        if not wanted:
            raise ValueError(f"Local {stage} model id is required")
        payload = self._load_payload()
        models = [
            self._validate_record(LocalModelRecord.model_validate(item))
            for item in payload["models"]
            if item.get("stage") == stage
        ]
        if not models:
            raise ValueError(f"No local models are registered for stage '{stage}'")
        for item in models:
            if item.model_id == wanted:
                return item
        raise ValueError(f"Unknown local {stage} model: {wanted}")

    def register(self, request: LocalModelRegistrationRequest) -> LocalModelRecord:
        """! @brief Register operation.
        @param request Request payload for the operation.
        @return Result produced by the operation.
        """
        self._validate_stage_runtime(request.stage, request.location, request.runtime)
        payload = self._load_payload()
        candidate_id = self._normalize_model_id(
            request.model_id or self._suggest_model_id(request.stage, request.runtime, request.name)
        )
        if any(item.get("model_id") == candidate_id for item in payload["models"]):
            raise ValueError(f"Model id '{candidate_id}' is already registered")

        record = LocalModelRecord(
            model_id=candidate_id,
            stage=request.stage,
            location=request.location,
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
        self._write_payload(payload)
        return validated

    def delete(self, model_id: str) -> None:
        """! @brief Delete operation.
        @param model_id Identifier of the target model.
        """
        payload = self._load_payload()
        next_models = [item for item in payload["models"] if item.get("model_id") != model_id]
        if len(next_models) == len(payload["models"]):
            raise ValueError(f"Unknown local model: {model_id}")
        payload["models"] = next_models
        self._write_payload(payload)

    def runtime_descriptors(self) -> list[LocalModelRuntimeDescriptor]:
        """! @brief Describe model runtime forms supported by the catalog.
        @return List of descriptors consumed by the frontend.
        """
        return [
            LocalModelRuntimeDescriptor(
                stage="diarization",
                location="local",
                runtime="pyannote",
                label="pyannote",
                description=(
                    "Use an existing pyannote diarization pipeline. Hugging Face-gated models must "
                    "already be downloaded or available in your cache."
                ),
                fields=[
                    LocalModelFieldDescriptor(
                        key="pipeline_path",
                        label="Pipeline config path",
                        placeholder="/abs/path/to/config.yaml",
                        help="Absolute path to the local pyannote pipeline config.yaml file.",
                    ),
                    LocalModelFieldDescriptor(
                        key="embedding_model_ref",
                        label="Embedding model ref",
                        placeholder="pyannote/wespeaker-voxceleb-resnet34-LM",
                        help="Embedding model name or local path used for saved speaker profile matching.",
                    ),
                ],
            ),
            LocalModelRuntimeDescriptor(
                stage="transcription",
                location="local",
                runtime="whisper.cpp",
                label="whisper.cpp",
                description="Use a whisper.cpp CLI binary with a local GGUF transcription model.",
                fields=[
                    LocalModelFieldDescriptor(
                        key="binary_path",
                        label="whisper.cpp binary",
                        placeholder="/abs/path/to/whisper-cli",
                        help="Path or PATH-resolvable command for the whisper.cpp CLI executable.",
                    ),
                    LocalModelFieldDescriptor(
                        key="model_path",
                        label="GGUF model path",
                        placeholder="/abs/path/to/model.gguf",
                        help="Absolute path to the local GGUF model used by whisper.cpp.",
                    ),
                ],
            ),
            LocalModelRuntimeDescriptor(
                stage="transcription",
                location="local",
                runtime="faster-whisper",
                label="faster-whisper",
                description="Use an existing CTranslate2 faster-whisper model directory.",
                fields=[
                    LocalModelFieldDescriptor(
                        key="model_path",
                        label="Model directory",
                        placeholder="/abs/path/to/ctranslate2-model",
                        help="Directory containing the faster-whisper model files.",
                    ),
                    LocalModelFieldDescriptor(
                        key="compute_type",
                        label="Compute type",
                        required=False,
                        placeholder="auto | float16 | int8 | int8_float16",
                        help="Optional CTranslate2 compute type. Leave empty for automatic selection.",
                    ),
                ],
            ),
            LocalModelRuntimeDescriptor(
                stage="formatter",
                location="local",
                runtime="ollama",
                label="Ollama",
                description="Use a locally installed Ollama tag, or pull a new tag through Ollama.",
                supports_install=True,
                fields=[
                    LocalModelFieldDescriptor(
                        key="tag",
                        label="Ollama tag",
                        placeholder="qwen2.5:3b-instruct-q5_K_M",
                        help="The exact local Ollama model tag to use for MoM generation.",
                    ),
                ],
            ),
            LocalModelRuntimeDescriptor(
                stage="formatter",
                location="local",
                runtime="command",
                label="Command",
                description="Advanced formatter runtime for a custom local command template.",
                advanced=True,
                fields=[
                    LocalModelFieldDescriptor(
                        key="command_template",
                        label="Command template",
                        placeholder="bash -lc \"... {model} ...\"",
                        help="Command that AutoMoM runs for formatting. It must include the expected placeholders.",
                    ),
                    LocalModelFieldDescriptor(
                        key="model_path",
                        label="Model path",
                        placeholder="/abs/path/to/model.gguf",
                        help="Path to the local model file passed into the command runtime.",
                    ),
                ],
            ),
            LocalModelRuntimeDescriptor(
                stage="diarization",
                location="remote",
                runtime="pyannote",
                label="pyannote worker",
                description="Use a remote LAN worker that exposes pyannote diarization over HTTP.",
                fields=[
                    LocalModelFieldDescriptor(
                        key="base_url",
                        label="Worker base URL",
                        placeholder="http://office-gpu:8010",
                        help="Base URL of the remote diarization worker.",
                    ),
                    LocalModelFieldDescriptor(
                        key="model_name",
                        label="Worker model name",
                        placeholder="pyannote-community-1",
                        help="Model identifier reported by the remote worker health endpoint.",
                    ),
                    LocalModelFieldDescriptor(
                        key="profile_model_ref",
                        label="Profile model ref",
                        placeholder="pyannote-community-1",
                        help="Compatibility id used for local voice-profile matching across local and remote workers.",
                    ),
                    LocalModelFieldDescriptor(
                        key="embedding_model_ref",
                        label="Embedding model ref",
                        placeholder="pyannote/wespeaker-voxceleb-resnet34-LM",
                        help="Embedding model reference reported by the remote worker.",
                    ),
                    LocalModelFieldDescriptor(
                        key="auth_token",
                        label="Bearer token",
                        required=False,
                        placeholder="optional",
                        help="Optional bearer token sent to the remote worker.",
                        input_type="password",
                    ),
                    LocalModelFieldDescriptor(
                        key="timeout_s",
                        label="Timeout (seconds)",
                        required=False,
                        placeholder="900",
                        help="Optional request timeout for the remote worker.",
                    ),
                ],
            ),
            LocalModelRuntimeDescriptor(
                stage="transcription",
                location="remote",
                runtime="whisper.cpp",
                label="whisper.cpp worker",
                description="Use a remote LAN worker that exposes whisper.cpp transcription over HTTP.",
                fields=[
                    LocalModelFieldDescriptor(
                        key="base_url",
                        label="Worker base URL",
                        placeholder="http://office-gpu:8011",
                        help="Base URL of the remote transcription worker.",
                    ),
                    LocalModelFieldDescriptor(
                        key="model_name",
                        label="Worker model name",
                        placeholder="large-v3",
                        help="Model identifier reported by the remote worker health endpoint.",
                    ),
                    LocalModelFieldDescriptor(
                        key="auth_token",
                        label="Bearer token",
                        required=False,
                        placeholder="optional",
                        help="Optional bearer token sent to the remote worker.",
                        input_type="password",
                    ),
                    LocalModelFieldDescriptor(
                        key="timeout_s",
                        label="Timeout (seconds)",
                        required=False,
                        placeholder="900",
                        help="Optional request timeout for the remote worker.",
                    ),
                ],
            ),
            LocalModelRuntimeDescriptor(
                stage="formatter",
                location="remote",
                runtime="ollama",
                label="Remote Ollama",
                description="Use an Ollama server reachable over your LAN.",
                fields=[
                    LocalModelFieldDescriptor(
                        key="base_url",
                        label="Ollama base URL",
                        placeholder="http://office-gpu:11434",
                        help="Base URL of the remote Ollama host.",
                    ),
                    LocalModelFieldDescriptor(
                        key="tag",
                        label="Ollama tag",
                        placeholder="qwen2.5:3b-instruct-q5_K_M",
                        help="The exact Ollama model tag to use for MoM generation.",
                    ),
                    LocalModelFieldDescriptor(
                        key="auth_token",
                        label="Bearer token",
                        required=False,
                        placeholder="optional",
                        help="Optional bearer token sent to the remote Ollama host.",
                        input_type="password",
                    ),
                    LocalModelFieldDescriptor(
                        key="timeout_s",
                        label="Timeout (seconds)",
                        required=False,
                        placeholder="300",
                        help="Optional request timeout for the remote Ollama host.",
                    ),
                ],
            ),
        ]

    def discover(
        self,
        stage: LocalModelStage,
        location: LocalModelLocation,
        runtime: LocalModelRuntime,
    ) -> LocalModelDiscoveryResponse:
        """! @brief Discover likely local models for a supported runtime.
        @param stage Value for stage.
        @param runtime Value for runtime.
        @return Suggestions found in known safe locations.
        """
        self._validate_stage_runtime(stage, location, runtime)
        if location != "local":
            return LocalModelDiscoveryResponse(stage=stage, runtime=runtime, suggestions=[])
        suggestions = {
            "pyannote": self._discover_pyannote,
            "whisper.cpp": self._discover_whisper_cpp,
            "faster-whisper": self._discover_faster_whisper,
            "ollama": self._discover_ollama,
            "command": self._discover_command,
        }[runtime](stage, runtime)
        return LocalModelDiscoveryResponse(stage=stage, runtime=runtime, suggestions=suggestions[:50])

    def start_install(self, request: LocalModelInstallRequest) -> LocalModelInstallTask:
        """! @brief Start an install task for runtimes with managed install support.
        @param request Request payload for the operation.
        @return Install task record.
        """
        self._validate_stage_runtime(request.stage, request.location, request.runtime)
        if request.runtime != "ollama":
            raise ValueError(f"Auto-install is not available for runtime '{request.runtime}'")
        if request.location != "local":
            raise ValueError("Auto-install is only available for local Ollama models")
        tag = request.config.get("tag", "").strip()
        if not tag:
            raise ValueError("Ollama install requires tag")

        now = self._now()
        task = LocalModelInstallTask(
            task_id=uuid4().hex,
            stage=request.stage,
            runtime=request.runtime,
            status="queued",
            message=f"Queued Ollama pull for {tag}",
            percent=0.0,
            created_at=now,
            updated_at=now,
        )
        with self._install_lock:
            self._install_tasks[task.task_id] = task

        thread = threading.Thread(target=self._run_ollama_install, args=(task.task_id, request), daemon=True)
        thread.start()
        return task

    def get_install_task(self, task_id: str) -> LocalModelInstallTask:
        """! @brief Get install task.
        @param task_id Identifier of the task.
        @return Install task.
        """
        with self._install_lock:
            task = self._install_tasks.get(task_id)
        if not task:
            raise KeyError(task_id)
        return task

    def list_install_tasks(self) -> list[LocalModelInstallTask]:
        """! @brief List install tasks.
        @return Task records.
        """
        with self._install_lock:
            return sorted(self._install_tasks.values(), key=lambda item: item.created_at, reverse=True)

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

    def _load_payload(self) -> dict[str, object]:
        """! @brief Load payload.
        @return Dictionary produced by the operation.
        """
        self._ensure_seeded()
        try:
            models = json.loads(self._catalog_path().read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid local model catalog JSON: {exc}") from exc

        normalized_models = [
            LocalModelRecord.model_validate(item).model_dump(mode="json")
            for item in models
        ]
        normalized_models, repaired = self._repair_seeded_default_paths(normalized_models)
        if repaired:
            self._write_payload({"models": normalized_models})
        return {"models": normalized_models}

    def _write_payload(self, payload: dict[str, object]) -> None:
        """! @brief Write payload.
        @param payload Value for payload.
        """
        self._catalog_path().parent.mkdir(parents=True, exist_ok=True)
        self._catalog_path().write_text(json.dumps(payload["models"], indent=2), encoding="utf-8")

    def _ensure_seeded(self) -> None:
        """! @brief Ensure seeded.
        """
        if self._catalog_path().exists():
            return
        seeded = self._seed_payload()
        self._write_payload(seeded)

    def _seed_payload(self) -> dict[str, object]:
        """! @brief Seed payload.
        @return Dictionary produced by the operation.
        """
        formatter_runtime: LocalModelRuntime = "command" if SETTINGS.formatter_backend == "command" else "ollama"
        formatter_model_id = "formatter-command-default" if formatter_runtime == "command" else "formatter-ollama-default"
        formatter_tag = SETTINGS.formatter_ollama_model
        models = [
            LocalModelRecord(
                model_id="pyannote-community-1",
                stage="diarization",
                location="local",
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
                location="local",
                runtime="whisper.cpp",
                name="whisper.cpp local ASR",
                installed=False,
                languages=["multilingual"],
                config={
                    "binary_path": SETTINGS.transcription_binary,
                    "model_path": SETTINGS.transcription_model_path,
                },
            ),
            LocalModelRecord(
                model_id=formatter_model_id,
                stage="formatter",
                location="local",
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
        return {"models": [self._validate_record(item).model_dump(mode="json") for item in models]}

    def _validate_record(self, record: LocalModelRecord) -> LocalModelRecord:
        """! @brief Validate record.
        @param record Value for record.
        @return Result produced by the operation.
        """
        validator = {
            ("local", "pyannote"): self._validate_pyannote,
            ("local", "whisper.cpp"): self._validate_whisper_cpp,
            ("local", "faster-whisper"): self._validate_faster_whisper,
            ("local", "ollama"): self._validate_ollama,
            ("local", "command"): self._validate_command,
            ("remote", "pyannote"): self._validate_remote_pyannote,
            ("remote", "whisper.cpp"): self._validate_remote_whisper_cpp,
            ("remote", "ollama"): self._validate_remote_ollama,
        }.get((record.location, record.runtime))
        if validator is None:
            return record.model_copy(
                update={
                    "installed": False,
                    "validation_error": f"Unsupported {record.location} runtime for {record.stage}: {record.runtime}",
                }
            )
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
            if record.model_id == "pyannote-community-1" and record.location == "local" and record.runtime == "pyannote":
                candidate = SETTINGS.diarization_pipeline_path or SETTINGS.diarization_model_path
                if self._should_repair_path(updated_config.get("pipeline_path", ""), candidate):
                    updated_config["pipeline_path"] = candidate
            elif record.model_id == "whispercpp-local" and record.location == "local" and record.runtime == "whisper.cpp":
                if self._should_repair_path(updated_config.get("binary_path", ""), SETTINGS.transcription_binary):
                    updated_config["binary_path"] = SETTINGS.transcription_binary
                if self._should_repair_path(updated_config.get("model_path", ""), SETTINGS.transcription_model_path):
                    updated_config["model_path"] = SETTINGS.transcription_model_path
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

    def _validate_remote_pyannote(self, config: dict[str, str]) -> tuple[bool, str | None]:
        base_url = config.get("base_url", "").strip()
        model_name = config.get("model_name", "").strip()
        profile_model_ref = config.get("profile_model_ref", "").strip()
        embedding_ref = config.get("embedding_model_ref", "").strip()
        if not base_url:
            return False, "Remote pyannote model requires base_url"
        if not model_name:
            return False, "Remote pyannote model requires model_name"
        if not profile_model_ref:
            return False, "Remote pyannote model requires profile_model_ref"
        if not embedding_ref:
            return False, "Remote pyannote model requires embedding_model_ref"
        health, error = self._remote_worker_health(base_url, config.get("auth_token", ""), config.get("timeout_s", ""), "diarization")
        if error:
            return False, error
        remote_name = str((health or {}).get("diarization", {}).get("model_name", "")).strip()
        if remote_name and remote_name != model_name:
            return False, f"Remote diarization worker model mismatch: expected '{model_name}', got '{remote_name}'"
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

    def _validate_remote_whisper_cpp(self, config: dict[str, str]) -> tuple[bool, str | None]:
        base_url = config.get("base_url", "").strip()
        model_name = config.get("model_name", "").strip()
        if not base_url:
            return False, "Remote whisper.cpp model requires base_url"
        if not model_name:
            return False, "Remote whisper.cpp model requires model_name"
        health, error = self._remote_worker_health(base_url, config.get("auth_token", ""), config.get("timeout_s", ""), "transcription")
        if error:
            return False, error
        transcription_payload = (health or {}).get("transcription", {})
        runtime_name = str(transcription_payload.get("runtime", "")).strip().lower()
        remote_name = str(transcription_payload.get("model_name", "")).strip()
        if runtime_name and runtime_name != "whisper.cpp":
            return False, f"Remote transcription worker runtime mismatch: expected 'whisper.cpp', got '{runtime_name}'"
        if remote_name and remote_name != model_name:
            return False, f"Remote transcription worker model mismatch: expected '{model_name}', got '{remote_name}'"
        return True, None

    def _validate_faster_whisper(self, config: dict[str, str]) -> tuple[bool, str | None]:
        """! @brief Validate faster whisper.
        @param config Value for config.
        @return Tuple produced by the operation.
        """
        model_path = config.get("model_path", "").strip()
        if not model_path:
            return False, "faster-whisper model requires model_path"
        valid_model_dir, model_dir_error = validate_faster_whisper_model_directory(Path(model_path))
        if not valid_model_dir:
            return False, model_dir_error
        compute_type = config.get("compute_type", "").strip()
        normalized_compute_type = compute_type.lower() if compute_type else "auto"
        if normalized_compute_type not in VALID_FASTER_WHISPER_COMPUTE_TYPES:
            allowed = ", ".join(sorted(VALID_FASTER_WHISPER_COMPUTE_TYPES))
            return False, f"Invalid faster-whisper compute_type '{compute_type}'. Allowed values: {allowed}"
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

    def _validate_remote_ollama(self, config: dict[str, str]) -> tuple[bool, str | None]:
        tag = config.get("tag", "").strip()
        base_url = config.get("base_url", "").strip()
        if not base_url:
            return False, "Remote Ollama model requires base_url"
        if not tag:
            return False, "Remote Ollama model requires tag"
        if not self._ollama_has_model(tag, base_url=base_url, auth_token=config.get("auth_token", ""), timeout_s=config.get("timeout_s", "")):
            return False, f"Remote Ollama tag is not available on {base_url}: {tag}"
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

    def _discover_pyannote(
        self,
        stage: LocalModelStage,
        runtime: LocalModelRuntime,
    ) -> list[LocalModelDiscoverySuggestion]:
        """! @brief Discover pyannote pipeline configs."""
        paths = [
            SETTINGS.diarization_pipeline_path,
            SETTINGS.diarization_model_path,
        ]
        paths.extend(str(path) for path in self._iter_known_files([SETTINGS.models_dir / "diarization"], "config.yaml"))
        paths.extend(str(path) for path in self._iter_known_files(self._hf_cache_roots(), "config.yaml", limit=40))
        suggestions: list[LocalModelDiscoverySuggestion] = []
        for path in self._dedupe_existing_paths(paths):
            suggestions.append(
                LocalModelDiscoverySuggestion(
                    stage=stage,
                    runtime=runtime,
                    name=self._display_name_from_path(path, "Pyannote diarization"),
                    source="local scan",
                    details=str(path),
                    config={
                        "pipeline_path": str(path),
                        "embedding_model_ref": SETTINGS.diarization_embedding_model,
                    },
                )
            )
        return suggestions

    def _discover_whisper_cpp(
        self,
        stage: LocalModelStage,
        runtime: LocalModelRuntime,
    ) -> list[LocalModelDiscoverySuggestion]:
        """! @brief Discover whisper.cpp binaries and GGUF model files."""
        repo_root = Path(__file__).resolve().parents[2]
        binary_paths = [
            SETTINGS.transcription_binary,
            str(repo_root / "tools" / "whisper.cpp" / "build-cuda" / "bin" / "whisper-cli"),
            str(repo_root / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
            shutil.which("whisper-cli") or "",
        ]
        binaries = self._dedupe_existing_paths(binary_paths)
        model_paths = [SETTINGS.transcription_model_path]
        model_paths.extend(str(path) for path in self._iter_known_files([SETTINGS.models_dir / "transcription"], "*.gguf"))
        model_paths.extend(str(path) for path in self._iter_known_files([SETTINGS.models_dir / "voxtral"], "*.gguf"))
        model_paths.extend(str(path) for path in self._iter_known_files(self._hf_cache_roots(), "*.gguf", limit=50))
        models = self._dedupe_existing_paths(model_paths)

        suggestions: list[LocalModelDiscoverySuggestion] = []
        for model_path in models:
            binary_path = str(binaries[0]) if binaries else SETTINGS.transcription_binary
            suggestions.append(
                LocalModelDiscoverySuggestion(
                    stage=stage,
                    runtime=runtime,
                    name=self._display_name_from_path(model_path, "whisper.cpp"),
                    source="local scan",
                    details=f"{model_path}",
                    config={"binary_path": binary_path, "model_path": str(model_path)},
                )
            )
        return suggestions

    def _discover_faster_whisper(
        self,
        stage: LocalModelStage,
        runtime: LocalModelRuntime,
    ) -> list[LocalModelDiscoverySuggestion]:
        """! @brief Discover faster-whisper model directories."""
        roots = [
            SETTINGS.models_dir / "transcription",
            SETTINGS.models_dir / "faster-whisper",
            *self._hf_cache_roots(),
        ]
        candidates: list[str] = []
        for path in self._iter_known_files(roots, "config.json", limit=50):
            candidates.append(str(path.parent))
        for path in self._iter_known_files(roots, "model.bin", limit=50):
            candidates.append(str(path.parent))

        suggestions: list[LocalModelDiscoverySuggestion] = []
        for path in self._dedupe_existing_paths(candidates):
            suggestions.append(
                LocalModelDiscoverySuggestion(
                    stage=stage,
                    runtime=runtime,
                    name=self._display_name_from_path(path, "faster-whisper"),
                    source="local scan",
                    details=str(path),
                    config={"model_path": str(path), "compute_type": "auto"},
                )
            )
        return suggestions

    def _discover_ollama(
        self,
        stage: LocalModelStage,
        runtime: LocalModelRuntime,
    ) -> list[LocalModelDiscoverySuggestion]:
        """! @brief Discover local Ollama model tags."""
        suggestions: list[LocalModelDiscoverySuggestion] = []
        for tag in self._ollama_tags():
            suggestions.append(
                LocalModelDiscoverySuggestion(
                    stage=stage,
                    runtime=runtime,
                    name=tag,
                    source="ollama",
                    details=f"{SETTINGS.ollama_host.rstrip('/')}/api/tags",
                    config={"tag": tag},
                )
            )
        return suggestions

    def _discover_command(
        self,
        stage: LocalModelStage,
        runtime: LocalModelRuntime,
    ) -> list[LocalModelDiscoverySuggestion]:
        """! @brief Suggest current command formatter settings when configured."""
        if not SETTINGS.formatter_command.strip() or not Path(SETTINGS.formatter_model_path).expanduser().exists():
            return []
        return [
            LocalModelDiscoverySuggestion(
                stage=stage,
                runtime=runtime,
                name="Current command formatter",
                source="settings",
                details=SETTINGS.formatter_model_path,
                config={
                    "command_template": SETTINGS.formatter_command,
                    "model_path": SETTINGS.formatter_model_path,
                },
            )
        ]

    def _run_ollama_install(self, task_id: str, request: LocalModelInstallRequest) -> None:
        """! @brief Pull an Ollama tag and register it when available."""
        tag = request.config.get("tag", "").strip()
        self._update_install_task(task_id, status="running", message=f"Checking Ollama tag {tag}", percent=2.0)
        try:
            if not self._ollama_has_model(tag):
                self._pull_ollama_tag(tag, lambda message, percent: self._update_install_task(
                    task_id,
                    status="running",
                    message=message,
                    percent=percent,
                ))
            self._update_install_task(task_id, status="running", message=f"Registering {tag}", percent=98.0)
            record = self.register(
                LocalModelRegistrationRequest(
                    stage=request.stage,
                    location="local",
                    runtime=request.runtime,
                    name=request.name.strip() or tag,
                    languages=request.languages,
                    notes=request.notes,
                    config={"tag": tag},
                )
            )
            self._update_install_task(
                task_id,
                status="completed",
                message=f"Installed and registered {tag}",
                percent=100.0,
                model_id=record.model_id,
            )
        except Exception as exc:  # pragma: no cover - thread guard
            self._update_install_task(
                task_id,
                status="failed",
                message=f"Unable to install {tag}",
                error=str(exc),
            )

    def _pull_ollama_tag(self, tag: str, progress_callback) -> None:
        """! @brief Pull an Ollama model tag with streaming progress."""
        body = json.dumps({"name": tag, "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            url=f"{SETTINGS.ollama_host.rstrip('/')}/api/pull",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                last_percent = 5.0
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line.decode("utf-8", errors="replace"))
                    message = str(payload.get("status") or f"Pulling {tag}")
                    total = int(payload.get("total") or 0)
                    completed = int(payload.get("completed") or 0)
                    if total > 0:
                        last_percent = min(97.0, max(last_percent, completed / total * 95.0))
                    progress_callback(message, last_percent)
                    if payload.get("error"):
                        raise RuntimeError(str(payload["error"]))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama pull failed for {tag}: {exc}") from exc

    def _update_install_task(self, task_id: str, **updates: object) -> None:
        """! @brief Update install task state."""
        with self._install_lock:
            task = self._install_tasks[task_id]
            self._install_tasks[task_id] = task.model_copy(
                update={**updates, "updated_at": self._now()}
            )

    def _ollama_tags(
        self,
        *,
        base_url: str | None = None,
        auth_token: str = "",
        timeout_s: str = "",
    ) -> list[str]:
        """! @brief List local Ollama tags."""
        headers = self._auth_headers(auth_token)
        timeout = self._parse_timeout(timeout_s, default=5)
        host = (base_url or SETTINGS.ollama_host).rstrip("/")
        request = urllib.request.Request(url=f"{host}/api/tags", method="GET", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return []
        return [
            str(item.get("name", "")).strip()
            for item in payload.get("models", [])
            if str(item.get("name", "")).strip()
        ]

    def _ollama_has_model(
        self,
        model_tag: str,
        *,
        base_url: str | None = None,
        auth_token: str = "",
        timeout_s: str = "",
    ) -> bool:
        """! @brief Ollama has model.
        @param model_tag Value for model tag.
        @return True when the requested condition is satisfied; otherwise False.
        """
        wanted = model_tag.strip().lower()
        for tag in self._ollama_tags(base_url=base_url, auth_token=auth_token, timeout_s=timeout_s):
            if tag.strip().lower() == wanted:
                return True
        return False

    def _remote_worker_health(
        self,
        base_url: str,
        auth_token: str,
        timeout_s: str,
        required_stage: str,
    ) -> tuple[dict[str, object] | None, str | None]:
        request = urllib.request.Request(
            url=f"{base_url.rstrip('/')}/health",
            method="GET",
            headers=self._auth_headers(auth_token),
        )
        try:
            with urllib.request.urlopen(request, timeout=self._parse_timeout(timeout_s, default=10)) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return None, f"Remote worker auth failed: {base_url}"
            return None, f"Remote worker request failed ({exc.code}): {base_url}"
        except (urllib.error.URLError, TimeoutError):
            return None, f"Remote worker unreachable: {base_url}"
        except json.JSONDecodeError:
            return None, f"Remote worker returned invalid health payload: {base_url}"

        if required_stage not in set(payload.get("enabled_stages", [])):
            return None, f"Remote worker does not support {required_stage}: {base_url}"
        return payload, None

    @staticmethod
    def _parse_timeout(raw_timeout: str, *, default: int) -> int:
        try:
            return max(1, int(float(raw_timeout.strip()))) if raw_timeout.strip() else default
        except (TypeError, ValueError, AttributeError):
            return default

    @staticmethod
    def _auth_headers(auth_token: str) -> dict[str, str]:
        token = auth_token.strip()
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _iter_known_files(roots: list[Path], pattern: str, limit: int = 25) -> list[Path]:
        """! @brief Iterate matching files under known roots."""
        matches: list[Path] = []
        for root in roots:
            root = root.expanduser()
            if not root.exists():
                continue
            if root.is_file() and root.match(pattern):
                matches.append(root)
            elif root.is_dir():
                for path in root.rglob(pattern):
                    if path.is_file():
                        matches.append(path)
                    if len(matches) >= limit:
                        return matches
        return matches

    @staticmethod
    def _hf_cache_roots() -> list[Path]:
        """! @brief Return known Hugging Face cache roots without broad home scanning."""
        roots = []
        for env_name in ("HF_HOME", "TRANSFORMERS_CACHE"):
            value = os.getenv(env_name, "").strip()
            if value:
                roots.append(Path(value))
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        return roots

    @staticmethod
    def _dedupe_existing_paths(paths: list[str]) -> list[Path]:
        """! @brief Deduplicate existing filesystem paths."""
        seen: set[str] = set()
        result: list[Path] = []
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if not path.exists():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
        return result

    @staticmethod
    def _display_name_from_path(path: Path, fallback: str) -> str:
        """! @brief Create a human name from a model path."""
        name = path.parent.name if path.name == "config.yaml" else path.stem if path.is_file() else path.name
        return name or fallback

    @staticmethod
    def _now() -> datetime:
        """! @brief Current UTC time."""
        return datetime.now(timezone.utc)

    def _resolve_binary(self, binary_path: str) -> str | None:
        """! @brief Resolve binary.
        @param binary_path Value for binary path.
        @return Result produced by the operation.
        """
        candidate = Path(binary_path).expanduser()
        if candidate.exists():
            return str(candidate)
        return shutil.which(binary_path)

    def _validate_stage_runtime(
        self,
        stage: LocalModelStage,
        location: LocalModelLocation,
        runtime: LocalModelRuntime,
    ) -> None:
        """! @brief Validate stage runtime.
        @param stage Value for stage.
        @param location Value for location.
        @param runtime Value for runtime.
        """
        self._validate_stage_name(stage)
        allowed = RUNTIMES_BY_STAGE_LOCATION.get((stage, location), ())
        if runtime not in allowed:
            raise ValueError(f"Runtime '{runtime}' is not supported for stage '{stage}' with location '{location}'")

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
