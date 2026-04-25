from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
import subprocess
from pathlib import Path
import shutil
from typing import Callable, Protocol
from functools import lru_cache

from backend.pipeline.compute import native_cuda_available, should_enable_native_gpu
from backend.pipeline.diarization import merge_transcript_segments
from backend.pipeline.openai_client import OpenAIAPIError, OpenAIClient
from backend.pipeline.remote_worker_client import RemoteWorkerClient, RemoteWorkerError
from backend.pipeline.subprocess_utils import run_cancellable_subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]

TIMESTAMP_RANGE_PATTERN = re.compile(
    r"(?:\[\s*)?\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\s*-->\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?(?:\s*\])?"
)
TIMESTAMP_BRACKET_PATTERN = re.compile(r"\[\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\s*\]")
TIMESTAMP_TOKEN_PATTERN = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\b")
WHISPER_TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")


class TranscriptionError(RuntimeError):
    pass


class LocalTranscriber(Protocol):
    def available(self) -> bool:
        """! @brief Available operation.
        @return True when the requested condition is satisfied; otherwise False.
        """

    def transcribe(self, segment_path: Path) -> str:
        """! @brief Transcribe operation.
        @param segment_path Path to the segment audio file.
        @return str result produced by the operation.
        """

    def runtime_report(self) -> dict[str, object]:
        """! @brief Runtime report.
        @return Dictionary produced by the operation.
        """

    def runtime_summary(self) -> str:
        """! @brief Runtime summary.
        @return str result produced by the operation.
        """


@dataclass(frozen=True)
class ASRBinaryCapabilities:
    binary_path: str
    is_whisper_cli: bool
    supported_flags: tuple[str, ...]
    linked_backends: tuple[str, ...]
    gpu_backend_available: bool

    def to_dict(self) -> dict[str, object]:
        """! @brief Serialize the current object to a dictionary.
        @return Dictionary produced by the operation.
        """
        return asdict(self)


@dataclass
class ASRRuntimeReport:
    binary_path: str
    model_path: str
    requested_mode: str
    available_mode: str
    active_mode: str
    gpu_requested: bool
    gpu_backend_available: bool
    gpu_verified_active: bool
    supported_flags: list[str]
    linked_backends: list[str]
    thread_count: int
    processor_count: int
    gpu_retry_disabled: bool = False
    last_error: str = ""

    def to_dict(self) -> dict[str, object]:
        """! @brief Serialize the current object to a dictionary.
        @return Dictionary produced by the operation.
        """
        return asdict(self)


class WhisperCppTranscriber:
    def __init__(
        self,
        binary_path: str | None,
        model_path: str | None,
        *,
        job_id: str | None = None,
        compute_device: str = "auto",
        cuda_device_id: int = 0,
        gpu_layers: int = 99,
        threads: int = 4,
        processors: int = 1,
    ) -> None:
        """! @brief Initialize the WhisperCppTranscriber instance.
        @param binary_path Value for binary path.
        @param model_path Value for model path.
        @param job_id Identifier of the job being processed.
        @param compute_device Requested compute device preference.
        @param cuda_device_id CUDA device index to prefer when GPU execution is enabled.
        @param gpu_layers Value for gpu layers.
        @param threads Value for threads.
        @param processors Value for processors.
        """
        self.binary_path = binary_path or ""
        self.model_path = model_path or ""
        self._job_id = job_id
        self._resolved_binary_path = self._resolve_preferred_binary_path(self.binary_path)
        self._cuda_device_id = max(0, int(cuda_device_id))
        self._gpu_layers = max(0, int(gpu_layers))
        self._threads = max(1, int(threads))
        self._processors = max(1, int(processors))
        self._requested_mode = (compute_device or "auto").strip().lower() or "auto"
        self._gpu_requested = should_enable_native_gpu(compute_device, self._cuda_device_id)
        self._gpu_retry_disabled = False
        self._runtime_available = bool(
            self._resolved_binary_path and self.model_path and Path(self.model_path).exists()
        )
        self._capabilities = _probe_asr_binary(self._resolved_binary_path or "")
        self._runtime_report = ASRRuntimeReport(
            binary_path=self._resolved_binary_path or "",
            model_path=self.model_path,
            requested_mode=self._requested_mode,
            available_mode=self._available_mode(),
            active_mode="cpu" if self._runtime_available else "fallback",
            gpu_requested=self._gpu_requested,
            gpu_backend_available=self._capabilities.gpu_backend_available,
            gpu_verified_active=False,
            supported_flags=list(self._capabilities.supported_flags),
            linked_backends=list(self._capabilities.linked_backends),
            thread_count=self._threads,
            processor_count=self._processors,
        )

    def available(self) -> bool:
        """! @brief Available operation.
        @return True when the requested condition is satisfied; otherwise False.
        """
        return self._runtime_available

    def transcribe(self, segment_path: Path) -> str:
        """! @brief Transcribe operation.
        @param segment_path Path to the segment audio file.
        @return str result produced by the operation.
        """
        if not self._runtime_available or not self._resolved_binary_path:
            raise TranscriptionError(self._missing_runtime_message())

        use_gpu = self._gpu_requested and not self._gpu_retry_disabled
        command = self._build_command(segment_path, use_gpu=use_gpu)
        process = run_cancellable_subprocess(command, job_id=self._job_id)
        invocation_failed = _invocation_failed(process)
        self._record_runtime_result(process, requested_gpu=use_gpu)
        if invocation_failed and use_gpu:
            # If GPU flags are unsupported, retry once on CPU and keep running.
            retry_command = self._build_command(segment_path, use_gpu=False)
            retry_process = run_cancellable_subprocess(retry_command, job_id=self._job_id)
            if not _invocation_failed(retry_process):
                process = retry_process
                invocation_failed = False
                self._gpu_retry_disabled = True
                self._record_runtime_result(retry_process, requested_gpu=False)
        if invocation_failed:
            hint = (
                "Ensure AUTOMOM_TRANSCRIPTION_BIN points to a working whisper.cpp CLI binary "
                "and AUTOMOM_TRANSCRIPTION_MODEL points to a compatible model file."
            )
            error_text = (process.stderr or process.stdout or "").strip()
            self._runtime_report.last_error = error_text
            raise TranscriptionError(f"ASR invocation failed for '{segment_path.name}': {error_text}. {hint}")
        raw_text = process.stdout if process.stdout.strip() else process.stderr
        text = clean_transcript_text(raw_text)
        if not text:
            raise TranscriptionError(
                f"ASR produced empty output for '{segment_path.name}'. "
                "Verify binary/model compatibility and command arguments."
            )
        return text

    def compute_mode(self) -> str:
        """! @brief Compute mode.
        @return str result produced by the operation.
        """
        if not self._runtime_available:
            return "fallback"
        if self._runtime_report.gpu_verified_active:
            return "cuda"
        if self._gpu_requested and self._gpu_retry_disabled:
            return "cpu(gpu_retry_disabled)"
        if self._gpu_requested and not self._capabilities.gpu_backend_available:
            return "cpu(gpu_backend_unavailable)"
        return "cpu"

    def runtime_report(self) -> dict[str, object]:
        """! @brief Runtime report.
        @return Dictionary produced by the operation.
        """
        self._runtime_report.available_mode = self._available_mode()
        self._runtime_report.gpu_retry_disabled = self._gpu_retry_disabled
        return self._runtime_report.to_dict()

    def runtime_summary(self) -> str:
        """! @brief Runtime summary.
        @return str result produced by the operation.
        """
        report = self.runtime_report()
        active_mode = str(report["active_mode"])
        if bool(report["gpu_verified_active"]):
            return "compute=cuda (verified active)"
        if bool(report["gpu_requested"]) and not bool(report["gpu_backend_available"]):
            return "compute=cpu (GPU backend unavailable in ASR binary)"
        if bool(report["gpu_requested"]) and bool(report["gpu_retry_disabled"]):
            return "compute=cpu (GPU retry disabled after runtime fallback)"
        if self._requested_mode == "cpu" or not bool(report["gpu_requested"]):
            return "compute=cpu (GPU disabled by config)"
        return f"compute={active_mode} (GPU requested but not verified active)"

    def _build_command(self, segment_path: Path, *, use_gpu: bool) -> list[str]:
        """! @brief Build command.
        @param segment_path Path to the segment audio file.
        @param use_gpu Value for use gpu.
        @return List produced by the operation.
        """
        command = [self._resolved_binary_path, "-m", self.model_path, "-f", str(segment_path)]
        if self._capabilities.is_whisper_cli:
            if _binary_supports_any_flag(self._resolved_binary_path, ("-t", "--threads")):
                command.extend(["-t", str(self._threads)])
            if _binary_supports_any_flag(self._resolved_binary_path, ("-p", "--processors")):
                command.extend(["-p", str(self._processors)])
        if not use_gpu or not self._resolved_binary_path:
            return command
        if not self._capabilities.is_whisper_cli or not self._capabilities.gpu_backend_available:
            return command

        # Probe the binary first so we only pass GPU flags that this specific whisper.cpp build
        # actually exposes; CUDA-enabled binaries are not consistent about option names.
        tokens = set(command)
        supports_gpu_layers = _binary_supports_any_flag(
            self._resolved_binary_path,
            ("-ngl", "--gpu-layers", "--n-gpu-layers"),
        )
        supports_device = _binary_supports_any_flag(
            self._resolved_binary_path,
            ("-dev", "--device"),
        )

        if (
            supports_gpu_layers
            and self._gpu_layers > 0
            and "-ngl" not in tokens
            and "--gpu-layers" not in tokens
            and "--n-gpu-layers" not in tokens
        ):
            command.extend(["-ngl", str(self._gpu_layers)])
        if supports_device and "-dev" not in tokens and "--device" not in tokens:
            command.extend(["-dev", str(self._cuda_device_id)])
        return command

    @staticmethod
    def _resolve_binary_path(binary_path: str) -> str | None:
        """! @brief Resolve binary path.
        @param binary_path Value for binary path.
        @return Result produced by the operation.
        """
        if not binary_path:
            return None
        as_path = Path(binary_path)
        if as_path.exists():
            return str(as_path)
        which_path = shutil.which(binary_path)
        if which_path:
            return which_path
        return None

    @classmethod
    def _resolve_preferred_binary_path(cls, binary_path: str) -> str | None:
        """! @brief Resolve preferred binary path.
        @param binary_path Value for binary path.
        @return Result produced by the operation.
        """
        configured = cls._resolve_binary_path(binary_path)
        repo_cuda = cls._resolve_binary_path(str(REPO_ROOT / "tools" / "whisper.cpp" / "build-cuda" / "bin" / "whisper-cli"))
        repo_cpu = cls._resolve_binary_path(str(REPO_ROOT / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"))
        path_binary = cls._resolve_binary_path("whisper-cli")

        if configured:
            configured_caps = _probe_asr_binary(configured)
            if configured_caps.gpu_backend_available:
                return configured

        if repo_cuda and repo_cuda != configured and _probe_asr_binary(repo_cuda).gpu_backend_available:
            return repo_cuda

        for candidate in (configured, repo_cpu, path_binary):
            if candidate:
                return candidate

        return None

    def _missing_runtime_message(self) -> str:
        """! @brief Missing runtime message.
        @return str result produced by the operation.
        """
        missing_parts: list[str] = []
        if not self._resolved_binary_path:
            missing_parts.append(
                "ASR binary is not configured or not found. "
                "Set AUTOMOM_TRANSCRIPTION_BIN to a valid whisper.cpp CLI executable."
            )
        if not self.model_path or not Path(self.model_path).exists():
            missing_parts.append(
                "ASR model file is missing. "
                "Set AUTOMOM_TRANSCRIPTION_MODEL to an existing local model file."
            )
        if not missing_parts:
            missing_parts.append("ASR runtime is unavailable.")
        return " ".join(missing_parts)

    def _available_mode(self) -> str:
        """! @brief Available mode.
        @return str result produced by the operation.
        """
        if not self._runtime_available:
            return "fallback"
        if self._gpu_requested and self._capabilities.gpu_backend_available:
            return "cuda"
        return "cpu"

    def _record_runtime_result(self, process: subprocess.CompletedProcess[str], *, requested_gpu: bool) -> None:
        """! @brief Record runtime result.
        @param process Value for process.
        @param requested_gpu Value for requested gpu.
        """
        combined_output = "\n".join(part for part in (process.stdout, process.stderr) if part).strip()
        verified_gpu = _gpu_verified_active(combined_output)
        active_mode = "cuda" if verified_gpu else "cpu"
        self._runtime_report.active_mode = active_mode
        self._runtime_report.gpu_verified_active = verified_gpu
        self._runtime_report.gpu_retry_disabled = self._gpu_retry_disabled
        if requested_gpu and not verified_gpu and self._capabilities.gpu_backend_available:
            self._runtime_report.last_error = _runtime_gpu_failure_reason(combined_output)
        elif requested_gpu and not self._capabilities.gpu_backend_available:
            self._runtime_report.last_error = "gpu_backend_unavailable"
        elif not requested_gpu:
            self._runtime_report.last_error = ""


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_path: str | None,
        *,
        compute_device: str = "auto",
        cuda_device_id: int = 0,
        compute_type: str = "auto",
    ) -> None:
        """! @brief Initialize the FasterWhisperTranscriber instance.
        @param model_path Value for model path.
        @param compute_device Requested compute device preference.
        @param cuda_device_id CUDA device index to prefer when GPU execution is enabled.
        @param compute_type Value for compute type.
        """
        self.model_path = model_path or ""
        self._requested_mode = (compute_device or "auto").strip().lower() or "auto"
        self._cuda_device_id = max(0, int(cuda_device_id))
        self._gpu_requested = self._requested_mode != "cpu"
        self._cuda_available = native_cuda_available(self._cuda_device_id)
        self._gpu_enabled = should_enable_native_gpu(compute_device, self._cuda_device_id)
        self._cuda_required_unavailable = self._requested_mode == "cuda" and not self._cuda_available
        self._compute_type = (compute_type or "auto").strip() or "auto"
        self._runtime_available = bool(
            self.model_path
            and Path(self.model_path).exists()
            and _faster_whisper_available()
            and not self._cuda_required_unavailable
        )
        self._device = "cuda" if self._gpu_enabled else "cpu"
        initial_error = "cuda_unavailable" if self._cuda_required_unavailable else ""
        self._model = None
        self._runtime_report = ASRRuntimeReport(
            binary_path="faster-whisper",
            model_path=self.model_path,
            requested_mode=self._requested_mode,
            available_mode="cuda" if self._gpu_enabled and self._runtime_available else ("cpu" if self._runtime_available else "fallback"),
            active_mode="cpu" if self._runtime_available else "fallback",
            gpu_requested=self._gpu_requested,
            gpu_backend_available=self._cuda_available,
            gpu_verified_active=False,
            supported_flags=[],
            linked_backends=["cuda", "cpu"] if self._cuda_available else ["cpu"],
            thread_count=0,
            processor_count=0,
            last_error=initial_error,
        )

    def available(self) -> bool:
        """! @brief Available operation.
        @return True when the requested condition is satisfied; otherwise False.
        """
        return self._runtime_available

    def transcribe(self, segment_path: Path) -> str:
        """! @brief Transcribe operation.
        @param segment_path Path to the segment audio file.
        @return str result produced by the operation.
        """
        if not self._runtime_available:
            raise TranscriptionError(self._missing_runtime_message())
        model = self._load_model()
        try:
            segments, _info = model.transcribe(str(segment_path), beam_size=1, vad_filter=False)
        except Exception as exc:
            self._runtime_report.last_error = str(exc)
            raise TranscriptionError(f"faster-whisper transcription failed for '{segment_path.name}': {exc}") from exc

        text = clean_transcript_text(" ".join(segment.text for segment in segments))
        if not text:
            # Short segments can occasionally return empty output with beam_size=1.
            # Retry once with a more robust decode setup before failing the job.
            try:
                fallback_segments, _info = model.transcribe(str(segment_path), beam_size=5, vad_filter=False)
            except Exception as exc:
                self._runtime_report.last_error = str(exc)
                raise TranscriptionError(f"faster-whisper transcription failed for '{segment_path.name}': {exc}") from exc
            text = clean_transcript_text(" ".join(segment.text for segment in fallback_segments))
        if not text:
            self._runtime_report.last_error = "blank_audio_segment"
            self._runtime_report.active_mode = self._device
            self._runtime_report.gpu_verified_active = self._device == "cuda"
            # Keep pipeline behavior aligned with whisper.cpp, which surfaces short
            # no-speech chunks as [BLANK_AUDIO] instead of failing the whole job.
            return "[BLANK_AUDIO]"
        self._runtime_report.active_mode = self._device
        self._runtime_report.gpu_verified_active = self._device == "cuda"
        self._runtime_report.last_error = ""
        return text

    def runtime_report(self) -> dict[str, object]:
        """! @brief Runtime report.
        @return Dictionary produced by the operation.
        """
        return self._runtime_report.to_dict()

    def runtime_summary(self) -> str:
        """! @brief Runtime summary.
        @return str result produced by the operation.
        """
        if self._runtime_report.gpu_verified_active:
            return "compute=cuda (verified active)"
        if self._requested_mode == "cpu":
            return "compute=cpu (GPU disabled by config)"
        if self._cuda_required_unavailable:
            return "compute=fallback (CUDA requested but no CUDA device is available)"
        if not self._runtime_available:
            return "compute=fallback (faster-whisper runtime unavailable)"
        if not self._runtime_report.gpu_backend_available:
            return "compute=cpu (CUDA unavailable; auto fallback)"
        return f"compute={self._runtime_report.active_mode} (GPU requested but not verified active)"

    def _load_model(self):
        """! @brief Load model.
        @return Result produced by the operation.
        """
        if self._model is not None:
            return self._model
        whisper_model_cls = _get_faster_whisper_model_class()
        if whisper_model_cls is None:
            raise TranscriptionError("faster-whisper is not installed. Fix: install the faster-whisper package.")
        compute_type = self._compute_type
        if compute_type == "auto":
            compute_type = "float16" if self._device == "cuda" else "int8"
        kwargs = {"device": self._device, "compute_type": compute_type}
        if self._device == "cuda":
            kwargs["device_index"] = self._cuda_device_id
        self._model = whisper_model_cls(self.model_path, **kwargs)
        return self._model

    def _missing_runtime_message(self) -> str:
        """! @brief Missing runtime message.
        @return str result produced by the operation.
        """
        if self._cuda_required_unavailable:
            return (
                "compute_device=cuda was requested but no CUDA device is available. "
                "Use AUTOMOM_COMPUTE_DEVICE=auto|cpu or fix CUDA visibility."
            )
        if not self.model_path or not Path(self.model_path).exists():
            return "faster-whisper model path is missing. Register a valid local model directory."
        return "faster-whisper runtime is unavailable. Fix: install the faster-whisper package."


class OpenAITranscriber:
    def __init__(self, api_key: str, model: str) -> None:
        """! @brief Initialize the OpenAITranscriber instance.
        @param api_key Value for api key.
        @param model Model identifier used by the operation.
        """
        self._client = OpenAIClient(api_key)
        self._model = model.strip() or "gpt-4o-transcribe"

    def transcribe(self, segment_path: Path) -> str:
        """! @brief Transcribe operation.
        @param segment_path Path to the segment audio file.
        @return str result produced by the operation.
        """
        try:
            return self._client.transcribe_audio(segment_path, model=self._model)
        except OpenAIAPIError as exc:
            raise TranscriptionError(f"OpenAI transcription failed for '{segment_path.name}': {exc}") from exc


class RemoteWhisperCppTranscriber:
    def __init__(self, *, base_url: str, model_name: str, auth_token: str = "", timeout_s: int = 900) -> None:
        self._client = RemoteWorkerClient(
            base_url=base_url,
            auth_token=auth_token,
            timeout_s=timeout_s,
        )
        self._runtime_report = ASRRuntimeReport(
            binary_path=f"{base_url.rstrip('/')}/transcribe",
            model_path=model_name,
            requested_mode="remote",
            available_mode="remote",
            active_mode="remote",
            gpu_requested=False,
            gpu_backend_available=True,
            gpu_verified_active=False,
            supported_flags=[],
            linked_backends=["remote"],
            thread_count=0,
            processor_count=0,
        )

    def available(self) -> bool:
        return True

    def transcribe(self, segment_path: Path) -> str:
        try:
            payload = self._client.transcribe(audio_path=segment_path)
        except RemoteWorkerError as exc:
            self._runtime_report.last_error = str(exc)
            raise TranscriptionError(str(exc)) from exc
        runtime = payload.get("runtime", {})
        self._runtime_report.active_mode = str(runtime.get("compute_active", "remote"))
        self._runtime_report.last_error = ""
        text = clean_transcript_text(str(payload.get("text", "")))
        if not text:
            raise TranscriptionError(f"Remote transcription produced empty output for '{segment_path.name}'.")
        return text

    def runtime_report(self) -> dict[str, object]:
        return self._runtime_report.to_dict()

    def runtime_summary(self) -> str:
        return f"compute={self._runtime_report.active_mode} (remote worker)"


def transcribe_segments(
    transcriber: LocalTranscriber | OpenAITranscriber,
    segment_jobs: list[dict[str, object]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, object]]:
    """! @brief Transcribe segments.
    @param transcriber Value for transcriber.
    @param segment_jobs Value for segment jobs.
    @param progress_callback Optional callback invoked with progress updates.
    @return List produced by the operation.
    """
    transcripts: list[dict[str, object]] = []
    total = len(segment_jobs)

    for idx, segment in enumerate(segment_jobs, start=1):
        audio_path = Path(str(segment["segment_path"]))
        text = transcriber.transcribe(audio_path)
        transcripts.append(
            {
                "speaker_id": segment["speaker_id"],
                "speaker_name": segment["speaker_name"],
                "start_s": float(segment["start_s"]),
                "end_s": float(segment["end_s"]),
                "text": text,
            }
        )
        if progress_callback:
            progress_callback(idx, total)

    ordered = sorted(transcripts, key=lambda item: (item["start_s"], item["end_s"]))
    return merge_transcript_segments(ordered)


def clean_transcript_text(raw_text: str) -> str:
    """! @brief Clean transcript text.
    @param raw_text Value for raw text.
    @return str result produced by the operation.
    """
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = WHISPER_TAG_PATTERN.sub(" ", line)
        line = TIMESTAMP_RANGE_PATTERN.sub(" ", line)
        line = TIMESTAMP_BRACKET_PATTERN.sub(" ", line)
        line = TIMESTAMP_TOKEN_PATTERN.sub(" ", line)
        line = re.sub(r"\s+", " ", line).strip(" -")
        if line:
            lines.append(line)

    if not lines:
        return ""

    merged = " ".join(lines)
    merged = re.sub(r"\s+([,.!?;:])", r"\1", merged)
    return re.sub(r"\s{2,}", " ", merged).strip()


def _invocation_failed(process: subprocess.CompletedProcess[str]) -> bool:
    """! @brief Invocation failed.
    @param process Value for process.
    @return True when the requested condition is satisfied; otherwise False.
    """
    if process.returncode != 0:
        return True

    stderr = process.stderr.lower()
    if "unknown argument" in stderr or "error:" in stderr:
        return True

    return False


def _runtime_gpu_failure_reason(output: str) -> str:
    """! @brief Runtime gpu failure reason.
    @param output Value for output.
    @return str result produced by the operation.
    """
    lowered = output.lower()
    if "no gpu found" in lowered:
        return "no_gpu_found"
    if "failed" in lowered and "gpu" in lowered:
        return "gpu_runtime_failure"
    return "gpu_not_verified"


def _gpu_verified_active(output: str) -> bool:
    """! @brief Gpu verified active.
    @param output Value for output.
    @return True when the requested condition is satisfied; otherwise False.
    """
    lowered = output.lower()
    if "no gpu found" in lowered:
        return False
    return bool(re.search(r"whisper_backend_init_gpu:\s*device\s+\d+:\s*(?!cpu\b).+", lowered))


@lru_cache(maxsize=8)
def _probe_asr_binary(binary_path: str) -> ASRBinaryCapabilities:
    """! @brief Probe asr binary.
    @param binary_path Value for binary path.
    @return Result produced by the operation.
    """
    if not binary_path:
        return ASRBinaryCapabilities("", False, tuple(), tuple(), False)

    resolved = str(Path(binary_path))
    name = Path(resolved).name.lower()
    help_text = _binary_help_text(resolved)
    supported_flags = tuple(sorted(set(re.findall(r"(?<!\w)(?:-{1,2}[a-z0-9][a-z0-9\-]*)", help_text))))
    linked_backends = _detect_linked_backends(resolved)
    is_whisper_cli = "whisper" in name
    if linked_backends:
        gpu_backend_available = any(item != "cpu" for item in linked_backends)
    else:
        gpu_backend_available = any(flag in supported_flags for flag in ("-dev", "--device", "-ngl", "--gpu-layers"))
    return ASRBinaryCapabilities(
        binary_path=resolved,
        is_whisper_cli=is_whisper_cli,
        supported_flags=supported_flags,
        linked_backends=linked_backends,
        gpu_backend_available=gpu_backend_available,
    )


def _detect_linked_backends(binary_path: str) -> tuple[str, ...]:
    """! @brief Detect linked backends.
    @param binary_path Value for binary path.
    @return Tuple produced by the operation.
    """
    try:
        process = subprocess.run(
            ["ldd", binary_path],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return tuple()

    if process.returncode != 0:
        return tuple()

    lowered = process.stdout.lower()
    backends: list[str] = []
    if "libggml-cpu" in lowered:
        backends.append("cpu")
    if "cuda" in lowered or "cublas" in lowered:
        backends.append("cuda")
    if "vulkan" in lowered:
        backends.append("vulkan")
    if "opencl" in lowered:
        backends.append("opencl")
    if "metal" in lowered:
        backends.append("metal")
    if "sycl" in lowered:
        backends.append("sycl")
    return tuple(dict.fromkeys(backends))


@lru_cache(maxsize=8)
def _binary_help_text(binary_path: str) -> str:
    """! @brief Binary help text.
    @param binary_path Value for binary path.
    @return str result produced by the operation.
    """
    try:
        process = subprocess.run(
            [binary_path, "--help"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return ""
    return f"{process.stdout}\n{process.stderr}"


@lru_cache(maxsize=8)
def _binary_supports_any_flag(binary_path: str, flags: tuple[str, ...]) -> bool:
    """! @brief Binary supports any flag.
    @param binary_path Value for binary path.
    @param flags Value for flags.
    @return True when the requested condition is satisfied; otherwise False.
    """
    help_text = _binary_help_text(binary_path)
    if not help_text:
        return False
    return any(flag in help_text for flag in flags)


@lru_cache(maxsize=1)
def _get_faster_whisper_model_class():
    """! @brief Get faster whisper model class.
    @return Result produced by the operation.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    return WhisperModel


def _faster_whisper_available() -> bool:
    """! @brief Faster whisper available.
    @return True when the requested condition is satisfied; otherwise False.
    """
    return _get_faster_whisper_model_class() is not None
