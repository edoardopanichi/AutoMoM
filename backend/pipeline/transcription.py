from __future__ import annotations

import re
import subprocess
from pathlib import Path
import shutil
from typing import Callable
from functools import lru_cache

from backend.pipeline.compute import should_enable_native_gpu
from backend.pipeline.diarization import merge_transcript_segments

TIMESTAMP_RANGE_PATTERN = re.compile(
    r"\[\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\s*-->\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\s*\]"
)
TIMESTAMP_BRACKET_PATTERN = re.compile(r"\[\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\s*\]")
TIMESTAMP_TOKEN_PATTERN = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\b")
WHISPER_TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")


class VoxtralTranscriber:
    def __init__(
        self,
        binary_path: str | None,
        model_path: str | None,
        *,
        compute_device: str = "auto",
        cuda_device_id: int = 0,
        gpu_layers: int = 99,
    ) -> None:
        self.binary_path = binary_path or ""
        self.model_path = model_path or ""
        self._resolved_binary_path = self._resolve_binary_path(self.binary_path)
        self._cuda_device_id = max(0, int(cuda_device_id))
        self._gpu_layers = max(0, int(gpu_layers))
        self._gpu_requested = should_enable_native_gpu(compute_device, self._cuda_device_id)
        self._gpu_retry_disabled = False
        self._runtime_available = bool(
            self._resolved_binary_path and self.model_path and Path(self.model_path).exists()
        )

    def available(self) -> bool:
        return self._runtime_available

    def transcribe(self, segment_path: Path) -> str:
        if not self._runtime_available or not self._resolved_binary_path:
            return self._fallback_transcription(segment_path)

        use_gpu = self._gpu_requested and not self._gpu_retry_disabled
        command = self._build_command(segment_path, use_gpu=use_gpu)
        process = subprocess.run(command, capture_output=True, text=True)
        invocation_failed = _invocation_failed(process)
        if invocation_failed and use_gpu:
            # If GPU flags are unsupported, retry once on CPU and keep running.
            retry_command = self._build_command(segment_path, use_gpu=False)
            retry_process = subprocess.run(retry_command, capture_output=True, text=True)
            if not _invocation_failed(retry_process):
                process = retry_process
                invocation_failed = False
                self._gpu_retry_disabled = True
        if invocation_failed:
            return self._fallback_transcription(segment_path)
        raw_text = process.stdout if process.stdout.strip() else process.stderr
        text = clean_transcript_text(raw_text)
        return text or self._fallback_transcription(segment_path)

    def compute_mode(self) -> str:
        if not self._runtime_available:
            return "fallback"
        if self._gpu_requested and not self._gpu_retry_disabled:
            return "cuda"
        if self._gpu_requested and self._gpu_retry_disabled:
            return "cpu(gpu_retry_disabled)"
        return "cpu"

    def _build_command(self, segment_path: Path, *, use_gpu: bool) -> list[str]:
        command = [self._resolved_binary_path, "-m", self.model_path, "-f", str(segment_path)]
        if not use_gpu or not self._resolved_binary_path:
            return command
        if "whisper" not in Path(self._resolved_binary_path).name.lower():
            return command

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
        if supports_device and self._cuda_device_id > 0 and "-dev" not in tokens and "--device" not in tokens:
            command.extend(["-dev", str(self._cuda_device_id)])
        return command

    @staticmethod
    def _resolve_binary_path(binary_path: str) -> str | None:
        if not binary_path:
            return None
        as_path = Path(binary_path)
        if as_path.exists():
            return str(as_path)
        which_path = shutil.which(binary_path)
        if which_path:
            return which_path
        return None

    @staticmethod
    def _fallback_transcription(segment_path: Path) -> str:
        name = segment_path.stem.replace("_", " ")
        return f"[Offline fallback transcript for {name}]"



def transcribe_segments(
    transcriber: VoxtralTranscriber,
    segment_jobs: list[dict[str, object]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, object]]:
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
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = WHISPER_TAG_PATTERN.sub(" ", line)
        line = TIMESTAMP_RANGE_PATTERN.sub(" ", line)
        line = TIMESTAMP_BRACKET_PATTERN.sub(" ", line)
        line = TIMESTAMP_TOKEN_PATTERN.sub(" ", line)
        line = line.replace("-->", " ")
        line = re.sub(r"\s+", " ", line).strip(" -")
        if line:
            lines.append(line)

    if not lines:
        return ""

    merged = " ".join(lines)
    merged = re.sub(r"\s+([,.!?;:])", r"\1", merged)
    return re.sub(r"\s{2,}", " ", merged).strip()


def _invocation_failed(process: subprocess.CompletedProcess[str]) -> bool:
    if process.returncode != 0:
        return True

    stderr = process.stderr.lower()
    if "unknown argument" in stderr or "error:" in stderr:
        return True

    return False


@lru_cache(maxsize=8)
def _binary_supports_any_flag(binary_path: str, flags: tuple[str, ...]) -> bool:
    try:
        process = subprocess.run(
            [binary_path, "--help"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return False

    help_text = f"{process.stdout}\n{process.stderr}"
    return any(flag in help_text for flag in flags)
