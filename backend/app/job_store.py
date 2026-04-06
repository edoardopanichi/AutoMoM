from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.config import SETTINGS
from backend.app.schemas import JobSpeakerInfo, JobState, SpeakerMappingItem


@dataclass
class OpenAIJobConfig:
    api_key: str
    diarization_execution: str = "local"
    transcription_execution: str = "local"
    formatter_execution: str = "local"
    diarization_model: str = "gpt-4o-transcribe-diarize"
    transcription_model: str = "gpt-4o-transcribe"
    formatter_model: str = "gpt-5-mini"


@dataclass
class JobRuntime:
    audio_path: Path
    original_filename: str | None
    template_id: str
    language_mode: str
    title: str | None
    local_diarization_model_id: str
    api_config: OpenAIJobConfig | None
    state: JobState
    speaker_mapping_event: threading.Event = field(default_factory=threading.Event)
    speaker_mapping_payload: list[SpeakerMappingItem] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRuntime] = {}
        self._lock = threading.RLock()

    def create_job(
        self,
        audio_path: Path,
        original_filename: str | None,
        template_id: str,
        language_mode: str,
        title: str | None,
        local_diarization_model_id: str,
        api_config: OpenAIJobConfig | None = None,
    ) -> JobRuntime:
        now = datetime.now(timezone.utc)
        job_id = self._build_job_id(now, title)
        state = JobState(
            job_id=job_id,
            status="created",
            created_at=now,
            updated_at=now,
        )
        runtime = JobRuntime(
            audio_path=audio_path,
            original_filename=original_filename,
            template_id=template_id,
            language_mode=language_mode,
            title=title,
            local_diarization_model_id=local_diarization_model_id,
            api_config=api_config,
            state=state,
        )
        with self._lock:
            self._jobs[job_id] = runtime

        job_dir = SETTINGS.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self._persist_state(job_id)
        return runtime

    @staticmethod
    def _slugify_title(title: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
        cleaned = cleaned.strip("_")
        return cleaned[:80] if cleaned else "meeting"

    def _build_job_id(self, now: datetime, title: str | None) -> str:
        timestamp = now.strftime("%Y-%m-%d-%H:%M")
        if title and title.strip():
            base = f"{timestamp}-{self._slugify_title(title)}"
        else:
            base = f"{timestamp}-meeting"

        candidate = base
        counter = 2
        with self._lock:
            while (SETTINGS.jobs_dir / candidate).exists() or candidate in self._jobs:
                candidate = f"{base}-{counter}"
                counter += 1
        return candidate

    def get_runtime(self, job_id: str) -> JobRuntime:
        with self._lock:
            runtime = self._jobs.get(job_id)
        if runtime is None:
            raise KeyError(job_id)
        return runtime

    def get_state(self, job_id: str) -> JobState:
        with self._lock:
            runtime = self._jobs.get(job_id)
            if runtime is None:
                raise KeyError(job_id)
            return runtime.state.model_copy(deep=True)

    def list_states(self) -> list[JobState]:
        with self._lock:
            values = [runtime.state.model_copy(deep=True) for runtime in self._jobs.values()]
        return sorted(values, key=lambda item: item.created_at, reverse=True)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.status = "running"
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def set_stage(self, job_id: str, stage: str, overall_percent: float) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.current_stage = stage
            runtime.state.stage_percent = 0.0
            runtime.state.stage_detail = None
            runtime.state.overall_percent = max(0.0, min(100.0, overall_percent))
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def set_stage_percent(
        self,
        job_id: str,
        value: float,
        overall_percent: float | None = None,
        stage_detail: str | None = None,
    ) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.stage_percent = max(0.0, min(100.0, value))
            if overall_percent is not None:
                runtime.state.overall_percent = max(0.0, min(100.0, overall_percent))
            if stage_detail is not None:
                runtime.state.stage_detail = stage_detail
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def set_transcription_progress(
        self,
        job_id: str,
        stage_percent: float,
        completed: int,
        total: int,
        overall_percent: float | None = None,
    ) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.stage_percent = max(0.0, min(100.0, stage_percent))
            if overall_percent is not None:
                runtime.state.overall_percent = max(0.0, min(100.0, overall_percent))
            runtime.state.transcript_segment_progress = f"{completed} of {total}"
            runtime.state.stage_detail = None
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            runtime.state.logs.append(f"[{timestamp}] {message}")
            if len(runtime.state.logs) > 200:
                runtime.state.logs = runtime.state.logs[-200:]
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def set_artifact(self, job_id: str, key: str, path: Path) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.artifact_paths[key] = str(path)
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def set_waiting_for_speaker_input(self, job_id: str, speaker_info: JobSpeakerInfo) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.status = "waiting_speaker_input"
            runtime.state.speaker_info = speaker_info
            runtime.state.stage_percent = 0.0
            runtime.state.stage_detail = None
            runtime.state.updated_at = datetime.now(timezone.utc)
            runtime.speaker_mapping_event.clear()
            self._persist_state(job_id)

    def submit_speaker_mapping(self, job_id: str, mappings: list[SpeakerMappingItem]) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            if runtime.state.status != "waiting_speaker_input":
                raise ValueError("Job is not waiting for speaker input")
            runtime.speaker_mapping_payload = mappings
            runtime.state.stage_percent = 100.0
            runtime.state.status = "running"
            runtime.state.stage_detail = None
            runtime.state.updated_at = datetime.now(timezone.utc)
            runtime.speaker_mapping_event.set()
            self._persist_state(job_id)

    def wait_for_mapping(self, job_id: str, timeout_s: float = 1.0) -> list[SpeakerMappingItem] | None:
        runtime = self.get_runtime(job_id)
        while True:
            if runtime.cancel_event.is_set():
                return None
            if runtime.speaker_mapping_event.wait(timeout=timeout_s):
                return runtime.speaker_mapping_payload

    def mark_completed(self, job_id: str) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.status = "completed"
            runtime.state.stage_percent = 100.0
            runtime.state.stage_detail = None
            runtime.state.overall_percent = 100.0
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            runtime.state.status = "failed"
            runtime.state.error = error
            runtime.state.stage_detail = None
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            if runtime.state.status in {"completed", "failed", "cancelled"}:
                return
            runtime.cancel_event.set()
            runtime.state.status = "cancelled"
            runtime.state.stage_detail = None
            runtime.state.updated_at = datetime.now(timezone.utc)
            self._persist_state(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        runtime = self.get_runtime(job_id)
        return runtime.cancel_event.is_set()

    def _persist_state(self, job_id: str) -> None:
        with self._lock:
            runtime = self.get_runtime(job_id)
            state_path = SETTINGS.jobs_dir / job_id / "job_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(runtime.state.model_dump(mode="json"), indent=2), encoding="utf-8")


JOB_STORE = JobStore()


def ensure_job_artifact_dir(job_id: str, *parts: str) -> Path:
    path = SETTINGS.jobs_dir / job_id
    for item in parts:
        path = path / item
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
