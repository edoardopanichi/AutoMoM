from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import SETTINGS
from backend.app.schemas import JobSpeakerInfo, JobState, SpeakerMappingItem


@dataclass
class JobRuntime:
    audio_path: Path
    template_id: str
    language_mode: str
    title: str | None
    state: JobState
    speaker_mapping_event: threading.Event = field(default_factory=threading.Event)
    speaker_mapping_payload: list[SpeakerMappingItem] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRuntime] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        audio_path: Path,
        template_id: str,
        language_mode: str,
        title: str | None,
    ) -> JobRuntime:
        now = datetime.now(timezone.utc)
        job_id = str(uuid4())
        state = JobState(
            job_id=job_id,
            status="created",
            created_at=now,
            updated_at=now,
        )
        runtime = JobRuntime(
            audio_path=audio_path,
            template_id=template_id,
            language_mode=language_mode,
            title=title,
            state=state,
        )
        with self._lock:
            self._jobs[job_id] = runtime

        job_dir = SETTINGS.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self._persist_state(job_id)
        return runtime

    def get_runtime(self, job_id: str) -> JobRuntime:
        with self._lock:
            runtime = self._jobs.get(job_id)
        if runtime is None:
            raise KeyError(job_id)
        return runtime

    def get_state(self, job_id: str) -> JobState:
        return self.get_runtime(job_id).state

    def list_states(self) -> list[JobState]:
        with self._lock:
            values = [runtime.state for runtime in self._jobs.values()]
        return sorted(values, key=lambda item: item.created_at, reverse=True)

    def mark_running(self, job_id: str) -> None:
        runtime = self.get_runtime(job_id)
        runtime.state.status = "running"
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def set_stage(self, job_id: str, stage: str, overall_percent: float) -> None:
        runtime = self.get_runtime(job_id)
        runtime.state.current_stage = stage
        runtime.state.stage_percent = 0.0
        runtime.state.overall_percent = max(0.0, min(100.0, overall_percent))
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def set_stage_percent(self, job_id: str, value: float, overall_percent: float | None = None) -> None:
        runtime = self.get_runtime(job_id)
        runtime.state.stage_percent = max(0.0, min(100.0, value))
        if overall_percent is not None:
            runtime.state.overall_percent = max(0.0, min(100.0, overall_percent))
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def set_segment_progress(self, job_id: str, completed: int, total: int) -> None:
        runtime = self.get_runtime(job_id)
        runtime.state.transcript_segment_progress = f"{completed} of {total}"
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def append_log(self, job_id: str, message: str) -> None:
        runtime = self.get_runtime(job_id)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        runtime.state.logs.append(f"[{timestamp}] {message}")
        if len(runtime.state.logs) > 200:
            runtime.state.logs = runtime.state.logs[-200:]
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def set_artifact(self, job_id: str, key: str, path: Path) -> None:
        runtime = self.get_runtime(job_id)
        runtime.state.artifact_paths[key] = str(path)
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def set_waiting_for_speaker_input(self, job_id: str, speaker_info: JobSpeakerInfo) -> None:
        runtime = self.get_runtime(job_id)
        runtime.state.status = "waiting_speaker_input"
        runtime.state.speaker_info = speaker_info
        runtime.state.stage_percent = 0.0
        runtime.state.updated_at = datetime.now(timezone.utc)
        runtime.speaker_mapping_event.clear()
        self._persist_state(job_id)

    def submit_speaker_mapping(self, job_id: str, mappings: list[SpeakerMappingItem]) -> None:
        runtime = self.get_runtime(job_id)
        runtime.speaker_mapping_payload = mappings
        runtime.state.stage_percent = 100.0
        runtime.state.status = "running"
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
        runtime = self.get_runtime(job_id)
        runtime.state.status = "completed"
        runtime.state.stage_percent = 100.0
        runtime.state.overall_percent = 100.0
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def mark_failed(self, job_id: str, error: str) -> None:
        runtime = self.get_runtime(job_id)
        runtime.state.status = "failed"
        runtime.state.error = error
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def cancel(self, job_id: str) -> None:
        runtime = self.get_runtime(job_id)
        runtime.cancel_event.set()
        runtime.state.status = "cancelled"
        runtime.state.updated_at = datetime.now(timezone.utc)
        self._persist_state(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        runtime = self.get_runtime(job_id)
        return runtime.cancel_event.is_set()

    def _persist_state(self, job_id: str) -> None:
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
