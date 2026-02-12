from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import RLock, Thread
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from backend.app.config import SETTINGS, ModelSpec, required_models
from backend.app.schemas import ModelStatus


CONSENT_PATH = SETTINGS.models_dir / "consent.json"


@dataclass
class DownloadResult:
    model_id: str
    bytes_written: int
    verified: bool


@dataclass
class DownloadProgress:
    model_id: str
    status: str
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    percent: float = 0.0
    verified: bool = False
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ModelManager:
    def __init__(self) -> None:
        self._specs = {spec.model_id: spec for spec in required_models()}
        self._consent = self._load_consent()
        self._download_progress: dict[str, DownloadProgress] = {}
        self._download_threads: dict[str, Thread] = {}
        self._download_lock = RLock()

    def _load_consent(self) -> dict[str, bool]:
        if CONSENT_PATH.exists():
            try:
                return json.loads(CONSENT_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _persist_consent(self) -> None:
        CONSENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONSENT_PATH.write_text(json.dumps(self._consent, indent=2), encoding="utf-8")

    def set_consent(self, model_id: str, approved: bool) -> None:
        self._spec(model_id)
        self._consent[model_id] = approved
        self._persist_consent()

    def statuses(self) -> list[ModelStatus]:
        statuses: list[ModelStatus] = []
        for spec in self._specs.values():
            statuses.append(
                ModelStatus(
                    model_id=spec.model_id,
                    name=spec.name,
                    size_mb=spec.size_mb,
                    source=spec.source,
                    required_disk_mb=spec.required_disk_mb,
                    installed=spec.file_path.exists(),
                    consent_granted=self._consent.get(spec.model_id, False),
                    file_path=str(spec.file_path),
                    checksum_sha256=spec.checksum_sha256,
                    download_url=spec.download_url,
                )
            )
        return statuses

    def all_required_present(self) -> bool:
        return all(spec.file_path.exists() for spec in self._specs.values())

    def missing_model_ids(self) -> list[str]:
        return [spec.model_id for spec in self._specs.values() if not spec.file_path.exists()]

    def validate_for_job_start(self) -> tuple[bool, str | None]:
        missing = self.missing_model_ids()
        if not missing:
            return True, None
        missing_without_consent = [mid for mid in missing if not self._consent.get(mid, False)]
        if missing_without_consent:
            ids = ", ".join(missing_without_consent)
            return False, f"Missing required model consent for: {ids}"

        blocked_downloads = [mid for mid in missing if not self._spec(mid).download_url]
        if blocked_downloads:
            ids = ", ".join(blocked_downloads)
            return (
                False,
                f"Missing models without configured download URL: {ids}. Install manually or set env download URLs.",
            )
        return False, "Models are missing; download them from Settings before starting a job."

    def download(
        self,
        model_id: str,
        retries: int = 2,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> DownloadResult:
        spec = self._spec(model_id)
        if spec.file_path.exists():
            file_size = spec.file_path.stat().st_size
            if progress_callback:
                progress_callback(file_size, file_size)
            return DownloadResult(model_id=model_id, bytes_written=file_size, verified=True)

        if not self._consent.get(model_id, False):
            raise PermissionError(f"Consent not granted for model {model_id}")
        if not spec.download_url:
            raise ValueError(f"No download URL configured for model {model_id}")

        spec.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = spec.file_path.with_suffix(spec.file_path.suffix + ".partial")

        last_error: Exception | None = None
        for _attempt in range(retries + 1):
            try:
                bytes_written, total_bytes = self._download_with_resume(
                    spec.download_url,
                    tmp_path,
                    progress_callback=progress_callback,
                )
                if progress_callback:
                    progress_callback(bytes_written, total_bytes or bytes_written)
                tmp_path.rename(spec.file_path)
                self._verify_checksum_if_needed(spec)
                return DownloadResult(model_id=model_id, bytes_written=bytes_written, verified=True)
            except Exception as exc:  # pragma: no cover - best effort retry behavior
                last_error = exc
                time.sleep(0.2)

        raise RuntimeError(f"Download failed for {model_id}: {last_error}")

    def _download_with_resume(
        self,
        url: str,
        target_path: Path,
        timeout: int = 30,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> tuple[int, int | None]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "file":
            src = Path(parsed.path)
            total_size = src.stat().st_size
            existing_size = target_path.stat().st_size if target_path.exists() else 0
            if progress_callback:
                progress_callback(existing_size, total_size)

            with src.open("rb") as src_file, target_path.open("ab") as handle:
                src_file.seek(existing_size)
                while True:
                    chunk = src_file.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    if progress_callback:
                        progress_callback(handle.tell(), total_size)
            return target_path.stat().st_size, total_size

        headers = {}
        mode = "wb"
        existing_size = target_path.stat().st_size if target_path.exists() else 0
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        request = urllib.request.Request(url=url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, target_path.open(mode) as output:
                total_bytes: int | None = None
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    total_bytes = existing_size + int(content_length)

                if progress_callback:
                    progress_callback(existing_size, total_bytes)

                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    if progress_callback:
                        progress_callback(output.tell(), total_bytes)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                file_size = target_path.stat().st_size
                return file_size, file_size
            raise

        return target_path.stat().st_size, total_bytes

    def _verify_checksum_if_needed(self, spec: ModelSpec) -> None:
        if not spec.checksum_sha256:
            return
        digest = hashlib.sha256(spec.file_path.read_bytes()).hexdigest()
        if digest != spec.checksum_sha256:
            spec.file_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {spec.model_id}")

    def _spec(self, model_id: str) -> ModelSpec:
        if model_id not in self._specs:
            raise KeyError(model_id)
        return self._specs[model_id]

    def start_download(self, model_id: str, retries: int = 2) -> dict[str, object]:
        spec = self._spec(model_id)
        if spec.file_path.exists():
            file_size = spec.file_path.stat().st_size
            self._set_download_state(
                model_id,
                status="completed",
                downloaded_bytes=file_size,
                total_bytes=file_size,
                percent=100.0,
                verified=True,
                error=None,
            )
            return self.download_status(model_id)

        if not self._consent.get(model_id, False):
            raise PermissionError(f"Consent not granted for model {model_id}")
        if not spec.download_url:
            raise ValueError(f"No download URL configured for model {model_id}")

        with self._download_lock:
            running = self._download_threads.get(model_id)
            if running and running.is_alive():
                return self.download_status(model_id)

            tmp_path = spec.file_path.with_suffix(spec.file_path.suffix + ".partial")
            existing_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            self._set_download_state(
                model_id,
                status="running",
                downloaded_bytes=existing_size,
                total_bytes=None,
                percent=0.0,
                verified=False,
                error=None,
            )

            worker = Thread(target=self._download_worker, args=(model_id, retries), daemon=True)
            self._download_threads[model_id] = worker
            worker.start()

        return self.download_status(model_id)

    def _download_worker(self, model_id: str, retries: int) -> None:
        try:
            result = self.download(model_id, retries=retries, progress_callback=lambda d, t: self._progress_callback(model_id, d, t))
            self._set_download_state(
                model_id,
                status="completed",
                downloaded_bytes=result.bytes_written,
                total_bytes=result.bytes_written,
                percent=100.0,
                verified=result.verified,
                error=None,
            )
        except Exception as exc:
            self._set_download_state(
                model_id,
                status="failed",
                error=str(exc),
            )

    def _progress_callback(self, model_id: str, downloaded_bytes: int, total_bytes: int | None) -> None:
        percent = 0.0
        if total_bytes and total_bytes > 0:
            percent = min(100.0, (downloaded_bytes / total_bytes) * 100.0)

        self._set_download_state(
            model_id,
            status="running",
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            percent=percent,
            verified=False,
            error=None,
        )

    def download_status(self, model_id: str) -> dict[str, object]:
        self._spec(model_id)
        with self._download_lock:
            state = self._download_progress.get(model_id)
            if state is None:
                installed = self._spec(model_id).file_path.exists()
                size = self._spec(model_id).file_path.stat().st_size if installed else 0
                status = "completed" if installed else "idle"
                state = DownloadProgress(
                    model_id=model_id,
                    status=status,
                    downloaded_bytes=size,
                    total_bytes=size if installed else None,
                    percent=100.0 if installed else 0.0,
                    verified=installed,
                    started_at=None,
                    updated_at=self._now_iso(),
                )
                self._download_progress[model_id] = state
            return state.to_dict()

    def all_download_statuses(self) -> list[dict[str, object]]:
        return [self.download_status(model_id) for model_id in sorted(self._specs.keys())]

    def _set_download_state(self, model_id: str, **updates: object) -> None:
        with self._download_lock:
            state = self._download_progress.get(model_id)
            if state is None:
                state = DownloadProgress(model_id=model_id, status="idle", updated_at=self._now_iso())
                self._download_progress[model_id] = state

            for key, value in updates.items():
                setattr(state, key, value)

            if state.started_at is None and state.status == "running":
                state.started_at = self._now_iso()
            state.updated_at = self._now_iso()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


MODEL_MANAGER = ModelManager()
