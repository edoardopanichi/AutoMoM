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


def _consent_path() -> Path:
    """! @brief Consent path.
    @return Path result produced by the operation.
    """
    return SETTINGS.models_dir / "consent.json"


def _formatter_model_path() -> Path:
    """! @brief Formatter model path.
    @return Path result produced by the operation.
    """
    return SETTINGS.models_dir / "formatter" / "selected_model.txt"


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
        """! @brief Serialize the current object to a dictionary.
        @return Dictionary produced by the operation.
        """
        return asdict(self)


class ModelManager:
    def __init__(self) -> None:
        """! @brief Initialize the ModelManager instance.
        """
        self._specs = {spec.model_id: spec for spec in required_models()}
        self._consent = self._load_consent()
        self._download_progress: dict[str, DownloadProgress] = {}
        self._download_threads: dict[str, Thread] = {}
        self._download_lock = RLock()

    def _load_consent(self) -> dict[str, bool]:
        """! @brief Load consent.
        @return Dictionary produced by the operation.
        """
        consent_path = _consent_path()
        if consent_path.exists():
            try:
                return json.loads(consent_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _persist_consent(self) -> None:
        """! @brief Persist consent.
        """
        consent_path = _consent_path()
        consent_path.parent.mkdir(parents=True, exist_ok=True)
        consent_path.write_text(json.dumps(self._consent, indent=2), encoding="utf-8")

    def get_formatter_model(self) -> str:
        """! @brief Get formatter model.
        @return str result produced by the operation.
        """
        formatter_model_path = _formatter_model_path()
        if formatter_model_path.exists():
            value = formatter_model_path.read_text(encoding="utf-8").strip()
            if value:
                return value
        return SETTINGS.formatter_ollama_model

    def set_formatter_model(self, model_tag: str) -> str:
        """! @brief Set formatter model.
        @param model_tag Value for model tag.
        @return str result produced by the operation.
        """
        normalized = model_tag.strip()
        if not normalized:
            raise ValueError("Formatter model tag cannot be empty")
        formatter_model_path = _formatter_model_path()
        formatter_model_path.parent.mkdir(parents=True, exist_ok=True)
        formatter_model_path.write_text(normalized, encoding="utf-8")
        object.__setattr__(SETTINGS, "formatter_ollama_model", normalized)
        return normalized

    def set_consent(self, model_id: str, approved: bool) -> None:
        """! @brief Set consent.
        @param model_id Identifier of the target model.
        @param approved Value for approved.
        """
        self._spec(model_id)
        self._consent[model_id] = approved
        self._persist_consent()

    def statuses(self) -> list[ModelStatus]:
        """! @brief Statuses operation.
        @return List produced by the operation.
        """
        statuses: list[ModelStatus] = []
        for spec in self._specs.values():
            installed = self._is_model_installed(spec)
            display_path = str(spec.file_path)
            if spec.model_id == "formatter":
                display_path = f"{SETTINGS.ollama_host}::{self.get_formatter_model()}"
            statuses.append(
                ModelStatus(
                    model_id=spec.model_id,
                    name=spec.name,
                    size_mb=spec.size_mb,
                    source=spec.source,
                    required_disk_mb=spec.required_disk_mb,
                    installed=installed,
                    consent_granted=self._consent.get(spec.model_id, False),
                    file_path=display_path,
                    checksum_sha256=spec.checksum_sha256,
                    download_url=spec.download_url,
                )
            )
        return statuses

    def missing_model_ids(self, required_model_ids: set[str] | None = None) -> list[str]:
        """! @brief Missing model ids.
        @param required_model_ids Value for required model ids.
        @return List produced by the operation.
        """
        required = set(self._specs.keys()) if required_model_ids is None else set(required_model_ids)
        return [
            spec.model_id
            for spec in self._specs.values()
            if spec.model_id in required and not self._is_model_installed(spec)
        ]

    def validate_for_job_start(self, required_model_ids: set[str] | None = None) -> tuple[bool, str | None]:
        """! @brief Validate for job start.
        @param required_model_ids Value for required model ids.
        @return Tuple produced by the operation.
        """
        required = set(self._specs.keys()) if required_model_ids is None else set(required_model_ids)
        for spec in self._specs.values():
            if spec.model_id not in required:
                continue
            if not self._is_model_installed(spec) or not spec.checksum_sha256:
                continue
            try:
                self._verify_checksum_if_needed(spec)
            except Exception as exc:
                return (
                    False,
                    (
                        f"Installed model checksum verification failed for {spec.model_id}: {exc}. "
                        "Please re-download the model."
                    ),
                )
        missing = self.missing_model_ids(required)
        if not missing:
            return True, None

        missing_specs = [self._spec(model_id) for model_id in missing]

        blocked_downloads = [mid for mid in missing if mid != "formatter" and not self._spec(mid).download_url]
        if blocked_downloads:
            details = "; ".join(
                f"{mid} expected at '{self._spec(mid).file_path}'"
                for mid in blocked_downloads
            )
            return (
                False,
                (
                    "Missing models without configured download URL: "
                    f"{', '.join(blocked_downloads)}. {details}. "
                    "Fix: install these files manually at the expected paths or set download URL env vars."
                ),
            )
        missing_details = "; ".join(f"{spec.model_id} -> '{spec.file_path}'" for spec in missing_specs)
        return (
            False,
            (
                f"Missing required models: {', '.join(missing)}. "
                f"Expected files: {missing_details}. "
                "Fix: download them from Settings > Model Manager or place files at those paths."
            ),
        )

    def download(
        self,
        model_id: str,
        retries: int = 2,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> DownloadResult:
        """! @brief Download operation.
        @param model_id Identifier of the target model.
        @param retries Value for retries.
        @param progress_callback Optional callback invoked with progress updates.
        @return Result produced by the operation.
        """
        spec = self._spec(model_id)
        if model_id == "formatter":
            downloaded_bytes = self._pull_formatter_model(progress_callback=progress_callback)
            return DownloadResult(model_id=model_id, bytes_written=downloaded_bytes, verified=True)

        if spec.file_path.exists():
            self._verify_checksum_if_needed(spec)
            file_size = spec.file_path.stat().st_size
            if progress_callback:
                progress_callback(file_size, file_size)
            return DownloadResult(model_id=model_id, bytes_written=file_size, verified=True)

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
        """! @brief Download with resume.
        @param url Value for url.
        @param target_path Value for target path.
        @param timeout Optional timeout in seconds.
        @param progress_callback Optional callback invoked with progress updates.
        @return Tuple produced by the operation.
        """
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
        existing_size = target_path.stat().st_size if target_path.exists() else 0
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        request = urllib.request.Request(url=url, headers=headers)
        total_bytes: int | None = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = int(getattr(response, "status", 200))
                has_content_range = bool(response.headers.get("Content-Range"))
                supports_resume = existing_size > 0 and (status_code == 206 or has_content_range)
                start_offset = existing_size if supports_resume else 0
                mode = "ab" if supports_resume else "wb"

                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    total_bytes = start_offset + int(content_length)

                if progress_callback:
                    progress_callback(start_offset, total_bytes)

                with target_path.open(mode) as output:
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
        """! @brief Verify checksum if needed.
        @param spec Value for spec.
        """
        if not spec.checksum_sha256:
            return
        digest = hashlib.sha256(spec.file_path.read_bytes()).hexdigest()
        if digest != spec.checksum_sha256:
            spec.file_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {spec.model_id}")

    def _spec(self, model_id: str) -> ModelSpec:
        """! @brief Spec operation.
        @param model_id Identifier of the target model.
        @return Result produced by the operation.
        """
        if model_id not in self._specs:
            raise KeyError(model_id)
        return self._specs[model_id]

    def start_download(self, model_id: str, retries: int = 2) -> dict[str, object]:
        """! @brief Start download.
        @param model_id Identifier of the target model.
        @param retries Value for retries.
        @return Dictionary produced by the operation.
        """
        spec = self._spec(model_id)
        if self._is_model_installed(spec):
            try:
                if model_id != "formatter":
                    self._verify_checksum_if_needed(spec)
            except Exception as exc:
                self._set_download_state(
                    model_id,
                    status="failed",
                    downloaded_bytes=0,
                    total_bytes=None,
                    percent=0.0,
                    verified=False,
                    error=str(exc),
                )
                raise
            file_size = spec.file_path.stat().st_size if spec.file_path.exists() else 0
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

        if model_id != "formatter" and not spec.download_url:
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
        """! @brief Download worker.
        @param model_id Identifier of the target model.
        @param retries Value for retries.
        """
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
        """! @brief Progress callback.
        @param model_id Identifier of the target model.
        @param downloaded_bytes Value for downloaded bytes.
        @param total_bytes Value for total bytes.
        """
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
        """! @brief Download status.
        @param model_id Identifier of the target model.
        @return Dictionary produced by the operation.
        """
        spec = self._spec(model_id)
        with self._download_lock:
            state = self._download_progress.get(model_id)
            if state is None:
                installed = self._is_model_installed(spec)
                verified = installed
                if installed and spec.checksum_sha256 and model_id != "formatter":
                    try:
                        self._verify_checksum_if_needed(spec)
                    except Exception:
                        installed = self._is_model_installed(spec)
                        verified = False
                    else:
                        verified = True
                size = spec.file_path.stat().st_size if installed else 0
                status = "completed" if installed and verified else "idle"
                state = DownloadProgress(
                    model_id=model_id,
                    status=status,
                    downloaded_bytes=size,
                    total_bytes=size if installed else None,
                    percent=100.0 if installed and verified else 0.0,
                    verified=verified,
                    started_at=None,
                    updated_at=self._now_iso(),
                )
                self._download_progress[model_id] = state
            return state.to_dict()

    def _is_model_installed(self, spec: ModelSpec) -> bool:
        """! @brief Is model installed.
        @param spec Value for spec.
        @return True when the requested condition is satisfied; otherwise False.
        """
        if spec.model_id != "formatter":
            return spec.file_path.exists()
        return self._ollama_has_model(self.get_formatter_model())

    def _ollama_has_model(self, model_tag: str) -> bool:
        """! @brief Ollama has model.
        @param model_tag Value for model tag.
        @return True when the requested condition is satisfied; otherwise False.
        """
        request = urllib.request.Request(
            url=f"{SETTINGS.ollama_host.rstrip('/')}/api/tags",
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8", errors="replace")
        except Exception:
            return False

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return False
        models = payload.get("models", [])
        wanted = model_tag.strip().lower()
        for item in models:
            if str(item.get("name", "")).strip().lower() == wanted:
                return True
        return False

    def _pull_formatter_model(
        self,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> int:
        """! @brief Pull formatter model.
        @param progress_callback Optional callback invoked with progress updates.
        @return int result produced by the operation.
        """
        model_tag = self.get_formatter_model()
        payload = {"name": model_tag}
        request = urllib.request.Request(
            url=f"{SETTINGS.ollama_host.rstrip('/')}/api/pull",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        downloaded_bytes = 0
        total_bytes: int | None = None
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    raw_line = response.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in item and str(item["error"]).strip():
                        raise RuntimeError(str(item["error"]).strip())

                    completed = item.get("completed")
                    total = item.get("total")
                    if isinstance(completed, int) and completed >= 0:
                        downloaded_bytes = completed
                    if isinstance(total, int) and total > 0:
                        total_bytes = total
                    if progress_callback:
                        progress_callback(downloaded_bytes, total_bytes)
        except urllib.error.HTTPError as exc:
            details = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            if body:
                try:
                    payload = json.loads(body)
                    details = str(payload.get("error") or body).strip()
                except json.JSONDecodeError:
                    details = body.strip()
            if details:
                raise RuntimeError(f"Ollama pull failed: {details}") from exc
            raise RuntimeError(f"Ollama pull failed: HTTP {exc.code}") from exc

        if not self._ollama_has_model(model_tag):
            raise RuntimeError(
                f"Ollama pull finished but model '{model_tag}' was not found in local tags."
            )
        if progress_callback:
            progress_callback(downloaded_bytes, total_bytes or downloaded_bytes)
        return downloaded_bytes

    def all_download_statuses(self) -> list[dict[str, object]]:
        """! @brief All download statuses.
        @return List produced by the operation.
        """
        return [self.download_status(model_id) for model_id in sorted(self._specs.keys())]

    def _set_download_state(self, model_id: str, **updates: object) -> None:
        """! @brief Set download state.
        @param model_id Identifier of the target model.
        @param updates Value for updates.
        """
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
        """! @brief Now iso.
        @return str result produced by the operation.
        """
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


MODEL_MANAGER = ModelManager()
