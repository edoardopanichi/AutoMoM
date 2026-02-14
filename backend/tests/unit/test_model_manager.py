from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error

from backend.app.config import ModelSpec, SETTINGS
from backend.models.manager import ModelManager


def test_permission_gated_download_flow(isolated_settings, monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "source.bin"
    source_bytes = b"mock-model-bytes"
    source_path.write_bytes(source_bytes)
    checksum = hashlib.sha256(source_bytes).hexdigest()

    model_path = SETTINGS.models_dir / "diarization" / "test_model.bin"

    manager = ModelManager()
    manager._consent = {}
    manager._specs = {
        "diarization": ModelSpec(
            model_id="diarization",
            name="Test Model",
            size_mb=1,
            source="local",
            required_disk_mb=1,
            file_path=model_path,
            checksum_sha256=checksum,
            download_url=f"file://{source_path}",
        )
    }

    try:
        manager.download("diarization")
    except PermissionError:
        pass
    else:
        raise AssertionError("Expected PermissionError without consent")

    manager.set_consent("diarization", True)
    result = manager.download("diarization")

    assert result.verified is True
    assert model_path.exists()
    assert model_path.read_bytes() == source_bytes


def test_async_download_status_progress(isolated_settings, tmp_path: Path) -> None:
    source_path = tmp_path / "large_source.bin"
    source_bytes = b"x" * (3 * 1024 * 1024)
    source_path.write_bytes(source_bytes)

    model_path = SETTINGS.models_dir / "voxtral" / "model.gguf"
    checksum = hashlib.sha256(source_bytes).hexdigest()

    manager = ModelManager()
    manager._consent = {"voxtral": True}
    manager._specs = {
        "voxtral": ModelSpec(
            model_id="voxtral",
            name="Voxtral",
            size_mb=3,
            source="local",
            required_disk_mb=3,
            file_path=model_path,
            checksum_sha256=checksum,
            download_url=f"file://{source_path}"
        )
    }

    start_state = manager.start_download("voxtral")
    assert start_state["status"] in {"running", "completed"}

    final_state = start_state
    for _ in range(200):
        final_state = manager.download_status("voxtral")
        if final_state["status"] == "completed":
            break
        time.sleep(0.01)

    assert final_state["status"] == "completed"
    assert final_state["percent"] == 100.0
    assert final_state["downloaded_bytes"] == len(source_bytes)
    assert model_path.exists()


def test_formatter_model_selection_and_ollama_install_check(isolated_settings, monkeypatch) -> None:
    manager = ModelManager()
    manager.set_formatter_model("qwen2.5:7b")

    def fake_has_model(tag: str) -> bool:
        return tag == "qwen2.5:7b"

    monkeypatch.setattr(manager, "_ollama_has_model", fake_has_model)

    statuses = {item.model_id: item for item in manager.statuses()}
    assert statuses["formatter"].installed is True
    assert "qwen2.5:7b" in statuses["formatter"].file_path


def test_formatter_pull_uses_stream_progress(isolated_settings, monkeypatch) -> None:
    manager = ModelManager()
    manager.set_formatter_model("qwen2.5:3b-instruct-q5_K_M")

    events = [
        {"status": "pulling manifest"},
        {"status": "downloading", "completed": 10, "total": 100},
        {"status": "downloading", "completed": 100, "total": 100},
        {"status": "success"},
    ]
    payload = "".join(json.dumps(item) + "\n" for item in events).encode("utf-8")

    class _FakeResponse:
        def __init__(self, raw: bytes) -> None:
            self._stream = io.BytesIO(raw)

        def readline(self) -> bytes:
            return self._stream.readline()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("backend.models.manager.urllib.request.urlopen", lambda *_args, **_kwargs: _FakeResponse(payload))
    monkeypatch.setattr(manager, "_ollama_has_model", lambda tag: tag == "qwen2.5:3b-instruct-q5_K_M")

    progress: list[tuple[int, int | None]] = []
    written = manager._pull_formatter_model(progress_callback=lambda done, total: progress.append((done, total)))

    assert written == 100
    assert progress[-1] == (100, 100)


def test_formatter_pull_surfaces_ollama_http_error(isolated_settings, monkeypatch) -> None:
    manager = ModelManager()
    manager.set_formatter_model("bad:model")

    class _FakeHTTPError(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__(
                url="http://127.0.0.1:11434/api/pull",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"model not found"}'),
            )

    def _raise(*_args, **_kwargs):
        raise _FakeHTTPError()

    monkeypatch.setattr("backend.models.manager.urllib.request.urlopen", _raise)

    try:
        manager._pull_formatter_model()
    except RuntimeError as exc:
        assert "model not found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for Ollama HTTP error")
