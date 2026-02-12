from __future__ import annotations

import hashlib
from pathlib import Path
import time

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

    model_path = SETTINGS.models_dir / "formatter" / "model.gguf"
    checksum = hashlib.sha256(source_bytes).hexdigest()

    manager = ModelManager()
    manager._consent = {"formatter": True}
    manager._specs = {
        "formatter": ModelSpec(
            model_id="formatter",
            name="Formatter",
            size_mb=3,
            source="local",
            required_disk_mb=3,
            file_path=model_path,
            checksum_sha256=checksum,
            download_url=f"file://{source_path}",
        )
    }

    start_state = manager.start_download("formatter")
    assert start_state["status"] in {"running", "completed"}

    final_state = start_state
    for _ in range(200):
        final_state = manager.download_status("formatter")
        if final_state["status"] == "completed":
            break
        time.sleep(0.01)

    assert final_state["status"] == "completed"
    assert final_state["percent"] == 100.0
    assert final_state["downloaded_bytes"] == len(source_bytes)
    assert model_path.exists()
