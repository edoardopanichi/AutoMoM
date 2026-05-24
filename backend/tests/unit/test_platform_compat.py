from __future__ import annotations

import subprocess
from pathlib import Path

from backend.pipeline.platform_utils import (
    detect_linked_backends,
    file_url_to_path,
    parse_command_args,
    terminate_process_tree_platform,
)


def test_file_url_to_path_windows_drive_letter() -> None:
    path = file_url_to_path("file:///C:/Models/model.gguf")
    assert str(path).endswith("C:/Models/model.gguf") or str(path).endswith("C:\\Models\\model.gguf")


def test_parse_command_args_handles_quoted_spaces_windows(monkeypatch) -> None:
    monkeypatch.setattr("backend.pipeline.platform_utils.is_windows", lambda: True)
    args = parse_command_args('"C:\\Program Files\\Tool\\tool.exe" --model "C:\\My Models\\a.gguf"')
    assert args[0] == "C:\\Program Files\\Tool\\tool.exe"
    assert args[-1] == "C:\\My Models\\a.gguf"


def test_detect_linked_backends_returns_empty_when_no_probe_tools(monkeypatch) -> None:
    monkeypatch.setattr("backend.pipeline.platform_utils.platform.system", lambda: "Windows")
    monkeypatch.setattr("backend.pipeline.platform_utils.shutil.which", lambda _name: None)
    assert detect_linked_backends("C:/x/whisper-cli.exe") == tuple()


def test_terminate_process_tree_platform_windows_uses_taskkill(monkeypatch) -> None:
    calls: list[list[str]] = []

    class DummyProc:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            raise AssertionError("terminate fallback should not be needed")

        def kill(self):
            raise AssertionError("kill fallback should not be needed")

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("backend.pipeline.platform_utils.is_windows", lambda: True)
    monkeypatch.setattr("backend.pipeline.platform_utils.subprocess.run", fake_run)

    terminate_process_tree_platform(DummyProc())

    assert calls
    assert calls[0][:4] == ["taskkill", "/PID", "123", "/T"]


def test_model_manager_file_download_file_url_windows_path(isolated_settings, tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"abc")
    converted = file_url_to_path(f"file:///{source.as_posix()}")
    assert converted.exists()
