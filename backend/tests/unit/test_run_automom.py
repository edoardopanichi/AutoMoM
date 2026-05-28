from __future__ import annotations

from pathlib import Path

import pytest

import run_automom


def test_choose_launcher_non_windows_uses_sh(monkeypatch) -> None:
    monkeypatch.setattr(run_automom.shutil, "which", lambda name: "/usr/bin/bash" if name == "bash" else None)
    kind, command = run_automom._choose_launcher("Linux")
    assert kind == "bash"
    assert command[0] == "/usr/bin/bash"
    assert Path(command[1]).parts[-2:] == ("scripts", "run_automom.sh")


def test_choose_launcher_windows_prefers_bash(monkeypatch) -> None:
    monkeypatch.setattr(
        run_automom.shutil,
        "which",
        lambda name: "C:/Program Files/Git/bin/bash.exe" if name == "bash" else ("C:/pwsh.exe" if name == "pwsh" else None),
    )
    kind, command = run_automom._choose_launcher("Windows")
    assert kind == "bash"
    assert Path(command[1]).parts[-2:] == ("scripts", "run_automom.sh")


def test_choose_launcher_windows_uses_powershell_without_bash(monkeypatch) -> None:
    monkeypatch.setattr(
        run_automom.shutil,
        "which",
        lambda name: "C:/pwsh.exe" if name == "pwsh" else None,
    )
    kind, command = run_automom._choose_launcher("Windows")
    assert kind == "powershell"
    assert command[0] == "C:/pwsh.exe"
    assert Path(command[-1]).parts[-2:] == ("scripts", "run_automom.ps1")


def test_choose_launcher_windows_without_executors_fails(monkeypatch) -> None:
    monkeypatch.setattr(run_automom.shutil, "which", lambda _name: None)
    with pytest.raises(run_automom.LauncherError):
        run_automom._choose_launcher("Windows")


def test_choose_launcher_windows_ignores_wsl_bash_and_uses_powershell(monkeypatch) -> None:
    monkeypatch.setattr(
        run_automom.shutil,
        "which",
        lambda name: "C:/Windows/System32/bash.exe" if name == "bash" else ("C:/pwsh.exe" if name == "pwsh" else None),
    )
    kind, command = run_automom._choose_launcher("Windows")
    assert kind == "powershell"
    assert command[0] == "C:/pwsh.exe"


def test_choose_mock_preparer_windows_ignores_wsl_bash_and_uses_powershell(monkeypatch) -> None:
    monkeypatch.setattr(
        run_automom.shutil,
        "which",
        lambda name: "C:/Windows/System32/bash.exe" if name == "bash" else ("C:/pwsh.exe" if name == "pwsh" else None),
    )
    kind, command = run_automom._choose_mock_preparer("Windows")
    assert kind == "powershell"
    assert command[0] == "C:/pwsh.exe"
