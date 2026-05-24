#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"


class LauncherError(RuntimeError):
    pass


def _find_powershell() -> str | None:
    for candidate in ("pwsh", "powershell"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _is_windows_compatible_bash(bash_path: str | None) -> bool:
    if not bash_path:
        return False
    lowered = bash_path.replace("\\", "/").lower()
    # Prefer Git Bash or MSYS-style bash on Windows. Ignore WSL's bash.exe.
    if "windows/system32/bash.exe" in lowered:
        return False
    return any(token in lowered for token in ("git", "msys", "mingw"))


def _choose_launcher(system_name: str | None = None) -> tuple[str, list[str]]:
    system_name = (system_name or platform.system()).strip().lower()
    bash = shutil.which("bash")
    powershell = _find_powershell()

    if system_name != "windows":
        if not bash:
            raise LauncherError("Bash is required on non-Windows systems. Install bash and retry.")
        script = SCRIPTS_DIR / "run_automom.sh"
        return "bash", [bash, str(script)]

    # Windows: prefer Bash (Git Bash parity), then native PowerShell.
    if _is_windows_compatible_bash(bash):
        script = SCRIPTS_DIR / "run_automom.sh"
        return "bash", [bash, str(script)]

    if powershell:
        script = SCRIPTS_DIR / "run_automom.ps1"
        return "powershell", [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]

    raise LauncherError(
        "No compatible launcher found on Windows. Install Git Bash or PowerShell (pwsh/powershell), then retry."
    )


def _choose_mock_preparer(system_name: str | None = None) -> tuple[str, list[str]]:
    system_name = (system_name or platform.system()).strip().lower()
    bash = shutil.which("bash")
    powershell = _find_powershell()

    if system_name != "windows":
        if not bash:
            raise LauncherError("Bash is required on non-Windows systems. Install bash and retry.")
        script = SCRIPTS_DIR / "prepare_mock_models.sh"
        return "bash", [bash, str(script)]

    if _is_windows_compatible_bash(bash):
        script = SCRIPTS_DIR / "prepare_mock_models.sh"
        return "bash", [bash, str(script)]

    if powershell:
        script = SCRIPTS_DIR / "prepare_mock_models.ps1"
        return "powershell", [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]

    raise LauncherError(
        "No compatible launcher found on Windows. Install Git Bash or PowerShell (pwsh/powershell), then retry."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-platform AutoMoM launcher")
    parser.add_argument(
        "--prepare-mock-models",
        action="store_true",
        help="Prepare mock model files instead of starting the app",
    )
    args = parser.parse_args()

    try:
        if args.prepare_mock_models:
            launcher, command = _choose_mock_preparer()
        else:
            launcher, command = _choose_launcher()
    except LauncherError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Using launcher: {launcher}")
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
