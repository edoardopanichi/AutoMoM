from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def is_windows() -> bool:
    return os.name == "nt" or platform.system().lower() == "windows"


def parse_command_args(command: str) -> list[str]:
    """Parse a simple executable+args command string across platforms."""
    args = shlex.split(command, posix=not is_windows())
    if is_windows():
        normalized: list[str] = []
        for token in args:
            if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
                normalized.append(token[1:-1])
            else:
                normalized.append(token)
        return normalized
    return args


def detect_linked_backends(binary_path: str) -> tuple[str, ...]:
    """Best-effort backend detection for whisper binaries across OSes."""
    if not binary_path:
        return tuple()

    system_name = platform.system().lower()
    output = ""

    try:
        if system_name == "linux" and shutil.which("ldd"):
            process = subprocess.run(["ldd", binary_path], capture_output=True, text=True, timeout=2)
            if process.returncode == 0:
                output = process.stdout
        elif system_name == "darwin" and shutil.which("otool"):
            process = subprocess.run(["otool", "-L", binary_path], capture_output=True, text=True, timeout=2)
            if process.returncode == 0:
                output = process.stdout
    except Exception:
        output = ""

    if not output:
        return tuple()

    lowered = output.lower()
    backends: list[str] = []
    if "libggml-cpu" in lowered or "cpu" in lowered:
        backends.append("cpu")
    if "cuda" in lowered or "cublas" in lowered:
        backends.append("cuda")
    if "vulkan" in lowered:
        backends.append("vulkan")
    if "opencl" in lowered:
        backends.append("opencl")
    if "metal" in lowered:
        backends.append("metal")
    if "sycl" in lowered:
        backends.append("sycl")
    return tuple(dict.fromkeys(backends))


def terminate_process_tree_platform(process: subprocess.Popen[str], *, grace_timeout_s: float = 1.5) -> None:
    if process.poll() is not None:
        return

    if is_windows():
        try:
            # taskkill /T reliably tears down child process trees on Windows.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(1.0, grace_timeout_s),
            )
            process.wait(timeout=grace_timeout_s)
            return
        except Exception:
            pass

        try:
            process.terminate()
            process.wait(timeout=grace_timeout_s)
            return
        except Exception:
            pass

        try:
            process.kill()
            process.wait(timeout=1.0)
        except Exception:
            pass
        return

    # POSIX path
    import signal

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        return

    try:
        process.wait(timeout=grace_timeout_s)
        return
    except Exception:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=1.0)
    except Exception:
        pass


def file_url_to_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file":
        raise ValueError(f"Unsupported URL scheme for local file: {parsed.scheme}")
    # Handles Windows drive letters and UNC shares via stdlib pathname conversion.
    raw_path = urllib.request.url2pathname(parsed.path or "")
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        raw_path = f"//{parsed.netloc}{raw_path}"
    return Path(raw_path)
