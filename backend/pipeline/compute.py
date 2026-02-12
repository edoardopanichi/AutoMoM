from __future__ import annotations

from functools import lru_cache
import os
import subprocess
import warnings


VALID_COMPUTE_PREFERENCES = {"auto", "cpu", "cuda"}


def normalize_compute_preference(preference: str | None) -> str:
    normalized = (preference or "auto").strip().lower()
    if normalized in VALID_COMPUTE_PREFERENCES:
        return normalized
    return "auto"


def _cuda_disabled_by_env() -> bool:
    value = os.getenv("AUTOMOM_DISABLE_CUDA", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@lru_cache(maxsize=8)
def torch_cuda_available(device_id: int = 0) -> bool:
    if _cuda_disabled_by_env():
        return False

    safe_device_id = max(0, int(device_id))
    try:
        import torch
    except Exception:
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if available else 0
        return available and count > safe_device_id
    except Exception:
        return False


@lru_cache(maxsize=8)
def native_cuda_available(device_id: int = 0) -> bool:
    if _cuda_disabled_by_env():
        return False

    safe_device_id = max(0, int(device_id))
    if torch_cuda_available(safe_device_id):
        return True

    try:
        process = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return False

    if process.returncode != 0:
        return False

    gpu_lines = [
        line.strip()
        for line in process.stdout.splitlines()
        if line.strip().startswith("GPU ")
    ]
    return len(gpu_lines) > safe_device_id


def resolve_torch_device(preference: str | None, device_id: int = 0) -> str:
    normalized = normalize_compute_preference(preference)
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "cuda" if torch_cuda_available(device_id) else "cpu"
    return "cuda" if torch_cuda_available(device_id) else "cpu"


def should_enable_native_gpu(preference: str | None, device_id: int = 0) -> bool:
    normalized = normalize_compute_preference(preference)
    if normalized == "cpu":
        return False
    if normalized == "cuda":
        return native_cuda_available(device_id)
    return native_cuda_available(device_id)
