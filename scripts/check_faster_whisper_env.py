from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.local_catalog import (
    VALID_FASTER_WHISPER_COMPUTE_TYPES,
    validate_faster_whisper_model_directory,
)


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception:
        return ""
    return str(getattr(module, "__version__", ""))


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _package_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in ("faster_whisper", "ctranslate2", "torch"):
        try:
            importlib.import_module(name)
            checks[name] = {
                "available": True,
                "version": _module_version(name),
            }
        except Exception as exc:
            checks[name] = {
                "available": False,
                "error": _safe_error(exc),
            }
    return checks


def _torch_checks(cuda_device_id: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": _safe_error(exc)}

    payload: dict[str, Any] = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_built": bool(torch.backends.cuda.is_built()),
    }
    try:
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
    except Exception as exc:
        payload["cuda_available"] = False
        payload["cuda_device_count"] = 0
        payload["cuda_query_error"] = _safe_error(exc)

    if payload["cuda_available"] and payload["cuda_device_count"] > cuda_device_id:
        try:
            payload["cuda_device_name"] = str(torch.cuda.get_device_name(cuda_device_id))
        except Exception as exc:
            payload["cuda_device_name_error"] = _safe_error(exc)
    return payload


def _ctranslate2_checks(cuda_device_id: int) -> dict[str, Any]:
    try:
        import ctranslate2
    except Exception as exc:
        return {"available": False, "error": _safe_error(exc)}

    payload: dict[str, Any] = {
        "available": True,
        "version": str(ctranslate2.__version__),
    }

    try:
        payload["cuda_device_count"] = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        payload["cuda_device_count"] = 0
        payload["cuda_device_count_error"] = _safe_error(exc)

    for device in ("cpu", "cuda"):
        key = f"supported_compute_types_{device}"
        try:
            payload[key] = sorted(str(item) for item in ctranslate2.get_supported_compute_types(device))
        except Exception as exc:
            payload[key] = []
            payload[f"{key}_error"] = _safe_error(exc)

    payload["cuda_device_ready"] = payload.get("cuda_device_count", 0) > cuda_device_id
    return payload


def _model_checks(model_path: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provided": bool(model_path),
        "path": model_path,
        "valid": False,
    }
    if not model_path:
        payload["error"] = "No model path provided"
        return payload

    model_dir = Path(model_path).expanduser()
    valid, error = validate_faster_whisper_model_directory(model_dir)
    payload["valid"] = bool(valid)
    if error:
        payload["error"] = error
    payload["exists"] = model_dir.exists()
    payload["is_dir"] = model_dir.is_dir()
    payload["config_json"] = (model_dir / "config.json").is_file()
    payload["weight_files"] = [
        name
        for name in ("model.bin", "model.safetensors", "model.bin.index.json", "model.safetensors.index.json")
        if (model_dir / name).is_file()
    ]
    return payload


def _runtime_checks() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg_path": shutil.which("ffmpeg") or "",
        "nvidia_smi_path": shutil.which("nvidia-smi") or "",
    }


def run_checks(*, model_path: str, compute_device: str, cuda_device_id: int, compute_type: str) -> dict[str, Any]:
    package_checks = _package_checks()
    torch_checks = _torch_checks(cuda_device_id)
    ct2_checks = _ctranslate2_checks(cuda_device_id)
    model_checks = _model_checks(model_path)

    normalized_compute_type = (compute_type or "auto").strip().lower()
    compute_type_valid = normalized_compute_type in VALID_FASTER_WHISPER_COMPUTE_TYPES

    failures: list[str] = []
    warnings: list[str] = []

    for dep in ("faster_whisper", "ctranslate2", "torch"):
        if not package_checks.get(dep, {}).get("available"):
            failures.append(f"Missing dependency: {dep}")

    if not model_checks["valid"]:
        failures.append(str(model_checks.get("error", "Model directory is invalid")))

    if not compute_type_valid:
        allowed = ", ".join(sorted(VALID_FASTER_WHISPER_COMPUTE_TYPES))
        failures.append(f"Invalid compute_type '{compute_type}'. Allowed values: {allowed}")

    normalized_compute_device = (compute_device or "auto").strip().lower()
    if normalized_compute_device == "cuda":
        if not ct2_checks.get("cuda_device_ready"):
            failures.append(
                f"compute_device=cuda requested but CTranslate2 sees no CUDA device at index {cuda_device_id}"
            )
    elif normalized_compute_device == "auto" and not ct2_checks.get("cuda_device_ready"):
        warnings.append("CUDA not available to CTranslate2; faster-whisper will run on CPU")

    if torch_checks.get("available") and not torch_checks.get("cuda_available", False):
        warnings.append("torch CUDA is unavailable in this environment")

    passed = not failures
    return {
        "passed": passed,
        "failures": failures,
        "warnings": warnings,
        "inputs": {
            "model_path": model_path,
            "compute_device": normalized_compute_device,
            "cuda_device_id": cuda_device_id,
            "compute_type": normalized_compute_type,
        },
        "runtime": _runtime_checks(),
        "packages": package_checks,
        "torch": torch_checks,
        "ctranslate2": ct2_checks,
        "model": model_checks,
    }


def _print_human_report(payload: dict[str, Any]) -> None:
    status = "PASS" if payload["passed"] else "FAIL"
    print(f"[faster-whisper preflight] {status}")
    if payload["failures"]:
        for item in payload["failures"]:
            print(f"- failure: {item}")
    if payload["warnings"]:
        for item in payload["warnings"]:
            print(f"- warning: {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local faster-whisper runtime prerequisites.")
    parser.add_argument("--model-path", default="", help="Path to a CTranslate2 faster-whisper model directory")
    parser.add_argument("--compute-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda-device-id", type=int, default=0)
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    payload = run_checks(
        model_path=args.model_path,
        compute_device=args.compute_device,
        cuda_device_id=max(0, int(args.cuda_device_id)),
        compute_type=args.compute_type,
    )

    if not args.json:
        _print_human_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
