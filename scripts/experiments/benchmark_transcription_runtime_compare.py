#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS
from backend.app.schemas import SpeakerMappingItem
from backend.pipeline.audio import extract_segment
from backend.pipeline.diarization import DiarizationSegment
from backend.pipeline.orchestrator import ORCHESTRATOR
from backend.pipeline.transcription import (
    FasterWhisperTranscriber,
    TranscriptionError,
    WhisperCppTranscriber,
    transcribe_segments,
)

ENV_TO_SETTINGS = {
    "AUTOMOM_COMPUTE_DEVICE": ("compute_device", str),
    "AUTOMOM_CUDA_DEVICE_ID": ("cuda_device_id", int),
    "AUTOMOM_TRANSCRIPTION_BIN": ("transcription_binary", str),
    "AUTOMOM_TRANSCRIPTION_MODEL": ("transcription_model_path", str),
    "AUTOMOM_TRANSCRIPTION_THREADS": ("transcription_threads", int),
    "AUTOMOM_TRANSCRIPTION_PROCESSORS": ("transcription_processors", int),
    "AUTOMOM_TRANSCRIPTION_GPU_LAYERS": ("transcription_gpu_layers", int),
    "AUTOMOM_TRANSCRIPTION_MERGE_GAP_S": ("transcription_merge_gap_s", float),
    "AUTOMOM_TRANSCRIPTION_MAX_CHUNK_S": ("transcription_max_chunk_s", float),
}


def _apply_local_env() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip('"').strip("'")
        resolved_value = os.environ.setdefault(key, value)
        target = ENV_TO_SETTINGS.get(key)
        if target is None:
            continue
        attr_name, caster = target
        object.__setattr__(SETTINGS, attr_name, caster(resolved_value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_segment_jobs(
    job_dir: Path, *, ffmpeg_bin: str, keep_segment_audio: bool
) -> tuple[list[dict[str, Any]], float, Path | None]:
    normalized_audio_path = job_dir / "audio_normalized.wav"
    diarization_path = job_dir / "diarization.json"
    metadata_path = job_dir / "audio_metadata.json"
    if not normalized_audio_path.exists() or not diarization_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("job_dir must contain audio_normalized.wav, diarization.json, and audio_metadata.json")

    diarization_segments = _read_json(diarization_path)
    metadata = _read_json(metadata_path)
    audio_duration_s = float(metadata.get("duration_s") or 0.0)

    mapping_path = job_dir / "speaker_mapping.json"
    if mapping_path.exists():
        mapping_payload = _read_json(mapping_path)
        mapping_rows = mapping_payload.get("expanded_mappings", mapping_payload.get("mappings", [])) if isinstance(mapping_payload, dict) else mapping_payload
        speaker_map = {
            str(item["speaker_id"]): str(item.get("name", "")).strip() or str(item["speaker_id"])
            for item in mapping_rows
        }
    else:
        speaker_map = {
            str(item["speaker_id"]): str(item["speaker_id"])
            for item in diarization_segments
        }

    collapsed = ORCHESTRATOR._collapse_labeled_segments(
        [
            DiarizationSegment(
                speaker_id=str(item["speaker_id"]),
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            )
            for item in diarization_segments
        ],
        speaker_map,
    )

    chunks = ORCHESTRATOR._plan_transcription_chunks(
        collapsed,
        max_gap_s=SETTINGS.transcription_merge_gap_s,
        max_chunk_s=SETTINGS.transcription_max_chunk_s,
    )

    segment_dir = job_dir / "benchmark_compare_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_jobs: list[dict[str, Any]] = []
    for idx, segment in enumerate(chunks, start=1):
        segment_path = segment_dir / f"segment_{idx:04d}.wav"
        extract_segment(
            normalized_audio_path,
            segment_path,
            max(0.0, float(segment["start_s"]) - 0.2),
            float(segment["end_s"]) + 0.2,
            ffmpeg_bin=ffmpeg_bin,
        )
        segment_jobs.append(
            {
                "segment_path": str(segment_path),
                "speaker_id": str(segment["speaker_id"]),
                "speaker_name": str(segment["speaker_name"]),
                "start_s": float(segment["start_s"]),
                "end_s": float(segment["end_s"]),
            }
        )

    cleanup_dir = None if keep_segment_audio else segment_dir
    return segment_jobs, audio_duration_s, cleanup_dir


def _run_runtime(name: str, transcriber, segment_jobs: list[dict[str, Any]], audio_duration_s: float) -> dict[str, Any]:
    if not transcriber.available():
        raise TranscriptionError(f"{name} runtime is unavailable: {transcriber.runtime_summary()}")
    started = time.monotonic()
    transcript = transcribe_segments(transcriber, segment_jobs)
    elapsed_s = time.monotonic() - started
    words = sum(len(str(item.get("text", "")).split()) for item in transcript)
    return {
        "runtime": name,
        "elapsed_s": round(elapsed_s, 3),
        "rtf": round(elapsed_s / audio_duration_s, 4) if audio_duration_s > 0 else None,
        "transcript_segments": len(transcript),
        "transcript_words": words,
        "runtime_summary": transcriber.runtime_summary(),
        "runtime_report": transcriber.runtime_report(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare whisper.cpp vs faster-whisper stage-6 runtime on same job chunks.")
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument(
        "--faster-model-path",
        default=str(ROOT_DIR / "data" / "models" / "faster-whisper" / "systran-faster-whisper-small"),
    )
    parser.add_argument("--faster-compute-device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--faster-compute-type", default="auto")
    parser.add_argument("--whisper-binary", default=SETTINGS.transcription_binary)
    parser.add_argument("--whisper-model", default=SETTINGS.transcription_model_path)
    parser.add_argument("--keep-segment-audio", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    _apply_local_env()

    segment_jobs, audio_duration_s, cleanup_dir = _build_segment_jobs(
        args.job_dir,
        ffmpeg_bin=SETTINGS.ffmpeg_bin,
        keep_segment_audio=args.keep_segment_audio,
    )

    whisper_cpp = WhisperCppTranscriber(
        args.whisper_binary,
        args.whisper_model,
        compute_device=SETTINGS.compute_device,
        cuda_device_id=SETTINGS.cuda_device_id,
        gpu_layers=SETTINGS.transcription_gpu_layers,
        threads=SETTINGS.transcription_threads,
        processors=SETTINGS.transcription_processors,
    )
    faster = FasterWhisperTranscriber(
        args.faster_model_path,
        compute_device=args.faster_compute_device,
        cuda_device_id=SETTINGS.cuda_device_id,
        compute_type=args.faster_compute_type,
    )

    faster_result = _run_runtime("faster-whisper", faster, segment_jobs, audio_duration_s)
    whisper_result = _run_runtime("whisper.cpp", whisper_cpp, segment_jobs, audio_duration_s)

    speedup = None
    if faster_result["elapsed_s"] > 0:
        speedup = round(float(whisper_result["elapsed_s"]) / float(faster_result["elapsed_s"]), 3)

    payload = {
        "job_dir": str(args.job_dir),
        "audio_duration_s": round(audio_duration_s, 3),
        "segment_count": len(segment_jobs),
        "faster_whisper": faster_result,
        "whisper_cpp": whisper_result,
        "whisper_cpp_over_faster_whisper_speedup": speedup,
    }

    if cleanup_dir and cleanup_dir.exists():
        for item in segment_jobs:
            Path(str(item["segment_path"])).unlink(missing_ok=True)

    output = json.dumps(payload, indent=2)
    print(output)
    if args.json_out:
        args.json_out.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
