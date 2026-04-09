#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS, ensure_directories
from backend.app.job_store import JOB_STORE
from backend.app.schemas import SpeakerMappingItem
from backend.pipeline.audio import extract_segment
from backend.pipeline.diarization import DiarizationSegment
from backend.pipeline.orchestrator import ORCHESTRATOR
from backend.pipeline.template_manager import TemplateManager
from backend.pipeline.transcription import VoxtralTranscriber, transcribe_segments


DEFAULT_FIXTURES = [
    ROOT_DIR / "audio_trace_for_testing" / "output_1min.mp3",
    ROOT_DIR / "audio_trace_for_testing" / "output_3min.mp3",
    ROOT_DIR / "audio_trace_for_testing" / "output_5min.mp3",
    ROOT_DIR / "audio_trace_for_testing" / "xsim_meeting_10min.aac",
]


ENV_TO_SETTINGS = {
    "AUTOMOM_COMPUTE_DEVICE": ("compute_device", str),
    "AUTOMOM_CUDA_DEVICE_ID": ("cuda_device_id", int),
    "AUTOMOM_VOXTRAL_BIN": ("voxtral_binary", str),
    "AUTOMOM_VOXTRAL_MODEL": ("voxtral_model_path", str),
    "AUTOMOM_VOXTRAL_THREADS": ("voxtral_threads", int),
    "AUTOMOM_VOXTRAL_PROCESSORS": ("voxtral_processors", int),
    "AUTOMOM_VOXTRAL_GPU_LAYERS": ("voxtral_gpu_layers", int),
    "AUTOMOM_TRANSCRIPTION_MERGE_GAP_S": ("transcription_merge_gap_s", float),
    "AUTOMOM_TRANSCRIPTION_MAX_CHUNK_S": ("transcription_max_chunk_s", float),
    "AUTOMOM_TRANSCRIPTION_KEEP_SEGMENT_AUDIO": (
        "transcription_keep_segment_audio",
        lambda value: value.strip().lower() in {"1", "true", "yes", "on"},
    ),
}


def _apply_local_env() -> None:
    """! @brief Apply local env.
    """
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


def _parse_stage6_seconds(logs: list[str]) -> float | None:
    """! @brief Parse stage6 seconds.
    @param logs Value for logs.
    @return Result produced by the operation.
    """
    stage6_ts = None
    stage7_ts = None
    for entry in logs:
        if not entry.startswith("[") or "] " not in entry:
            continue
        timestamp, message = entry[1:].split("] ", 1)
        if message.startswith("Stage 6/9:"):
            stage6_ts = timestamp
        if message.startswith("Stage 7/9:"):
            stage7_ts = timestamp
            break
    if stage6_ts is None or stage7_ts is None:
        return None
    return time.mktime(time.strptime(stage7_ts[:19], "%Y-%m-%dT%H:%M:%S")) - time.mktime(
        time.strptime(stage6_ts[:19], "%Y-%m-%dT%H:%M:%S")
    )


def _collect_metrics(job_id: str, wall_clock_s: float) -> dict[str, Any]:
    """! @brief Collect metrics.
    @param job_id Identifier of the job being processed.
    @param wall_clock_s Value for wall clock s.
    @return Dictionary produced by the operation.
    """
    state = JOB_STORE.get_state(job_id)
    runtime_path = state.artifact_paths.get("transcription_runtime")
    diarization_path = state.artifact_paths.get("diarization")
    transcript_path = state.artifact_paths.get("segments_transcript")
    metadata_path = state.artifact_paths.get("audio_normalized")

    diarization_count = None
    transcript_count = None
    runtime_payload: dict[str, Any] = {}
    audio_duration_s = None

    if diarization_path:
        diarization_count = len(json.loads(Path(diarization_path).read_text(encoding="utf-8")))
    if transcript_path:
        transcript_count = len(json.loads(Path(transcript_path).read_text(encoding="utf-8")))
    if runtime_path:
        runtime_payload = json.loads(Path(runtime_path).read_text(encoding="utf-8"))
    if metadata_path:
        import soundfile as sf

        audio_data, sample_rate = sf.read(str(metadata_path), always_2d=False)
        audio_duration_s = len(audio_data) / sample_rate

    stage6_s = _parse_stage6_seconds(state.logs)
    chunk_count = None
    if runtime_payload:
        chunk_dir = Path(state.artifact_paths["segments_transcript"]).parent / "transcription_segments"
        chunk_count = len(list(chunk_dir.glob("*.wav"))) if chunk_dir.exists() else 0

    return {
        "job_id": job_id,
        "status": state.status,
        "wall_clock_s": round(wall_clock_s, 3),
        "stage6_s": None if stage6_s is None else round(stage6_s, 3),
        "audio_duration_s": None if audio_duration_s is None else round(audio_duration_s, 3),
        "stage6_rtf": None
        if stage6_s is None or not audio_duration_s
        else round(stage6_s / audio_duration_s, 3),
        "diarization_segments": diarization_count,
        "transcript_segments": transcript_count,
        "temp_chunk_files": chunk_count,
        "runtime": runtime_payload,
    }


def _run_benchmark(audio_path: Path, template_id: str, title_prefix: str) -> dict[str, Any]:
    """! @brief Run benchmark.
    @param audio_path Path to the audio file.
    @param template_id Identifier of the template.
    @param title_prefix Value for title prefix.
    @return Dictionary produced by the operation.
    """
    runtime = JOB_STORE.create_job(
        audio_path=audio_path,
        original_filename=audio_path.name,
        template_id=template_id,
        language_mode="auto",
        title=f"{title_prefix} {audio_path.stem}",
    )
    job_id = runtime.state.job_id
    started = time.monotonic()
    ORCHESTRATOR.submit(job_id)

    terminal = {"completed", "failed", "cancelled"}
    while True:
        state = JOB_STORE.get_state(job_id)
        if state.status == "waiting_speaker_input" and state.speaker_info:
            mappings = [
                SpeakerMappingItem(
                    speaker_id=speaker.speaker_id,
                    name=speaker.suggested_name or speaker.speaker_id,
                    save_voice_profile=False,
                )
                for speaker in state.speaker_info.speakers
            ]
            JOB_STORE.submit_speaker_mapping(job_id, mappings)
        if state.status in terminal:
            return _collect_metrics(job_id, time.monotonic() - started)
        time.sleep(0.5)


def _run_stage6_benchmark(job_dir: Path) -> dict[str, Any]:
    """! @brief Run stage6 benchmark.
    @param job_dir Value for job dir.
    @return Dictionary produced by the operation.
    """
    normalized_audio_path = job_dir / "audio_normalized.wav"
    diarization_path = job_dir / "diarization.json"
    if not normalized_audio_path.exists() or not diarization_path.exists():
        raise FileNotFoundError(f"{job_dir} must contain audio_normalized.wav and diarization.json")

    diarization_segments = json.loads(diarization_path.read_text(encoding="utf-8"))
    mapping_path = job_dir / "speaker_mapping.json"
    if mapping_path.exists():
        speaker_map = {
            item["speaker_id"]: item["name"].strip() or item["speaker_id"]
            for item in json.loads(mapping_path.read_text(encoding="utf-8"))
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
    temp_dir = job_dir / "benchmark_transcription_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segment_jobs = []
    for idx, segment in enumerate(chunks, start=1):
        segment_path = temp_dir / f"segment_{idx:04d}.wav"
        extract_segment(
            normalized_audio_path,
            segment_path,
            max(0.0, float(segment["start_s"]) - 0.2),
            float(segment["end_s"]) + 0.2,
            ffmpeg_bin=SETTINGS.ffmpeg_bin,
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

    transcriber = VoxtralTranscriber(
        SETTINGS.voxtral_binary,
        SETTINGS.voxtral_model_path,
        compute_device=SETTINGS.compute_device,
        cuda_device_id=SETTINGS.cuda_device_id,
        gpu_layers=SETTINGS.voxtral_gpu_layers,
        threads=SETTINGS.voxtral_threads,
        processors=SETTINGS.voxtral_processors,
    )
    started = time.monotonic()
    transcript_segments = transcribe_segments(transcriber, segment_jobs)
    wall_clock_s = time.monotonic() - started
    if not SETTINGS.transcription_keep_segment_audio:
        for item in segment_jobs:
            Path(str(item["segment_path"])).unlink(missing_ok=True)

    import soundfile as sf

    audio_data, sample_rate = sf.read(str(normalized_audio_path), always_2d=False)
    audio_duration_s = len(audio_data) / sample_rate
    return {
        "job_id": job_dir.name,
        "status": "completed",
        "wall_clock_s": round(wall_clock_s, 3),
        "stage6_s": round(wall_clock_s, 3),
        "audio_duration_s": round(audio_duration_s, 3),
        "stage6_rtf": round(wall_clock_s / audio_duration_s, 3),
        "diarization_segments": len(diarization_segments),
        "transcript_segments": len(transcript_segments),
        "temp_chunk_files": len(segment_jobs) if SETTINGS.transcription_keep_segment_audio else 0,
        "runtime": transcriber.runtime_report(),
    }


def main() -> int:
    """! @brief Run the module entry point.
    @return int result produced by the operation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_paths", nargs="*", type=Path)
    parser.add_argument("--job-dir", action="append", type=Path, default=[])
    parser.add_argument("--template-id", default="default")
    parser.add_argument("--title-prefix", default="Benchmark")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    _apply_local_env()
    ensure_directories()
    TemplateManager()

    audio_paths = list(args.audio_paths)
    if not audio_paths and not args.job_dir:
        audio_paths = [path for path in DEFAULT_FIXTURES if path.exists()]
    if not audio_paths and not args.job_dir:
        raise SystemExit("No benchmark audio files found.")

    results = []
    for job_dir in args.job_dir:
        print(f"Benchmarking transcription stage from {job_dir} ...")
        results.append(_run_stage6_benchmark(job_dir))
    for audio_path in audio_paths:
        print(f"Benchmarking {audio_path} ...")
        results.append(_run_benchmark(audio_path, template_id=args.template_id, title_prefix=args.title_prefix))

    payload = {"generated_at": time.time(), "results": results}
    output = json.dumps(payload, indent=2)
    print(output)
    if args.json_out is not None:
        args.json_out.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
