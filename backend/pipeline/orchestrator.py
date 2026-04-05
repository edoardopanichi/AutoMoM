from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
from typing import Any

from backend.app.config import SETTINGS
from backend.app.job_store import JOB_STORE, OpenAIJobConfig, ensure_job_artifact_dir, write_json
from backend.app.schemas import JobSpeakerInfo, SpeakerSnippet, SpeakerState
from backend.pipeline.audio import extract_segment, normalize_audio, validate_audio_input
from backend.pipeline.diarization import DiarizationResult, DiarizationSegment, diarize
from backend.pipeline.formatter import Formatter
from backend.pipeline.openai_client import OpenAIAPIError, OpenAIClient, OpenAIDiarizationResult
from backend.pipeline.snippets import extract_snippets, pick_snippet_ranges
from backend.pipeline.template_manager import TEMPLATE_MANAGER
from backend.pipeline.transcription import OpenAITranscriber, TranscriptionError, VoxtralTranscriber, transcribe_segments
from backend.pipeline.vad import detect_speech_regions
from backend.profiles.manager import VOICE_PROFILE_MANAGER


STAGES = [
    "Validate/Normalize",
    "VAD",
    "Diarization",
    "Snippet extraction",
    "Speaker naming",
    "Transcription",
    "Transcript assembly",
    "MoM formatting",
    "Export",
]


class CancelledError(RuntimeError):
    pass


class PipelineOrchestrator:
    STAGE_KEYS = [
        "validate_normalize",
        "vad",
        "diarization",
        "snippet_extraction",
        "speaker_naming",
        "transcription",
        "transcript_assembly",
        "mom_formatting",
        "export",
    ]

    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=SETTINGS.max_workers)
        self._futures: dict[str, concurrent.futures.Future[None]] = {}

    def submit(self, job_id: str) -> None:
        future = self._executor.submit(self._run_job, job_id)
        self._futures[job_id] = future

    def cancel(self, job_id: str) -> None:
        JOB_STORE.cancel(job_id)

    def _run_job(self, job_id: str) -> None:
        run_started_at = datetime.now(timezone.utc)
        run_started_monotonic = time.monotonic()
        active_stage_key: str | None = None
        active_stage_name: str | None = None
        active_stage_started_at: datetime | None = None
        active_stage_started_monotonic: float | None = None
        stage_timings: dict[str, dict[str, object]] = {}
        runtime = None
        job_dir: Path | None = None
        diarization_result: DiarizationResult | None = None
        transcript_segments: list[dict[str, object]] = []
        audio_metadata_payload: dict[str, object] | None = None
        transcription_runtime_payload: dict[str, object] | None = None
        formatter: Formatter | None = None
        mapping_items = None
        try:
            JOB_STORE.mark_running(job_id)
            runtime = JOB_STORE.get_runtime(job_id)
            job_dir = ensure_job_artifact_dir(job_id)
            api_config = runtime.api_config
            api_diarization_result: OpenAIDiarizationResult | None = None
            if api_config is not None:
                JOB_STORE.append_log(
                    job_id,
                    (
                        "Cloud execution plan: "
                        f"diarization={api_config.diarization_execution}, "
                        f"transcription={api_config.transcription_execution}, "
                        f"formatter={api_config.formatter_execution}"
                    ),
                )

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 0)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(0)
            )
            JOB_STORE.append_log(job_id, "Stage 1/9: validating, normalizing, and denoising audio")
            validate_audio_input(runtime.audio_path)
            normalized_audio_path = job_dir / "audio_normalized.wav"
            audio_metadata_payload = normalize_audio(runtime.audio_path, normalized_audio_path, ffmpeg_bin=SETTINGS.ffmpeg_bin)
            write_json(job_dir / "audio_metadata.json", audio_metadata_payload)
            JOB_STORE.set_artifact(job_id, "audio_normalized", normalized_audio_path)
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(0, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 1)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(1)
            )
            JOB_STORE.append_log(job_id, "Stage 2/9: voice activity detection")
            speech_regions = detect_speech_regions(normalized_audio_path)
            vad_payload = [asdict(item) for item in speech_regions]
            write_json(job_dir / "vad_regions.json", vad_payload)
            JOB_STORE.set_artifact(job_id, "vad_regions", job_dir / "vad_regions.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(1, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 2)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(2)
            )
            JOB_STORE.append_log(job_id, "Stage 3/9: speaker diarization")
            if api_config is not None and api_config.diarization_execution == "api":
                api_diarization_result = self._diarize_with_openai(
                    runtime=runtime,
                    normalized_audio_path=normalized_audio_path,
                )
                diarization_result = self._diarization_result_from_openai(api_diarization_result)
            else:
                min_speakers = SETTINGS.diarization_min_speakers if SETTINGS.diarization_min_speakers > 0 else None
                max_speakers = SETTINGS.diarization_max_speakers if SETTINGS.diarization_max_speakers > 0 else None
                diarization_result = diarize(
                    normalized_audio_path,
                    speech_regions,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    max_chunk_s=SETTINGS.diarization_max_chunk_s,
                    backend=SETTINGS.diarization_backend,
                    model_path=Path(SETTINGS.diarization_model_path),
                    pipeline_path=SETTINGS.diarization_pipeline_path,
                    embedding_model=SETTINGS.diarization_embedding_model,
                    compute_device=SETTINGS.compute_device,
                    cuda_device_id=SETTINGS.cuda_device_id,
                )
            write_json(job_dir / "diarization.json", diarization_result.to_json())
            JOB_STORE.set_artifact(job_id, "diarization", job_dir / "diarization.json")
            JOB_STORE.append_log(job_id, f"Diarization mode: {diarization_result.mode}")
            if diarization_result.details:
                JOB_STORE.append_log(job_id, f"Diarization details: {diarization_result.details}")
            JOB_STORE.append_log(job_id, f"Detected speakers: {diarization_result.speaker_count}")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(2, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 3)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(3)
            )
            JOB_STORE.append_log(job_id, "Stage 4/9: extracting labeling snippets")
            snippet_ranges = pick_snippet_ranges(diarization_result.segments)
            snippets = extract_snippets(
                source_audio_path=normalized_audio_path,
                output_dir=job_dir / "snippets",
                snippet_ranges=snippet_ranges,
                ffmpeg_bin=SETTINGS.ffmpeg_bin,
            )
            write_json(
                job_dir / "snippets.json",
                [
                    {
                        "speaker_id": item.speaker_id,
                        "path": str(item.path),
                        "start_s": item.start_s,
                        "end_s": item.end_s,
                    }
                    for item in snippets
                ],
            )
            JOB_STORE.set_artifact(job_id, "snippets", job_dir / "snippets.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(3, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 4)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(4)
            )
            JOB_STORE.append_log(job_id, "Stage 5/9: waiting for speaker naming")
            segments_by_speaker = self._group_segments_by_speaker(diarization_result.segments)
            speaker_info = self._build_speaker_info(job_id, normalized_audio_path, segments_by_speaker, snippets)
            JOB_STORE.set_waiting_for_speaker_input(job_id, speaker_info)
            mapping_items = JOB_STORE.wait_for_mapping(job_id)
            if mapping_items is None:
                raise CancelledError("Job cancelled during speaker naming")
            speaker_map = {item.speaker_id: item.name.strip() or item.speaker_id for item in mapping_items}
            write_json(
                job_dir / "speaker_mapping.json",
                [item.model_dump() for item in mapping_items],
            )
            JOB_STORE.set_artifact(job_id, "speaker_mapping", job_dir / "speaker_mapping.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(4, 100.0))

            for mapping in mapping_items:
                if not mapping.save_voice_profile:
                    continue
                segments = segments_by_speaker.get(mapping.speaker_id, [])[:8]
                embedding = VOICE_PROFILE_MANAGER.compute_embedding_from_segments(normalized_audio_path, segments)
                VOICE_PROFILE_MANAGER.upsert_profile(mapping.name, embedding)
                JOB_STORE.append_log(job_id, f"Saved voice profile for {mapping.name}")
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 5)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(5)
            )
            JOB_STORE.append_log(job_id, "Stage 6/9: transcribing diarized segments")
            transcription_dir = ensure_job_artifact_dir(job_id, "transcription_segments")
            segment_jobs: list[dict[str, object]] = []
            segments_for_transcription = self._collapse_labeled_segments(diarization_result.segments, speaker_map)
            if len(segments_for_transcription) != len(diarization_result.segments):
                JOB_STORE.append_log(
                    job_id,
                    (
                        "Post-label diarization consolidation applied: "
                        f"{len(diarization_result.segments)} -> {len(segments_for_transcription)} segments"
                    ),
                )
            transcription_chunks = self._plan_transcription_chunks(
                segments_for_transcription,
                max_gap_s=SETTINGS.transcription_merge_gap_s,
                max_chunk_s=SETTINGS.transcription_max_chunk_s,
            )
            if len(transcription_chunks) != len(segments_for_transcription):
                JOB_STORE.append_log(
                    job_id,
                    (
                        "Transcription chunk planner applied: "
                        f"{len(segments_for_transcription)} -> {len(transcription_chunks)} chunks"
                    ),
                )
            if transcription_chunks:
                durations = [float(item["end_s"]) - float(item["start_s"]) for item in transcription_chunks]
                JOB_STORE.append_log(
                    job_id,
                    (
                        "Transcription chunk stats: "
                        f"avg={statistics.mean(durations):.2f}s "
                        f"max={max(durations):.2f}s"
                    ),
                )
            if (
                SETTINGS.transcription_max_segments > 0
                and len(transcription_chunks) > SETTINGS.transcription_max_segments
            ):
                transcription_chunks = transcription_chunks[: SETTINGS.transcription_max_segments]
                JOB_STORE.append_log(
                    job_id,
                    f"Transcription segment cap applied: {SETTINGS.transcription_max_segments}",
                )

            def _segment_progress(done: int, total: int) -> None:
                stage_percent = (done / max(1, total)) * 100
                JOB_STORE.set_transcription_progress(
                    job_id,
                    stage_percent=stage_percent,
                    completed=done,
                    total=total,
                    overall_percent=self._overall(5, stage_percent),
                )

            if api_config is not None and api_config.transcription_execution == "api":
                if api_diarization_result is not None:
                    transcript_segments = self._transcript_segments_from_openai_diarization(
                        api_diarization_result,
                        speaker_map,
                    )
                    _segment_progress(len(transcript_segments), len(transcript_segments))
                    JOB_STORE.append_log(
                        job_id,
                        f"Transcription executed via OpenAI diarized transcript ({api_config.diarization_model})",
                    )
                else:
                    transcriber = OpenAITranscriber(api_config.api_key, api_config.transcription_model)
                    for idx, segment in enumerate(transcription_chunks, start=1):
                        padding = 0.2
                        start_s = max(0.0, float(segment["start_s"]) - padding)
                        end_s = float(segment["end_s"]) + padding
                        segment_path = transcription_dir / f"segment_{idx:04d}.wav"
                        extract_segment(
                            normalized_audio_path,
                            segment_path,
                            start_s,
                            end_s,
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
                    JOB_STORE.append_log(
                        job_id,
                        f"Transcription uses OpenAI model {api_config.transcription_model}",
                    )
                    transcript_segments = transcribe_segments(
                        transcriber,
                        segment_jobs,
                        progress_callback=_segment_progress,
                    )
            else:
                transcriber = VoxtralTranscriber(
                    SETTINGS.voxtral_binary,
                    SETTINGS.voxtral_model_path,
                    compute_device=SETTINGS.compute_device,
                    cuda_device_id=SETTINGS.cuda_device_id,
                    gpu_layers=SETTINGS.voxtral_gpu_layers,
                    threads=SETTINGS.voxtral_threads,
                    processors=SETTINGS.voxtral_processors,
                )
                if not transcriber.available():
                    raise TranscriptionError(
                        "ASR runtime unavailable. Fix: set AUTOMOM_VOXTRAL_BIN to a valid ASR binary "
                        "and AUTOMOM_VOXTRAL_MODEL to an existing model file before starting a job."
                    )
                for idx, segment in enumerate(transcription_chunks, start=1):
                    padding = 0.2
                    start_s = max(0.0, float(segment["start_s"]) - padding)
                    end_s = float(segment["end_s"]) + padding
                    segment_path = transcription_dir / f"segment_{idx:04d}.wav"
                    extract_segment(normalized_audio_path, segment_path, start_s, end_s, ffmpeg_bin=SETTINGS.ffmpeg_bin)
                    segment_jobs.append(
                        {
                            "segment_path": str(segment_path),
                            "speaker_id": str(segment["speaker_id"]),
                            "speaker_name": str(segment["speaker_name"]),
                            "start_s": float(segment["start_s"]),
                            "end_s": float(segment["end_s"]),
                        }
                    )
                initial_runtime = transcriber.runtime_report()
                JOB_STORE.append_log(
                    job_id,
                    (
                        "ASR runtime detected; transcription uses external binary "
                        f"(requested={initial_runtime['requested_mode']} "
                        f"available={initial_runtime['available_mode']} "
                        f"binary={initial_runtime['binary_path']})"
                    ),
                )
                transcript_segments = transcribe_segments(transcriber, segment_jobs, progress_callback=_segment_progress)
                transcription_runtime_payload = transcriber.runtime_report()
                write_json(job_dir / "transcription_runtime.json", transcription_runtime_payload)
                JOB_STORE.set_artifact(job_id, "transcription_runtime", job_dir / "transcription_runtime.json")
                JOB_STORE.append_log(
                    job_id,
                    f"ASR runtime result: {transcriber.runtime_summary()}",
                )
                if not SETTINGS.transcription_keep_segment_audio:
                    for item in segment_jobs:
                        Path(str(item["segment_path"])).unlink(missing_ok=True)
            write_json(job_dir / "segments_transcript.json", transcript_segments)
            JOB_STORE.set_artifact(job_id, "segments_transcript", job_dir / "segments_transcript.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(5, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 6)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(6)
            )
            JOB_STORE.append_log(job_id, "Stage 7/9: assembling transcript")
            transcript_payload = {
                "speakers": sorted(set(item["speaker_name"] for item in transcript_segments)),
                "segments": transcript_segments,
            }
            write_json(job_dir / "transcript.json", transcript_payload)
            JOB_STORE.set_artifact(job_id, "transcript", job_dir / "transcript.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(6, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 7)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(7)
            )
            JOB_STORE.append_log(job_id, "Stage 8/9: formatting minutes of meeting")
            use_api_formatter = api_config is not None and api_config.formatter_execution == "api"
            use_legacy_formatter_command = SETTINGS.formatter_backend == "command" and not use_api_formatter
            JOB_STORE.append_log(
                job_id,
                (
                    f"Formatter backend: OpenAI ({api_config.formatter_model})"
                    if use_api_formatter
                    else f"Formatter backend: {'command' if use_legacy_formatter_command else 'ollama'}"
                ),
            )
            formatter = Formatter(
                command_template=SETTINGS.formatter_command if use_legacy_formatter_command else "",
                model_path=SETTINGS.formatter_model_path if use_legacy_formatter_command else "",
                ollama_host=SETTINGS.ollama_host,
                ollama_model=SETTINGS.formatter_ollama_model,
                openai_api_key=api_config.api_key if use_api_formatter else "",
                openai_model=api_config.formatter_model if use_api_formatter else "",
                timeout_s=SETTINGS.formatter_timeout_s,
            )
            title = runtime.title or runtime.audio_path.stem
            speakers = transcript_payload["speakers"]
            mom_path = job_dir / "mom.md"
            structured, prompt = formatter.write_model_output_to_mom(
                transcript=transcript_segments,
                speakers=speakers,
                title=title,
                template_id=runtime.template_id,
                output_path=mom_path,
            )
            JOB_STORE.append_log(job_id, f"Formatter mode: {formatter.last_mode}")
            if formatter.last_stdout:
                (job_dir / "formatter_stdout.txt").write_text(formatter.last_stdout, encoding="utf-8")
                JOB_STORE.set_artifact(job_id, "formatter_stdout", job_dir / "formatter_stdout.txt")
            if formatter.last_stderr:
                (job_dir / "formatter_stderr.txt").write_text(formatter.last_stderr, encoding="utf-8")
                JOB_STORE.set_artifact(job_id, "formatter_stderr", job_dir / "formatter_stderr.txt")
            if formatter.last_raw_output:
                (job_dir / "formatter_raw_output.txt").write_text(formatter.last_raw_output, encoding="utf-8")
                JOB_STORE.set_artifact(job_id, "formatter_raw_output", job_dir / "formatter_raw_output.txt")
            write_json(job_dir / "mom_structured.json", structured)
            (job_dir / "formatter_prompt.txt").write_text(prompt, encoding="utf-8")
            JOB_STORE.set_artifact(job_id, "mom_markdown", mom_path)
            JOB_STORE.set_artifact(job_id, "mom_structured", job_dir / "mom_structured.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(7, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 8)
            active_stage_key, active_stage_name, active_stage_started_at, active_stage_started_monotonic = (
                self._begin_stage_timing(8)
            )
            JOB_STORE.append_log(job_id, "Stage 9/9: export completed")
            export_dir = ensure_job_artifact_dir(job_id, "export")
            export_path = export_dir / "mom.md"
            export_path.write_text((job_dir / "mom.md").read_text(encoding="utf-8"), encoding="utf-8")
            JOB_STORE.set_artifact(job_id, "export_markdown", export_path)
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(8, 100.0))
            self._finish_stage_timing(
                stage_timings,
                active_stage_key,
                active_stage_name,
                active_stage_started_at,
                active_stage_started_monotonic,
            )
            active_stage_key = active_stage_name = None
            active_stage_started_at = None
            active_stage_started_monotonic = None

            JOB_STORE.mark_completed(job_id)
            JOB_STORE.append_log(job_id, "Job completed successfully")
        except CancelledError as exc:
            JOB_STORE.append_log(job_id, str(exc))
            JOB_STORE.cancel(job_id)
        except Exception as exc:  # pragma: no cover - this is fallback guard
            JOB_STORE.mark_failed(job_id, str(exc))
            JOB_STORE.append_log(job_id, f"Job failed: {exc}")
        finally:
            if (
                active_stage_key is not None
                and active_stage_name is not None
                and active_stage_started_at is not None
                and active_stage_started_monotonic is not None
                and active_stage_key not in stage_timings
            ):
                self._finish_stage_timing(
                    stage_timings,
                    active_stage_key,
                    active_stage_name,
                    active_stage_started_at,
                    active_stage_started_monotonic,
                )
            if runtime is not None and job_dir is not None:
                self._write_job_summary(
                    job_id=job_id,
                    runtime=runtime,
                    job_dir=job_dir,
                    run_started_at=run_started_at,
                    run_started_monotonic=run_started_monotonic,
                    stage_timings=stage_timings,
                    diarization_result=diarization_result,
                    mapping_items=mapping_items,
                    transcript_segments=transcript_segments,
                    audio_metadata_payload=audio_metadata_payload,
                    transcription_runtime_payload=transcription_runtime_payload,
                    formatter=formatter,
                )

    def _begin_stage_timing(self, stage_index: int) -> tuple[str, str, datetime, float]:
        return (
            self.STAGE_KEYS[stage_index],
            STAGES[stage_index],
            datetime.now(timezone.utc),
            time.monotonic(),
        )

    @staticmethod
    def _finish_stage_timing(
        stage_timings: dict[str, dict[str, object]],
        stage_key: str | None,
        stage_name: str | None,
        started_at: datetime | None,
        started_monotonic: float | None,
    ) -> None:
        if stage_key is None or stage_name is None or started_at is None or started_monotonic is None:
            return
        finished_at = datetime.now(timezone.utc)
        stage_timings[stage_key] = {
            "stage": stage_name,
            "started_at": started_at.isoformat(timespec="seconds"),
            "completed_at": finished_at.isoformat(timespec="seconds"),
            "duration_s": round(max(0.0, time.monotonic() - started_monotonic), 3),
        }

    def _write_job_summary(
        self,
        *,
        job_id: str,
        runtime,
        job_dir: Path,
        run_started_at: datetime,
        run_started_monotonic: float,
        stage_timings: dict[str, dict[str, object]],
        diarization_result: DiarizationResult | None,
        mapping_items,
        transcript_segments: list[dict[str, object]],
        audio_metadata_payload: dict[str, object] | None,
        transcription_runtime_payload: dict[str, object] | None,
        formatter: Formatter | None,
    ) -> None:
        state = JOB_STORE.get_state(job_id)
        template_name = None
        try:
            template_name = TEMPLATE_MANAGER.load(runtime.template_id).name
        except Exception:
            template_name = None

        speaker_names: dict[str, str] = {}
        if mapping_items is not None:
            speaker_names = {item.speaker_id: item.name.strip() or item.speaker_id for item in mapping_items}
        elif transcript_segments:
            for item in transcript_segments:
                speaker_id = str(item.get("speaker_id", "")).strip()
                speaker_name = str(item.get("speaker_name", "")).strip()
                if speaker_id:
                    speaker_names[speaker_id] = speaker_name or speaker_id

        detected_speaker_ids: list[str] = []
        if diarization_result is not None:
            detected_speaker_ids = sorted({item.speaker_id for item in diarization_result.segments})
        elif state.speaker_info is not None:
            detected_speaker_ids = [item.speaker_id for item in state.speaker_info.speakers]
        elif speaker_names:
            detected_speaker_ids = sorted(speaker_names.keys())

        audio_info: dict[str, object] = {
            "original_filename": runtime.original_filename or runtime.audio_path.name,
            "stored_path": str(runtime.audio_path),
        }
        if runtime.audio_path.exists():
            audio_info["file_size_bytes"] = runtime.audio_path.stat().st_size
        if audio_metadata_payload is not None:
            audio_info["normalized_path"] = str(audio_metadata_payload.get("path", ""))
            audio_info["duration_s"] = audio_metadata_payload.get("duration_s")
            audio_info["sample_rate_hz"] = audio_metadata_payload.get("sample_rate")
            audio_info["channels"] = audio_metadata_payload.get("channels")

        summary = {
            "job_id": job_id,
            "meeting_title": runtime.title or runtime.audio_path.stem,
            "template_id": runtime.template_id,
            "template_name": template_name,
            "status": state.status,
            "created_at": state.created_at.isoformat(timespec="seconds"),
            "completed_at": state.updated_at.isoformat(timespec="seconds"),
            "error": state.error,
            "audio": audio_info,
            "speakers": {
                "count": diarization_result.speaker_count if diarization_result is not None else len(detected_speaker_ids),
                "detected_speaker_ids": detected_speaker_ids,
                "final_names": [
                    {"speaker_id": speaker_id, "name": speaker_names.get(speaker_id, speaker_id)}
                    for speaker_id in detected_speaker_ids
                ],
            },
            "execution": self._build_execution_summary(runtime, transcription_runtime_payload, formatter),
            "timings": {
                "total_s": round(max(0.0, time.monotonic() - run_started_monotonic), 3),
                "job_started_at": run_started_at.isoformat(timespec="seconds"),
                "job_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stages": stage_timings,
            },
            "artifacts": state.artifact_paths,
        }

        summary_path = job_dir / "job_summary.json"
        write_json(summary_path, summary)
        JOB_STORE.set_artifact(job_id, "job_summary", summary_path)

    @staticmethod
    def _build_execution_summary(runtime, transcription_runtime_payload: dict[str, object] | None, formatter: Formatter | None) -> dict[str, dict[str, object]]:
        api_config = runtime.api_config
        diarization_api = api_config is not None and api_config.diarization_execution == "api"
        transcription_api = api_config is not None and api_config.transcription_execution == "api"
        formatter_api = api_config is not None and api_config.formatter_execution == "api"

        formatter_backend = "openai" if formatter_api else ("command" if SETTINGS.formatter_backend == "command" else "ollama")
        if formatter_api:
            formatter_model = api_config.formatter_model
            formatter_compute_active = "cloud"
        elif formatter_backend == "command":
            formatter_model = SETTINGS.formatter_model_path
            formatter_compute_active = "unknown"
        else:
            formatter_model = SETTINGS.formatter_ollama_model
            formatter_compute_active = "unknown"

        return {
            "diarization": {
                "mode": "api" if diarization_api else "local",
                "model": api_config.diarization_model if diarization_api else SETTINGS.diarization_model_path,
                "backend": "openai" if diarization_api else SETTINGS.diarization_backend,
                "compute_requested": "cloud" if diarization_api else SETTINGS.compute_device,
                "compute_active": "cloud" if diarization_api else SETTINGS.compute_device,
            },
            "transcription": {
                "mode": "api" if transcription_api else "local",
                "model": (
                    api_config.transcription_model
                    if transcription_api
                    else transcription_runtime_payload.get("model_path", SETTINGS.voxtral_model_path)
                    if transcription_runtime_payload is not None
                    else SETTINGS.voxtral_model_path
                ),
                "backend": "openai" if transcription_api else "voxtral",
                "binary_path": (
                    None
                    if transcription_api
                    else transcription_runtime_payload.get("binary_path")
                    if transcription_runtime_payload is not None
                    else SETTINGS.voxtral_binary
                ),
                "compute_requested": (
                    "cloud"
                    if transcription_api
                    else transcription_runtime_payload.get("requested_mode", SETTINGS.compute_device)
                    if transcription_runtime_payload is not None
                    else SETTINGS.compute_device
                ),
                "compute_active": (
                    "cloud"
                    if transcription_api
                    else transcription_runtime_payload.get("active_mode", "unknown")
                    if transcription_runtime_payload is not None
                    else "unknown"
                ),
            },
            "formatter": {
                "mode": "api" if formatter_api else "local",
                "model": formatter_model,
                "backend": formatter_backend,
                "formatter_mode": formatter.last_mode if formatter is not None else None,
                "compute_requested": "cloud" if formatter_api else SETTINGS.compute_device,
                "compute_active": formatter_compute_active,
            },
        }

    def _diarize_with_openai(
        self,
        runtime,
        normalized_audio_path: Path,
    ) -> OpenAIDiarizationResult:
        api_config = runtime.api_config
        if api_config is None:
            raise RuntimeError("OpenAI diarization requested without API configuration.")
        client = OpenAIClient(api_config.api_key)
        audio_path = self._pick_openai_audio_source(runtime.audio_path, normalized_audio_path)
        try:
            return client.diarize_audio(audio_path, model=api_config.diarization_model)
        except OpenAIAPIError as exc:
            raise RuntimeError(f"OpenAI diarization failed: {exc}") from exc

    @staticmethod
    def _pick_openai_audio_source(original_audio_path: Path, normalized_audio_path: Path) -> Path:
        supported_suffixes = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
        original_suffix = original_audio_path.suffix.lower()
        if (
            original_audio_path.exists()
            and original_suffix in supported_suffixes
            and original_audio_path.stat().st_size <= 25 * 1024 * 1024
        ):
            return original_audio_path
        if normalized_audio_path.exists() and normalized_audio_path.stat().st_size <= 25 * 1024 * 1024:
            return normalized_audio_path
        raise RuntimeError(
            "OpenAI audio upload requires a supported source file under 25 MB. "
            "Use a shorter/compressed input or keep this step on local execution."
        )

    @staticmethod
    def _diarization_result_from_openai(result: OpenAIDiarizationResult) -> DiarizationResult:
        segments = [
            DiarizationSegment(
                speaker_id=item.speaker_id,
                start_s=item.start_s,
                end_s=item.end_s,
                confidence=1.0,
            )
            for item in result.segments
        ]
        return DiarizationResult(
            segments=segments,
            speaker_count=len({item.speaker_id for item in segments}),
            mode="openai",
            details="openai_api",
        )

    @staticmethod
    def _transcript_segments_from_openai_diarization(
        result: OpenAIDiarizationResult,
        speaker_map: dict[str, str],
    ) -> list[dict[str, object]]:
        transcript_segments = [
            {
                "speaker_id": item.speaker_id,
                "speaker_name": speaker_map.get(item.speaker_id, item.speaker_id),
                "start_s": item.start_s,
                "end_s": item.end_s,
                "text": item.text,
            }
            for item in result.segments
            if item.text.strip()
        ]
        return sorted(transcript_segments, key=lambda item: (float(item["start_s"]), float(item["end_s"])))

    def _build_speaker_info(
        self,
        job_id: str,
        normalized_audio_path: Path,
        segments_by_speaker: dict[str, list[tuple[float, float]]],
        snippets,
    ) -> JobSpeakerInfo:
        snippet_by_speaker: dict[str, list[SpeakerSnippet]] = {}
        for snippet in snippets:
            snippet_by_speaker.setdefault(snippet.speaker_id, []).append(
                SpeakerSnippet(
                    speaker_id=snippet.speaker_id,
                    snippet_path=str(snippet.path),
                    start_s=snippet.start_s,
                    end_s=snippet.end_s,
                )
            )

        audio_data, sample_rate = VOICE_PROFILE_MANAGER.load_mono_audio(normalized_audio_path)
        profiles = VOICE_PROFILE_MANAGER.list_profiles()
        speakers: list[SpeakerState] = []
        for speaker_id in sorted(segments_by_speaker):
            speaker_segments = segments_by_speaker[speaker_id][:8]
            embedding = VOICE_PROFILE_MANAGER.compute_embedding_from_segments(
                normalized_audio_path,
                speaker_segments,
                audio_data=audio_data,
                sample_rate=sample_rate,
            )
            match = VOICE_PROFILE_MANAGER.match(embedding, profiles=profiles)
            suggested_name = None
            if match.best_match and not match.ambiguous_matches:
                suggested_name = match.best_match.name
                JOB_STORE.append_log(job_id, f"Auto-identified {speaker_id} as {suggested_name}")

            speakers.append(
                SpeakerState(
                    speaker_id=speaker_id,
                    suggested_name=suggested_name,
                    snippets=snippet_by_speaker.get(speaker_id, []),
                )
            )

        return JobSpeakerInfo(detected_speakers=len(speakers), speakers=speakers)

    @staticmethod
    def _group_segments_by_speaker(segments: list[DiarizationSegment]) -> dict[str, list[tuple[float, float]]]:
        grouped: dict[str, list[tuple[float, float]]] = {}
        for segment in segments:
            grouped.setdefault(segment.speaker_id, []).append((segment.start_s, segment.end_s))
        return grouped

    @staticmethod
    def _collapse_labeled_segments(
        segments: list[DiarizationSegment],
        speaker_map: dict[str, str],
        max_gap_s: float = 0.6,
    ) -> list[dict[str, object]]:
        if not segments:
            return []

        canonical_ids_by_name: dict[str, str] = {}
        normalized: list[dict[str, object]] = []
        for segment in sorted(segments, key=lambda item: (item.start_s, item.end_s)):
            mapped_name = (speaker_map.get(segment.speaker_id, segment.speaker_id) or segment.speaker_id).strip()
            if not mapped_name:
                mapped_name = segment.speaker_id
            canonical_id = canonical_ids_by_name.setdefault(mapped_name, segment.speaker_id)
            normalized.append(
                {
                    "speaker_id": canonical_id,
                    "speaker_name": mapped_name,
                    "start_s": float(segment.start_s),
                    "end_s": float(segment.end_s),
                }
            )

        collapsed: list[dict[str, object]] = [normalized[0]]
        for segment in normalized[1:]:
            current = collapsed[-1]
            if (
                str(segment["speaker_id"]) == str(current["speaker_id"])
                and float(segment["start_s"]) - float(current["end_s"]) <= max_gap_s
            ):
                current["end_s"] = float(segment["end_s"])
            else:
                collapsed.append(segment.copy())
        return collapsed

    @staticmethod
    def _plan_transcription_chunks(
        segments: list[dict[str, object]],
        max_gap_s: float,
        max_chunk_s: float,
    ) -> list[dict[str, object]]:
        if not segments:
            return []

        ordered = sorted(segments, key=lambda item: (float(item["start_s"]), float(item["end_s"])))
        chunks: list[dict[str, object]] = [ordered[0].copy()]
        for segment in ordered[1:]:
            current = chunks[-1]
            same_speaker = str(segment["speaker_id"]) == str(current["speaker_id"])
            gap_s = float(segment["start_s"]) - float(current["end_s"])
            proposed_duration = float(segment["end_s"]) - float(current["start_s"])
            if same_speaker and gap_s <= max_gap_s and proposed_duration <= max_chunk_s:
                current["end_s"] = float(segment["end_s"])
            else:
                chunks.append(segment.copy())
        return chunks

    def _set_stage(self, job_id: str, stage_index: int) -> None:
        JOB_STORE.set_stage(job_id, STAGES[stage_index], overall_percent=self._overall(stage_index, 0.0))

    def _overall(self, stage_index: int, stage_percent: float) -> float:
        base = 100 / len(STAGES)
        return min(100.0, (stage_index * base) + (stage_percent / 100.0) * base)

    @staticmethod
    def _ensure_not_cancelled(job_id: str) -> None:
        if JOB_STORE.is_cancelled(job_id):
            raise CancelledError("Job cancelled")


ORCHESTRATOR = PipelineOrchestrator()
