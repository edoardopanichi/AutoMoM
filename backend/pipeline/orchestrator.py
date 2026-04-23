from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import threading
import time
from typing import Any

from backend.app.config import SETTINGS
from backend.app.job_store import JOB_STORE, OpenAIJobConfig, ensure_job_artifact_dir, write_json
from backend.app.schemas import (
    JobSpeakerInfo,
    SpeakerMappingItem,
    SpeakerProfileMatch,
    SpeakerSnippet,
    SpeakerSnippetAction,
    SpeakerState,
)
from backend.models.diarization_registry import resolve_local_diarization_model
from backend.models.local_catalog import LOCAL_MODEL_CATALOG
from backend.pipeline.audio import extract_segment, normalize_audio, validate_audio_input
from backend.pipeline.diarization import (
    CHUNKED_LOCAL_DIARIZATION_THRESHOLD_S,
    DiarizationResult,
    DiarizationSegment,
    diarize,
)
from backend.pipeline.formatter import Formatter
from backend.pipeline.openai_client import (
    OPENAI_MAX_FILE_BYTES,
    OpenAIAPIError,
    OpenAIClient,
    OpenAIDiarizationResult,
    OpenAIDiarizedSegment,
)
from backend.pipeline.remote_worker_client import RemoteWorkerClient, RemoteWorkerError
from backend.pipeline.snippets import extract_snippets, pick_snippet_ranges
from backend.pipeline.subprocess_utils import SubprocessCancelledError
from backend.pipeline.template_manager import TEMPLATE_MANAGER
from backend.pipeline.transcription import (
    FasterWhisperTranscriber,
    OpenAITranscriber,
    RemoteWhisperCppTranscriber,
    TranscriptionError,
    WhisperCppTranscriber,
    transcribe_segments,
)
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

PROFILE_EVIDENCE_FALLBACK_THRESHOLD = 0.55
PROFILE_EVIDENCE_FALLBACK_MARGIN = 0.18


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
        """! @brief Initialize the PipelineOrchestrator instance.
        """
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=SETTINGS.max_workers)
        self._futures: dict[str, concurrent.futures.Future[None]] = {}

    def submit(self, job_id: str) -> None:
        """! @brief Submit operation.
        @param job_id Identifier of the job being processed.
        """
        future = self._executor.submit(self._run_job, job_id)
        self._futures[job_id] = future

    def cancel(self, job_id: str) -> None:
        """! @brief Cancel operation.
        @param job_id Identifier of the job being processed.
        """
        JOB_STORE.cancel(job_id)

    def _run_job(self, job_id: str) -> None:
        """! @brief Run job.
        @param job_id Identifier of the job being processed.
        """
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
            audio_metadata_payload = normalize_audio(
                runtime.audio_path,
                normalized_audio_path,
                ffmpeg_bin=SETTINGS.ffmpeg_bin,
                job_id=job_id,
            )
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
            diarization_model = None
            diarization_progress_stop: threading.Event | None = None
            diarization_progress_thread: threading.Thread | None = None
            if api_config is not None and api_config.diarization_execution == "api":
                api_diarization_result = self._diarize_with_openai(
                    runtime=runtime,
                    normalized_audio_path=normalized_audio_path,
                    job_id=job_id,
                    job_dir=job_dir,
                    audio_metadata_payload=audio_metadata_payload,
                )
                diarization_result = self._diarization_result_from_openai(api_diarization_result)
            else:
                diarization_model = resolve_local_diarization_model(runtime.local_diarization_model_id)
                min_speakers = SETTINGS.diarization_min_speakers if SETTINGS.diarization_min_speakers > 0 else None
                max_speakers = SETTINGS.diarization_max_speakers if SETTINGS.diarization_max_speakers > 0 else None
                JOB_STORE.set_stage_percent(job_id, 6.0, overall_percent=self._overall(2, 6.0))
                if getattr(diarization_model, "location", "local") == "remote":
                    JOB_STORE.append_log(job_id, f"Calling remote diarization worker at {diarization_model.base_url}")
                    diarization_result = self._diarize_with_remote_worker(
                        diarization_model=diarization_model,
                        normalized_audio_path=normalized_audio_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                else:
                    JOB_STORE.append_log(job_id, "Loading local diarization pipeline")
                    use_chunked_progress = bool(audio_metadata_payload) and float(audio_metadata_payload.get("duration_s", 0.0)) > CHUNKED_LOCAL_DIARIZATION_THRESHOLD_S
                    if not use_chunked_progress:
                        diarization_progress_stop, diarization_progress_thread = self._start_stage_heartbeat(job_id, 2)

                    def _diarization_progress(event: dict[str, object]) -> None:
                        """! @brief Diarization progress.
                        @param event Value for event.
                        """
                        total_s = max(1.0, float(event.get("total_s", 0.0) or 0.0))
                        processed_s = max(0.0, float(event.get("processed_s", 0.0) or 0.0))
                        phase = str(event.get("phase") or "")
                        detail = str(event.get("detail") or "").strip()
                        if phase == "finalizing":
                            stage_percent = 99.0
                        else:
                            stage_percent = min(99.0, (processed_s / total_s) * 99.0)
                        if detail:
                            JOB_STORE.set_stage_percent(
                                job_id,
                                stage_percent,
                                overall_percent=self._overall(2, stage_percent),
                                stage_detail=detail,
                            )

                    try:
                        diarization_chunk_s = (
                            SETTINGS.diarization_pyannote_chunk_s
                            if SETTINGS.diarization_backend in {"auto", "pyannote"}
                            else SETTINGS.diarization_max_chunk_s
                        )
                        diarization_result = diarize(
                            normalized_audio_path,
                            speech_regions,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                            max_chunk_s=diarization_chunk_s,
                            backend=SETTINGS.diarization_backend,
                            model_path=Path(diarization_model.pipeline_path),
                            pipeline_path=diarization_model.pipeline_path,
                            embedding_model=diarization_model.embedding_model_ref,
                            compute_device=SETTINGS.compute_device,
                            cuda_device_id=SETTINGS.cuda_device_id,
                            job_id=job_id,
                            progress_callback=_diarization_progress if use_chunked_progress else None,
                        )
                    finally:
                        if diarization_progress_stop is not None:
                            diarization_progress_stop.set()
                        if diarization_progress_thread is not None:
                            diarization_progress_thread.join(timeout=0.2)
            write_json(job_dir / "diarization.json", diarization_result.to_json())
            JOB_STORE.set_artifact(job_id, "diarization", job_dir / "diarization.json")
            if diarization_result.chunk_plan:
                write_json(job_dir / "diarization_chunks.json", diarization_result.chunk_plan)
                JOB_STORE.set_artifact(job_id, "diarization_chunks", job_dir / "diarization_chunks.json")
            if diarization_result.stitching_debug:
                write_json(job_dir / "diarization_stitching.json", diarization_result.stitching_debug)
                JOB_STORE.set_artifact(job_id, "diarization_stitching", job_dir / "diarization_stitching.json")
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
            snippet_ranges = pick_snippet_ranges(diarization_result.segments, audio_path=normalized_audio_path)
            snippets = extract_snippets(
                source_audio_path=normalized_audio_path,
                output_dir=job_dir / "snippets",
                snippet_ranges=snippet_ranges,
                ffmpeg_bin=SETTINGS.ffmpeg_bin,
                job_id=job_id,
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
            speaker_info = self._build_speaker_info(
                runtime,
                job_id,
                normalized_audio_path,
                segments_by_speaker,
                snippets,
                diarization_result,
            )
            JOB_STORE.set_waiting_for_speaker_input(job_id, speaker_info)
            mapping_items = JOB_STORE.wait_for_mapping(job_id)
            if mapping_items is None:
                raise CancelledError("Job cancelled during speaker naming")
            snippet_actions = list(runtime.speaker_snippet_actions_payload)
            diarization_result.segments = self._apply_snippet_splits(
                diarization_result.segments,
                snippets,
                snippet_actions,
            )
            diarization_result.speaker_count = len({item.speaker_id for item in diarization_result.segments})
            segments_by_speaker = self._group_segments_by_speaker(diarization_result.segments)
            expanded_mapping_items = self._expand_speaker_mappings(mapping_items)
            speaker_map = self._speaker_map_from_mappings(expanded_mapping_items)
            excluded_speaker_ids = {
                item.speaker_id
                for item in expanded_mapping_items
                if item.exclude_from_mom
            }
            write_json(
                job_dir / "speaker_mapping.json",
                {
                    "mappings": [item.model_dump() for item in mapping_items],
                    "expanded_mappings": [item.model_dump() for item in expanded_mapping_items],
                    "snippet_actions": [item.model_dump() for item in snippet_actions],
                },
            )
            JOB_STORE.set_artifact(job_id, "speaker_mapping", job_dir / "speaker_mapping.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(4, 100.0))

            for mapping in mapping_items:
                if not mapping.save_voice_profile or mapping.exclude_from_mom:
                    continue
                if diarization_model is None:
                    JOB_STORE.append_log(
                        job_id,
                        f"Skipped voice profile save for {mapping.name}: local diarization embeddings are unavailable for API diarization",
                    )
                    continue
                segments = self._profile_clip_ranges_for_mapping(
                    mapping,
                    snippets,
                    snippet_actions,
                    segments_by_speaker,
                )
                profile_save_kwargs: dict[str, object] = {}
                if diarization_model.location == "remote":
                    embedding_result = self._remote_embedding_for_ranges(diarization_model, normalized_audio_path, segments)
                    profile_save_kwargs = {
                        "embedding_vector": embedding_result.vector,
                        "library_version": embedding_result.library_version,
                        "engine_kind": embedding_result.engine_kind,
                        "threshold": embedding_result.threshold,
                    }
                profile = VOICE_PROFILE_MANAGER.save_profile_sample(
                    name=mapping.name,
                    source_audio_path=normalized_audio_path,
                    clip_ranges=segments,
                    diarization_model_id=diarization_model.profile_model_ref or diarization_model.model_id,
                    embedding_model_ref=diarization_model.embedding_model_ref,
                    source_job_id=job_id,
                    source_speaker_id=mapping.speaker_id,
                    compute_device=SETTINGS.compute_device,
                    cuda_device_id=SETTINGS.cuda_device_id,
                    **profile_save_kwargs,
                )
                action = "updated" if profile.sample_count > 1 else "saved"
                JOB_STORE.append_log(
                    job_id,
                    f"Voice profile {action} for {mapping.name} (samples={profile.sample_count})",
                )
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
                """! @brief Segment progress.
                @param done Value for done.
                @param total Value for total.
                """
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
                    # Reuse the diarized transcript that OpenAI already returned instead of uploading
                    # per-speaker chunks again for a second cloud transcription step.
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
                    audio_duration_s = self._audio_duration_s(normalized_audio_path, audio_metadata_payload)
                    for idx, segment in enumerate(transcription_chunks, start=1):
                        self._append_openai_transcription_jobs(
                            job_id=job_id,
                            source_audio_path=normalized_audio_path,
                            output_dir=transcription_dir,
                            segment_index=idx,
                            segment=segment,
                            segment_jobs=segment_jobs,
                            total_duration_s=audio_duration_s,
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
                transcription_model = LOCAL_MODEL_CATALOG.resolve_model("transcription", runtime.local_transcription_model_id)
                transcriber = self._build_local_transcriber(job_id, transcription_model)
                if not transcriber.available():
                    raise TranscriptionError(
                        f"ASR runtime unavailable for local model '{transcription_model.name}'. "
                        f"Fix: {transcription_model.validation_error or 'verify the configured runtime details.'}"
                    )
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
                        job_id=job_id,
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
                initial_runtime = transcriber.runtime_report()
                JOB_STORE.append_log(
                    job_id,
                    (
                        f"ASR runtime detected; local model={transcription_model.name} "
                        f"runtime={transcription_model.runtime} "
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
            formatter_transcript_segments = self._transcript_for_formatter(
                transcript_segments,
                excluded_speaker_ids,
            )
            named_speakers = self._named_speakers_for_formatter(formatter_transcript_segments)
            transcript_payload = {
                "speakers": named_speakers,
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
            local_formatter_model = None if use_api_formatter else LOCAL_MODEL_CATALOG.resolve_model(
                "formatter",
                runtime.local_formatter_model_id,
            )
            use_legacy_formatter_command = bool(local_formatter_model is not None and local_formatter_model.runtime == "command")
            JOB_STORE.append_log(
                job_id,
                (
                    f"Formatter backend: OpenAI ({api_config.formatter_model})"
                    if use_api_formatter
                    else f"Formatter backend: {local_formatter_model.runtime} ({local_formatter_model.name})"
                ),
            )
            formatter = Formatter(
                command_template=local_formatter_model.config.get("command_template", "") if use_legacy_formatter_command else "",
                model_path=local_formatter_model.config.get("model_path", "") if use_legacy_formatter_command else "",
                ollama_host=local_formatter_model.config.get("base_url", SETTINGS.ollama_host),
                ollama_model="" if use_api_formatter or use_legacy_formatter_command else local_formatter_model.config.get("tag", ""),
                openai_api_key=api_config.api_key if use_api_formatter else "",
                openai_model=api_config.formatter_model if use_api_formatter else "",
                job_id=job_id,
                timeout_s=int(local_formatter_model.config.get("timeout_s", SETTINGS.formatter_timeout_s)) if not use_api_formatter else SETTINGS.formatter_timeout_s,
            )
            title = runtime.title or runtime.audio_path.stem
            speakers = transcript_payload["speakers"]
            mom_path = job_dir / "mom.md"
            formatter_result = formatter.write_model_output_to_mom(
                transcript=formatter_transcript_segments,
                speakers=speakers,
                title=title,
                template_id=runtime.template_id,
                output_path=mom_path,
            )
            JOB_STORE.append_log(job_id, f"Formatter mode: {formatter.last_mode}")
            if formatter_result.validation.get("reduction_used"):
                JOB_STORE.append_log(
                    job_id,
                    f"Formatter long-input mode enabled (estimated_tokens={formatter_result.validation.get('estimated_tokens')})",
                )
            if not formatter_result.validation.get("valid", True):
                JOB_STORE.append_log(
                    job_id,
                    "Formatter validation failed: " + "; ".join(formatter_result.validation.get("errors", [])),
                )
            if formatter.last_stdout:
                (job_dir / "formatter_stdout.txt").write_text(formatter.last_stdout, encoding="utf-8")
                JOB_STORE.set_artifact(job_id, "formatter_stdout", job_dir / "formatter_stdout.txt")
            if formatter.last_stderr:
                (job_dir / "formatter_stderr.txt").write_text(formatter.last_stderr, encoding="utf-8")
                JOB_STORE.set_artifact(job_id, "formatter_stderr", job_dir / "formatter_stderr.txt")
            if formatter.last_raw_output:
                (job_dir / "formatter_raw_output.txt").write_text(formatter.last_raw_output, encoding="utf-8")
                JOB_STORE.set_artifact(job_id, "formatter_raw_output", job_dir / "formatter_raw_output.txt")
            write_json(job_dir / "mom_structured.json", formatter_result.structured)
            (job_dir / "formatter_system_prompt.txt").write_text(formatter_result.system_prompt, encoding="utf-8")
            (job_dir / "formatter_user_prompt.txt").write_text(formatter_result.user_prompt, encoding="utf-8")
            (job_dir / "full_meeting_transcript.md").write_text(
                self._render_full_meeting_transcript(
                    title=title,
                    speakers=speakers,
                    transcript_segments=formatter_transcript_segments,
                ),
                encoding="utf-8",
            )
            write_json(job_dir / "formatter_validation.json", formatter_result.validation)
            JOB_STORE.set_artifact(job_id, "formatter_system_prompt", job_dir / "formatter_system_prompt.txt")
            JOB_STORE.set_artifact(job_id, "formatter_user_prompt", job_dir / "formatter_user_prompt.txt")
            JOB_STORE.set_artifact(job_id, "full_meeting_transcript", job_dir / "full_meeting_transcript.md")
            JOB_STORE.set_artifact(job_id, "formatter_validation", job_dir / "formatter_validation.json")
            if formatter_result.reduced_notes:
                write_json(job_dir / "formatter_reduced_notes.json", formatter_result.reduced_notes)
                JOB_STORE.set_artifact(job_id, "formatter_reduced_notes", job_dir / "formatter_reduced_notes.json")
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

            self._write_job_summary(
                job_id=job_id,
                runtime=runtime,
                job_dir=job_dir,
                run_started_at=run_started_at,
                run_started_monotonic=run_started_monotonic,
                stage_timings=stage_timings,
                diarization_result=diarization_result,
                mapping_items=expanded_mapping_items,
                transcript_segments=transcript_segments,
                audio_metadata_payload=audio_metadata_payload,
                transcription_runtime_payload=transcription_runtime_payload,
                formatter=formatter,
            )
            JOB_STORE.mark_completed(job_id)
            JOB_STORE.append_log(job_id, "Job completed successfully")
        except (CancelledError, SubprocessCancelledError) as exc:
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

    def _begin_stage_timing(self, stage_index: int) -> tuple[str, str, datetime, float]:
        """! @brief Begin stage timing.
        @param stage_index Value for stage index.
        @return Tuple produced by the operation.
        """
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
        """! @brief Finish stage timing.
        @param stage_timings Value for stage timings.
        @param stage_key Value for stage key.
        @param stage_name Value for stage name.
        @param started_at Value for started at.
        @param started_monotonic Value for started monotonic.
        """
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
        """! @brief Write job summary.
        @param job_id Identifier of the job being processed.
        @param runtime Value for runtime.
        @param job_dir Value for job dir.
        @param run_started_at Value for run started at.
        @param run_started_monotonic Value for run started monotonic.
        @param stage_timings Value for stage timings.
        @param diarization_result Value for diarization result.
        @param mapping_items Value for mapping items.
        @param transcript_segments Value for transcript segments.
        @param audio_metadata_payload Value for audio metadata payload.
        @param transcription_runtime_payload Value for transcription runtime payload.
        @param formatter Value for formatter.
        """
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
            "status": "completed" if state.status == "running" and state.error is None else state.status,
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
        """! @brief Build execution summary.
        @param runtime Value for runtime.
        @param transcription_runtime_payload Value for transcription runtime payload.
        @param formatter Value for formatter.
        @return Dictionary produced by the operation.
        """
        api_config = runtime.api_config
        diarization_api = api_config is not None and api_config.diarization_execution == "api"
        transcription_api = api_config is not None and api_config.transcription_execution == "api"
        formatter_api = api_config is not None and api_config.formatter_execution == "api"

        diarization_model = None if diarization_api else resolve_local_diarization_model(runtime.local_diarization_model_id)
        local_transcription_model = None if transcription_api else LOCAL_MODEL_CATALOG.resolve_model(
            "transcription",
            runtime.local_transcription_model_id,
        )
        local_formatter_model = None if formatter_api else LOCAL_MODEL_CATALOG.resolve_model(
            "formatter",
            runtime.local_formatter_model_id,
        )

        formatter_backend = "openai" if formatter_api else local_formatter_model.runtime
        if formatter_api:
            formatter_model = api_config.formatter_model
            formatter_compute_active = "cloud"
        elif formatter_backend == "command":
            formatter_model = local_formatter_model.config.get("model_path", "")
            formatter_compute_active = "unknown"
        else:
            formatter_model = local_formatter_model.config.get("tag", "")
            formatter_compute_active = "unknown"

        diarization_mode = "api" if diarization_api else (getattr(diarization_model, "location", "local") if diarization_model is not None else "local")
        transcription_mode = "api" if transcription_api else (getattr(local_transcription_model, "location", "local") if local_transcription_model is not None else "local")
        formatter_mode = "api" if formatter_api else (getattr(local_formatter_model, "location", "local") if local_formatter_model is not None else "local")
        return {
            "diarization": {
                "mode": "api" if diarization_api else "local",
                "location": "cloud" if diarization_api else "lan-remote" if diarization_mode == "remote" else "same-machine",
                "host": None if diarization_api or diarization_mode != "remote" else getattr(diarization_model, "base_url", ""),
                "model": api_config.diarization_model if diarization_api else getattr(diarization_model, "base_url", "") or diarization_model.pipeline_path,
                "model_id": None if diarization_api else diarization_model.model_id,
                "embedding_model_ref": None if diarization_api else diarization_model.embedding_model_ref,
                "backend": "openai" if diarization_api else diarization_model.runtime,
                "compute_requested": "cloud" if diarization_api else SETTINGS.compute_device,
                "compute_active": (
                    "cloud"
                    if diarization_api
                    else "remote"
                    if diarization_mode == "remote"
                    else SETTINGS.compute_device
                ),
            },
            "transcription": {
                "mode": "api" if transcription_api else "local",
                "location": "cloud" if transcription_api else "lan-remote" if transcription_mode == "remote" else "same-machine",
                "host": None if transcription_api or transcription_mode != "remote" else local_transcription_model.config.get("base_url", ""),
                "model": (
                    api_config.transcription_model
                    if transcription_api
                    else transcription_runtime_payload.get("model_path", SETTINGS.transcription_model_path)
                    if transcription_runtime_payload is not None
                    else local_transcription_model.config.get("model_path", SETTINGS.transcription_model_path)
                ),
                "model_id": None if transcription_api else local_transcription_model.model_id,
                "backend": "openai" if transcription_api else local_transcription_model.runtime,
                "binary_path": (
                    None
                    if transcription_api
                    else transcription_runtime_payload.get("binary_path")
                    if transcription_runtime_payload is not None
                    else local_transcription_model.config.get("binary_path", "")
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
                "location": "cloud" if formatter_api else "lan-remote" if formatter_mode == "remote" else "same-machine",
                "host": None if formatter_api or formatter_mode != "remote" else local_formatter_model.config.get("base_url", ""),
                "model": formatter_model,
                "model_id": None if formatter_api else local_formatter_model.model_id,
                "backend": formatter_backend,
                "formatter_mode": formatter.last_mode if formatter is not None else None,
                "compute_requested": "cloud" if formatter_api else SETTINGS.compute_device,
                "compute_active": formatter_compute_active,
            },
        }

    def _build_local_transcriber(self, job_id: str, model_record):
        """! @brief Build local transcriber.
        @param job_id Identifier of the job being processed.
        @param model_record Value for model record.
        @return Result produced by the operation.
        """
        if model_record.location == "remote" and model_record.runtime == "whisper.cpp":
            return RemoteWhisperCppTranscriber(
                base_url=model_record.config.get("base_url", ""),
                model_name=model_record.config.get("model_name", ""),
                auth_token=model_record.config.get("auth_token", ""),
                timeout_s=int(model_record.config.get("timeout_s", "900") or "900"),
            )
        if model_record.runtime == "whisper.cpp":
            return WhisperCppTranscriber(
                model_record.config.get("binary_path", ""),
                model_record.config.get("model_path", ""),
                job_id=job_id,
                compute_device=SETTINGS.compute_device,
                cuda_device_id=SETTINGS.cuda_device_id,
                gpu_layers=SETTINGS.transcription_gpu_layers,
                threads=SETTINGS.transcription_threads,
                processors=SETTINGS.transcription_processors,
            )
        if model_record.runtime == "faster-whisper":
            return FasterWhisperTranscriber(
                model_record.config.get("model_path", ""),
                compute_device=SETTINGS.compute_device,
                cuda_device_id=SETTINGS.cuda_device_id,
                compute_type=model_record.config.get("compute_type", "auto"),
            )
        raise TranscriptionError(f"Unsupported transcription runtime: {model_record.runtime}")

    @staticmethod
    def _remote_worker_client(model_record) -> RemoteWorkerClient:
        timeout_s = int(str(model_record.config.get("timeout_s", "900") or "900"))
        return RemoteWorkerClient(
            base_url=model_record.config.get("base_url", ""),
            auth_token=model_record.config.get("auth_token", ""),
            timeout_s=timeout_s,
        )

    def _diarize_with_remote_worker(
        self,
        *,
        diarization_model,
        normalized_audio_path: Path,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizationResult:
        model_record = LOCAL_MODEL_CATALOG.resolve_model("diarization", diarization_model.model_id)
        client = self._remote_worker_client(model_record)
        try:
            payload = client.diarize(
                audio_path=normalized_audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except RemoteWorkerError as exc:
            raise RuntimeError(str(exc)) from exc
        segments = [
            DiarizationSegment(
                speaker_id=str(item.get("speaker_id", "")),
                start_s=float(item.get("start_s", 0.0)),
                end_s=float(item.get("end_s", 0.0)),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            )
            for item in payload.get("segments", [])
        ]
        return DiarizationResult(
            segments=segments,
            speaker_count=int(payload.get("speaker_count", len({item.speaker_id for item in segments}))),
            mode=str(payload.get("mode", "remote-pyannote")),
            details=str(payload.get("details", model_record.config.get("base_url", diarization_model.base_url))),
            speaker_embeddings={
                str(key): [float(value) for value in values]
                for key, values in dict(payload.get("speaker_embeddings", {})).items()
            } or None,
            profile_model_ref=str(payload.get("profile_model_ref", diarization_model.profile_model_ref or diarization_model.model_id)),
            embedding_model_ref=str(payload.get("embedding_model_ref", diarization_model.embedding_model_ref)),
            embedding_library_version=str(payload.get("library_version", "")) or None,
            embedding_engine_kind=str(payload.get("engine_kind", "remote_pyannote")),
        )

    def _remote_embedding_for_ranges(self, diarization_model, normalized_audio_path: Path, clip_ranges: list[tuple[float, float]]):
        model_record = LOCAL_MODEL_CATALOG.resolve_model("diarization", diarization_model.model_id)
        client = self._remote_worker_client(model_record)
        return client.embed(audio_path=normalized_audio_path, clip_ranges=clip_ranges)

    @staticmethod
    def _render_full_meeting_transcript(
        *,
        title: str,
        speakers: list[str],
        transcript_segments: list[dict[str, object]],
    ) -> str:
        """! @brief Render the downloadable full meeting transcript artifact."""
        lines = [
            f"# {title}",
            "",
            "## Participants",
            ", ".join(speakers) if speakers else "None",
            "",
            "## Transcript",
        ]
        for segment in transcript_segments:
            speaker = str(segment.get("speaker_name") or segment.get("speaker_id") or "Speaker")
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            start_s = PipelineOrchestrator._safe_seconds(segment.get("start_s"))
            end_s = PipelineOrchestrator._safe_seconds(segment.get("end_s"))
            timestamp = f"[{PipelineOrchestrator._format_timestamp(start_s)}-{PipelineOrchestrator._format_timestamp(end_s)}]"
            lines.append(f"- {timestamp} **{speaker}**: {text}")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _safe_seconds(value: object) -> float:
        """! @brief Convert a timestamp-like value to seconds."""
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """! @brief Format seconds as an HH:MM:SS transcript timestamp."""
        total_seconds = int(round(max(0.0, seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _diarize_with_openai(
        self,
        runtime,
        normalized_audio_path: Path,
        job_id: str,
        job_dir: Path,
        audio_metadata_payload: dict[str, object] | None,
    ) -> OpenAIDiarizationResult:
        """! @brief Diarize with openai.
        @param runtime Value for runtime.
        @param normalized_audio_path Value for normalized audio path.
        @param job_id Identifier of the job being processed.
        @param job_dir Value for job dir.
        @param audio_metadata_payload Value for audio metadata payload.
        @return Result produced by the operation.
        """
        api_config = runtime.api_config
        if api_config is None:
            raise RuntimeError("OpenAI diarization requested without API configuration.")
        client = OpenAIClient(api_config.api_key)
        audio_path = self._pick_openai_audio_source(runtime.audio_path, normalized_audio_path, raise_on_oversize=False)
        if audio_path is None:
            return self._diarize_with_openai_chunks(
                client=client,
                model=api_config.diarization_model,
                normalized_audio_path=normalized_audio_path,
                job_id=job_id,
                job_dir=job_dir,
                audio_metadata_payload=audio_metadata_payload,
            )
        try:
            return client.diarize_audio(audio_path, model=api_config.diarization_model)
        except OpenAIAPIError as exc:
            raise RuntimeError(f"OpenAI diarization failed: {exc}") from exc

    @staticmethod
    def _pick_openai_audio_source(
        original_audio_path: Path,
        normalized_audio_path: Path,
        *,
        raise_on_oversize: bool = True,
    ) -> Path | None:
        """! @brief Pick openai audio source.
        @param original_audio_path Value for original audio path.
        @param normalized_audio_path Value for normalized audio path.
        @param raise_on_oversize Whether to raise when no direct upload candidate is available.
        @return Path result produced by the operation.
        """
        supported_suffixes = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
        original_suffix = original_audio_path.suffix.lower()
        if (
            original_audio_path.exists()
            and original_suffix in supported_suffixes
            and original_audio_path.stat().st_size <= OPENAI_MAX_FILE_BYTES
        ):
            return original_audio_path
        if normalized_audio_path.exists() and normalized_audio_path.stat().st_size <= OPENAI_MAX_FILE_BYTES:
            return normalized_audio_path
        if not raise_on_oversize:
            return None
        raise RuntimeError(
            "OpenAI audio upload requires a supported source file under 25 MB. "
            "Use a shorter/compressed input or keep this step on local execution."
        )

    def _diarize_with_openai_chunks(
        self,
        *,
        client: OpenAIClient,
        model: str,
        normalized_audio_path: Path,
        job_id: str,
        job_dir: Path,
        audio_metadata_payload: dict[str, object] | None,
    ) -> OpenAIDiarizationResult:
        duration_s = self._audio_duration_s(normalized_audio_path, audio_metadata_payload)
        chunk_plan = self._plan_openai_audio_chunks(
            normalized_audio_path,
            total_duration_s=duration_s,
            span_start_s=0.0,
            span_end_s=duration_s,
            overlap_s=2.0,
        )
        if not chunk_plan:
            raise RuntimeError("OpenAI audio chunking could not build an upload plan.")

        chunks_dir = ensure_job_artifact_dir(job_id, "openai_audio_chunks")
        JOB_STORE.append_log(job_id, f"OpenAI diarization chunking enabled ({len(chunk_plan)} chunks)")
        write_json(job_dir / "openai_audio_chunks.json", chunk_plan)
        JOB_STORE.set_artifact(job_id, "openai_audio_chunks", job_dir / "openai_audio_chunks.json")

        merged_segments: list[OpenAIDiarizedSegment] = []
        previous_mapping: dict[str, str] = {}
        next_speaker_index = 0
        merged_text_parts: list[str] = []
        for index, chunk in enumerate(chunk_plan, start=1):
            JOB_STORE.append_log(job_id, f"OpenAI diarization chunk {index}/{len(chunk_plan)}")
            chunk_path = chunks_dir / f"diarization_chunk_{index:04d}.wav"
            extract_segment(
                normalized_audio_path,
                chunk_path,
                float(chunk["audio_start_s"]),
                float(chunk["audio_end_s"]),
                ffmpeg_bin=SETTINGS.ffmpeg_bin,
                job_id=job_id,
            )
            if chunk_path.stat().st_size > OPENAI_MAX_FILE_BYTES:
                raise RuntimeError(
                    f"OpenAI diarization chunk '{chunk_path.name}' exceeds the 25 MB upload limit after splitting."
                )
            try:
                result = client.diarize_audio(chunk_path, model=model)
            except OpenAIAPIError as exc:
                raise RuntimeError(f"OpenAI diarization failed for chunk {index}/{len(chunk_plan)}: {exc}") from exc
            if result.text:
                merged_text_parts.append(result.text)
            owned, previous_mapping, next_speaker_index = self._globalize_openai_chunk_segments(
                result.segments,
                chunk,
                existing_segments=merged_segments,
                previous_mapping=previous_mapping,
                next_speaker_index=next_speaker_index,
            )
            merged_segments.extend(owned)
            stage_percent = min(99.0, (index / max(1, len(chunk_plan))) * 99.0)
            JOB_STORE.set_stage_percent(
                job_id,
                stage_percent,
                overall_percent=self._overall(2, stage_percent),
                stage_detail=f"OpenAI chunk {index}/{len(chunk_plan)}",
            )

        ordered = sorted(merged_segments, key=lambda item: (item.start_s, item.end_s))
        return OpenAIDiarizationResult(text="\n".join(merged_text_parts).strip(), segments=ordered)

    @staticmethod
    def _globalize_openai_chunk_segments(
        segments: list[OpenAIDiarizedSegment],
        chunk: dict[str, object],
        *,
        existing_segments: list[OpenAIDiarizedSegment],
        previous_mapping: dict[str, str],
        next_speaker_index: int,
    ) -> tuple[list[OpenAIDiarizedSegment], dict[str, str], int]:
        audio_start_s = float(chunk["audio_start_s"])
        own_start_s = float(chunk["own_start_s"])
        own_end_s = float(chunk["own_end_s"])
        mapping = dict(previous_mapping)
        globalized_raw: list[OpenAIDiarizedSegment] = []

        for item in sorted(segments, key=lambda segment: (segment.start_s, segment.end_s)):
            global_start = audio_start_s + float(item.start_s)
            global_end = audio_start_s + float(item.end_s)
            speaker_id = mapping.get(item.speaker_id)
            if speaker_id is None:
                speaker_id = PipelineOrchestrator._match_openai_speaker_by_overlap(
                    global_start,
                    global_end,
                    existing_segments,
                )
            if speaker_id is None:
                speaker_id = f"SPEAKER_{next_speaker_index}"
                next_speaker_index += 1
            mapping[item.speaker_id] = speaker_id
            globalized_raw.append(
                OpenAIDiarizedSegment(
                    speaker_id=speaker_id,
                    start_s=global_start,
                    end_s=global_end,
                    text=item.text,
                )
            )

        owned: list[OpenAIDiarizedSegment] = []
        for item in globalized_raw:
            midpoint = (item.start_s + item.end_s) / 2.0
            if midpoint < own_start_s or midpoint > own_end_s:
                continue
            clipped_start = max(item.start_s, own_start_s)
            clipped_end = min(item.end_s, own_end_s)
            if clipped_end <= clipped_start:
                continue
            owned.append(
                OpenAIDiarizedSegment(
                    speaker_id=item.speaker_id,
                    start_s=clipped_start,
                    end_s=clipped_end,
                    text=item.text,
                )
            )
        return owned, mapping, next_speaker_index

    @staticmethod
    def _match_openai_speaker_by_overlap(
        start_s: float,
        end_s: float,
        existing_segments: list[OpenAIDiarizedSegment],
    ) -> str | None:
        overlap_by_speaker: dict[str, float] = {}
        for item in existing_segments:
            overlap = max(0.0, min(end_s, item.end_s) - max(start_s, item.start_s))
            if overlap > 0.0:
                overlap_by_speaker[item.speaker_id] = overlap_by_speaker.get(item.speaker_id, 0.0) + overlap
        if not overlap_by_speaker:
            return None
        speaker_id, overlap_s = max(overlap_by_speaker.items(), key=lambda item: item[1])
        return speaker_id if overlap_s >= 0.1 else None

    @staticmethod
    def _audio_duration_s(audio_path: Path, audio_metadata_payload: dict[str, object] | None = None) -> float:
        if audio_metadata_payload is not None:
            try:
                duration = float(audio_metadata_payload.get("duration_s") or 0.0)
                if duration > 0.0:
                    return duration
            except (TypeError, ValueError):
                pass
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            return float(info.duration)
        except Exception:
            return max(1.0, audio_path.stat().st_size / 32000.0)

    @staticmethod
    def _plan_openai_audio_chunks(
        audio_path: Path,
        *,
        total_duration_s: float,
        span_start_s: float,
        span_end_s: float,
        overlap_s: float,
    ) -> list[dict[str, object]]:
        span_start_s = max(0.0, span_start_s)
        span_end_s = min(total_duration_s, max(span_start_s, span_end_s))
        span_duration_s = span_end_s - span_start_s
        if span_duration_s <= 0.0:
            return []
        safe_bytes = OPENAI_MAX_FILE_BYTES - (1024 * 1024)
        bytes_per_second = max(1.0, audio_path.stat().st_size / max(1.0, total_duration_s))
        max_chunk_s = max(5.0, math.floor((safe_bytes / bytes_per_second) * 0.95))
        if span_duration_s <= max_chunk_s:
            return [
                {
                    "chunk_index": 1,
                    "own_start_s": span_start_s,
                    "own_end_s": span_end_s,
                    "audio_start_s": span_start_s,
                    "audio_end_s": span_end_s,
                }
            ]

        chunk_count = max(1, int(math.ceil(span_duration_s / max_chunk_s)))
        own_chunk_s = span_duration_s / chunk_count
        chunks: list[dict[str, object]] = []
        for index in range(chunk_count):
            own_start = span_start_s + (own_chunk_s * index)
            own_end = span_end_s if index == chunk_count - 1 else span_start_s + (own_chunk_s * (index + 1))
            chunks.append(
                {
                    "chunk_index": index + 1,
                    "own_start_s": own_start,
                    "own_end_s": own_end,
                    "audio_start_s": max(span_start_s, own_start - overlap_s),
                    "audio_end_s": min(span_end_s, own_end + overlap_s),
                }
            )
        return chunks

    def _append_openai_transcription_jobs(
        self,
        *,
        job_id: str,
        source_audio_path: Path,
        output_dir: Path,
        segment_index: int,
        segment: dict[str, object],
        segment_jobs: list[dict[str, object]],
        total_duration_s: float,
    ) -> None:
        padding = 0.2
        padded_start_s = max(0.0, float(segment["start_s"]) - padding)
        padded_end_s = min(total_duration_s, float(segment["end_s"]) + padding)
        plan = self._plan_openai_audio_chunks(
            source_audio_path,
            total_duration_s=total_duration_s,
            span_start_s=padded_start_s,
            span_end_s=padded_end_s,
            overlap_s=0.0,
        )
        if len(plan) > 1:
            JOB_STORE.append_log(
                job_id,
                f"OpenAI transcription segment {segment_index} split into {len(plan)} upload chunks",
            )
        for chunk_index, chunk in enumerate(plan, start=1):
            suffix = f"_{chunk_index:02d}" if len(plan) > 1 else ""
            segment_path = output_dir / f"segment_{segment_index:04d}{suffix}.wav"
            extract_segment(
                source_audio_path,
                segment_path,
                float(chunk["audio_start_s"]),
                float(chunk["audio_end_s"]),
                ffmpeg_bin=SETTINGS.ffmpeg_bin,
                job_id=job_id,
            )
            if segment_path.stat().st_size > OPENAI_MAX_FILE_BYTES:
                raise TranscriptionError(
                    f"OpenAI transcription chunk '{segment_path.name}' exceeds the 25 MB upload limit after splitting."
                )
            segment_jobs.append(
                {
                    "segment_path": str(segment_path),
                    "speaker_id": str(segment["speaker_id"]),
                    "speaker_name": str(segment["speaker_name"]),
                    "start_s": max(float(segment["start_s"]), float(chunk["own_start_s"])),
                    "end_s": min(float(segment["end_s"]), float(chunk["own_end_s"])),
                }
            )

    @staticmethod
    def _diarization_result_from_openai(result: OpenAIDiarizationResult) -> DiarizationResult:
        """! @brief Diarization result from openai.
        @param result Value for result.
        @return Result produced by the operation.
        """
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
        """! @brief Transcript segments from openai diarization.
        @param result Value for result.
        @param speaker_map Mapping from speaker ids to final speaker names.
        @return List produced by the operation.
        """
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
        runtime,
        job_id: str,
        normalized_audio_path: Path,
        segments_by_speaker: dict[str, list[tuple[float, float]]],
        snippets,
        diarization_result: DiarizationResult | None = None,
    ) -> JobSpeakerInfo:
        """! @brief Build speaker info.
        @param runtime Value for runtime.
        @param job_id Identifier of the job being processed.
        @param normalized_audio_path Value for normalized audio path.
        @param segments_by_speaker Value for segments by speaker.
        @param snippets Value for snippets.
        @return Result produced by the operation.
        """
        snippet_by_speaker = self._group_snippets_by_speaker(snippets)

        if runtime.api_config is not None and runtime.api_config.diarization_execution == "api":
            return JobSpeakerInfo(
                detected_speakers=len(segments_by_speaker),
                speakers=[
                    SpeakerState(
                        speaker_id=speaker_id,
                        speaker_ids=[speaker_id],
                        review_group_id=speaker_id,
                        snippets=snippet_by_speaker.get(speaker_id, []),
                    )
                    for speaker_id in sorted(segments_by_speaker)
                ],
            )

        profiles = VOICE_PROFILE_MANAGER.list_profiles()
        if not profiles:
            return JobSpeakerInfo(
                detected_speakers=len(segments_by_speaker),
                speakers=[
                    SpeakerState(
                        speaker_id=speaker_id,
                        speaker_ids=[speaker_id],
                        review_group_id=speaker_id,
                        snippets=snippet_by_speaker.get(speaker_id, []),
                    )
                    for speaker_id in sorted(segments_by_speaker)
                ],
            )
        diarization_model = resolve_local_diarization_model(runtime.local_diarization_model_id)
        speakers: list[SpeakerState] = []
        for speaker_id in sorted(segments_by_speaker):
            precomputed_embedding = None
            if diarization_result is not None and diarization_result.speaker_embeddings is not None:
                raw_embedding = diarization_result.speaker_embeddings.get(speaker_id)
                if raw_embedding:
                    precomputed_embedding = raw_embedding
            suggested_name, matched_profile = self._match_speaker_from_evidence(
                job_id=job_id,
                speaker_id=speaker_id,
                normalized_audio_path=normalized_audio_path,
                speaker_segments=segments_by_speaker[speaker_id],
                speaker_snippets=snippet_by_speaker.get(speaker_id, []),
                diarization_model_id=getattr(diarization_model, "profile_model_ref", "") or diarization_model.model_id,
                embedding_model_ref=diarization_model.embedding_model_ref,
                profiles=profiles,
                precomputed_embedding=precomputed_embedding,
            )

            speakers.append(
                SpeakerState(
                    speaker_id=speaker_id,
                    speaker_ids=[speaker_id],
                    review_group_id=speaker_id,
                    suggested_name=suggested_name,
                    matched_profile=matched_profile,
                    snippets=snippet_by_speaker.get(speaker_id, []),
                )
            )

        return JobSpeakerInfo(
            detected_speakers=len(segments_by_speaker),
            speakers=self._group_duplicate_profile_matches(job_id, speakers),
        )

    @staticmethod
    def _expand_speaker_mappings(mapping_items: list[SpeakerMappingItem]) -> list[SpeakerMappingItem]:
        """! @brief Expand grouped speaker mapping rows into one row per underlying speaker id.
        @param mapping_items Submitted mapping rows.
        @return Expanded mappings.
        """
        expanded: list[SpeakerMappingItem] = []
        seen: set[str] = set()
        for item in mapping_items:
            speaker_ids = item.speaker_ids or [item.speaker_id]
            for speaker_id in speaker_ids:
                if speaker_id in seen:
                    continue
                seen.add(speaker_id)
                expanded.append(
                    SpeakerMappingItem(
                        speaker_id=speaker_id,
                        name=item.name,
                        save_voice_profile=item.save_voice_profile,
                        speaker_ids=[speaker_id],
                        assigned_snippet_ids=item.assigned_snippet_ids,
                        exclude_from_mom=item.exclude_from_mom,
                    )
                )
        return expanded

    @staticmethod
    def _speaker_map_from_mappings(mapping_items: list[SpeakerMappingItem]) -> dict[str, str]:
        """! @brief Build final speaker id to speaker name map.
        @param mapping_items Expanded mapping rows.
        @return Mapping from speaker ids to final labels.
        """
        speaker_map: dict[str, str] = {}
        for item in mapping_items:
            name = item.name.strip() or item.speaker_id
            if item.exclude_from_mom and (not item.name.strip() or item.name.strip() == item.speaker_id):
                name = "Unattributed speaker"
            speaker_map[item.speaker_id] = name
        return speaker_map

    @staticmethod
    def _apply_snippet_splits(
        segments: list[DiarizationSegment],
        snippets,
        snippet_actions: list[SpeakerSnippetAction],
    ) -> list[DiarizationSegment]:
        """! @brief Apply exact snippet-range speaker splits requested during review.
        @param segments Diarization segments before review.
        @param snippets Extracted snippets.
        @param snippet_actions Submitted snippet actions.
        @return Updated diarization segments.
        """
        snippet_ranges = {Path(str(snippet.path)).stem: snippet for snippet in snippets}
        split_actions = [
            action
            for action in snippet_actions
            if action.action == "split" and action.target_speaker_id and action.snippet_id in snippet_ranges
        ]
        if not split_actions:
            return segments

        updated = [
            DiarizationSegment(item.speaker_id, item.start_s, item.end_s, item.confidence)
            for item in segments
        ]
        for action in split_actions:
            snippet = snippet_ranges[action.snippet_id]
            start_s = float(snippet.start_s)
            end_s = float(snippet.end_s)
            next_segments: list[DiarizationSegment] = []
            for segment in updated:
                if segment.speaker_id != action.source_speaker_id or segment.end_s <= start_s or segment.start_s >= end_s:
                    next_segments.append(segment)
                    continue
                if segment.start_s < start_s:
                    next_segments.append(
                        DiarizationSegment(segment.speaker_id, segment.start_s, start_s, segment.confidence)
                    )
                overlap_start = max(segment.start_s, start_s)
                overlap_end = min(segment.end_s, end_s)
                if overlap_end > overlap_start:
                    next_segments.append(
                        DiarizationSegment(action.target_speaker_id or segment.speaker_id, overlap_start, overlap_end, segment.confidence)
                    )
                if segment.end_s > end_s:
                    next_segments.append(
                        DiarizationSegment(segment.speaker_id, end_s, segment.end_s, segment.confidence)
                    )
            updated = next_segments
        return sorted(updated, key=lambda item: (item.start_s, item.end_s, item.speaker_id))

    @staticmethod
    def _profile_clip_ranges_for_mapping(
        mapping: SpeakerMappingItem,
        snippets,
        snippet_actions: list[SpeakerSnippetAction],
        segments_by_speaker: dict[str, list[tuple[float, float]]],
    ) -> list[tuple[float, float]]:
        """! @brief Select reviewed clip ranges for saving a voice profile.
        @return Clip ranges.
        """
        excluded = {
            action.snippet_id
            for action in snippet_actions
            if action.action in {"exclude", "split"}
            and action.source_speaker_id in (mapping.speaker_ids or [mapping.speaker_id])
        }
        explicit = set(mapping.assigned_snippet_ids)
        source_speaker_ids = set(mapping.speaker_ids or [mapping.speaker_id])
        ranges: list[tuple[float, float]] = []
        for snippet in snippets:
            snippet_id = Path(str(snippet.path)).stem
            if snippet_id in excluded:
                continue
            if explicit and snippet_id not in explicit:
                continue
            if snippet.speaker_id in source_speaker_ids or snippet_id in explicit:
                ranges.append((float(snippet.start_s), float(snippet.end_s)))
            if len(ranges) >= 2:
                break
        if ranges:
            return ranges
        fallback_ranges: list[tuple[float, float]] = []
        for speaker_id in source_speaker_ids:
            fallback_ranges.extend(segments_by_speaker.get(speaker_id, []))
        return PipelineOrchestrator._select_profile_segments(fallback_ranges, max_segments=2)

    @staticmethod
    def _transcript_for_formatter(
        transcript_segments: list[dict[str, object]],
        excluded_speaker_ids: set[str],
    ) -> list[dict[str, object]]:
        """! @brief Prepare transcript segments for formatter prompts.
        @return Transcript with unnamed speakers de-identified.
        """
        prepared: list[dict[str, object]] = []
        for segment in transcript_segments:
            item = segment.copy()
            if str(item.get("speaker_id") or "") in excluded_speaker_ids:
                item["speaker_name"] = "Unattributed speaker"
                item["exclude_from_mom"] = True
            prepared.append(item)
        return prepared

    @staticmethod
    def _named_speakers_for_formatter(transcript_segments: list[dict[str, object]]) -> list[str]:
        """! @brief Return named speakers eligible for MoM participants.
        @return Sorted speaker list.
        """
        speakers: set[str] = set()
        for segment in transcript_segments:
            if segment.get("exclude_from_mom"):
                continue
            speaker = str(segment.get("speaker_name") or "").strip()
            if not speaker or speaker.startswith("SPEAKER_") or speaker == "Unattributed speaker":
                continue
            speakers.add(speaker)
        return sorted(speakers)

    @staticmethod
    def _group_snippets_by_speaker(snippets) -> dict[str, list[SpeakerSnippet]]:
        """! @brief Group extracted snippets by diarized speaker.
        @param snippets Extracted snippet records.
        @return Snippets keyed by speaker id.
        """
        snippet_by_speaker: dict[str, list[SpeakerSnippet]] = {}
        for snippet in snippets:
            snippet_id = Path(str(snippet.path)).stem
            snippet_by_speaker.setdefault(snippet.speaker_id, []).append(
                SpeakerSnippet(
                    speaker_id=snippet.speaker_id,
                    snippet_path=str(snippet.path),
                    start_s=snippet.start_s,
                    end_s=snippet.end_s,
                    snippet_id=snippet_id,
                )
            )
        return snippet_by_speaker

    def _match_speaker_from_evidence(
        self,
        *,
        job_id: str,
        speaker_id: str,
        normalized_audio_path: Path,
        speaker_segments: list[tuple[float, float]],
        speaker_snippets: list[SpeakerSnippet],
        diarization_model_id: str,
        embedding_model_ref: str,
        profiles,
        precomputed_embedding: list[float] | None = None,
    ) -> tuple[str | None, SpeakerProfileMatch | None]:
        """! @brief Match one diarized speaker against saved profiles using clip-level evidence.
        @return Suggested name and match metadata.
        """
        candidates: list[tuple[float, float, str]] = [
            (snippet.start_s, snippet.end_s, snippet.snippet_id or Path(snippet.snippet_path).stem)
            for snippet in speaker_snippets
        ]
        for start_s, end_s in self._select_profile_segments(speaker_segments):
            if any(abs(start_s - existing[0]) < 0.05 and abs(end_s - existing[1]) < 0.05 for existing in candidates):
                continue
            candidates.append((start_s, end_s, f"{speaker_id}:{start_s:.2f}-{end_s:.2f}"))

        best_clear: tuple[float, object, str] | None = None
        best_ambiguous: tuple[float, object, str] | None = None
        for start_s, end_s, evidence_id in candidates:
            if precomputed_embedding is not None:
                embedding = precomputed_embedding
            else:
                embedding = VOICE_PROFILE_MANAGER.compute_embedding(
                    normalized_audio_path,
                    diarization_model_id=diarization_model_id,
                    embedding_model_ref=embedding_model_ref,
                    compute_device=SETTINGS.compute_device,
                    cuda_device_id=SETTINGS.cuda_device_id,
                    segments=[(start_s, end_s)],
                )
            match = VOICE_PROFILE_MANAGER.match(
                embedding,
                diarization_model_id=diarization_model_id,
                embedding_model_ref=embedding_model_ref,
                profiles=profiles,
            )
            if not match.best_match:
                ranked = VOICE_PROFILE_MANAGER.rank_matches(
                    embedding,
                    diarization_model_id=diarization_model_id,
                    embedding_model_ref=embedding_model_ref,
                    profiles=profiles,
                )
                if not ranked:
                    continue
                runner_up = ranked[1].score if len(ranked) > 1 else -1.0
                if (
                    ranked[0].score < PROFILE_EVIDENCE_FALLBACK_THRESHOLD
                    or (ranked[0].score - runner_up) < PROFILE_EVIDENCE_FALLBACK_MARGIN
                ):
                    continue
                match = type(
                    "EvidenceMatch",
                    (),
                    {
                        "best_match": ranked[0],
                        "ambiguous_matches": [],
                    },
                )()
            score = float(match.best_match.score)
            if match.ambiguous_matches:
                if best_ambiguous is None or score > best_ambiguous[0]:
                    best_ambiguous = (score, match, evidence_id)
            elif best_clear is None or score > best_clear[0]:
                best_clear = (score, match, evidence_id)
            if precomputed_embedding is not None:
                break

        if best_clear is not None:
            _, match, evidence_id = best_clear
            best = match.best_match
            JOB_STORE.append_log(
                job_id,
                f"Auto-identified {speaker_id} as {best.name} (score={best.score:.2f}, evidence={evidence_id})",
            )
            return (
                best.name,
                SpeakerProfileMatch(
                    profile_id=best.profile_id,
                    sample_id=getattr(best, "sample_id", None),
                    name=best.name,
                    score=best.score,
                    model_key=getattr(best, "model_key", None),
                    status="matched",
                ),
            )

        if best_ambiguous is not None:
            _, match, evidence_id = best_ambiguous
            best = match.best_match
            JOB_STORE.append_log(
                job_id,
                (
                    f"Profile candidates for {speaker_id}: "
                    f"{best.name} ({best.score:.2f}, evidence={evidence_id}), "
                    + ", ".join(f"{item.name} ({item.score:.2f})" for item in match.ambiguous_matches)
                ),
            )
            return (
                None,
                SpeakerProfileMatch(
                    profile_id=best.profile_id,
                    sample_id=getattr(best, "sample_id", None),
                    name=best.name,
                    score=best.score,
                    model_key=getattr(best, "model_key", None),
                    status="ambiguous",
                    ambiguous_names=[item.name for item in match.ambiguous_matches],
                ),
            )

        return None, None

    @staticmethod
    def _group_duplicate_profile_matches(job_id: str, speakers: list[SpeakerState]) -> list[SpeakerState]:
        """! @brief Collapse multiple diarized speakers that matched the same saved profile into one review group.
        @param job_id Identifier of the job being processed.
        @param speakers Speaker states to group.
        @return Grouped speaker states.
        """
        matched_groups: dict[str, list[SpeakerState]] = {}
        ungrouped: list[SpeakerState] = []
        for speaker in speakers:
            if speaker.matched_profile is not None and speaker.matched_profile.status == "matched":
                matched_groups.setdefault(speaker.matched_profile.profile_id, []).append(speaker)
            else:
                ungrouped.append(speaker)

        grouped: list[SpeakerState] = []
        for profile_id, items in matched_groups.items():
            if len(items) == 1:
                grouped.append(items[0])
                continue
            speaker_ids = [speaker.speaker_id for speaker in items]
            snippets = sorted(
                [snippet for speaker in items for snippet in speaker.snippets],
                key=lambda item: (item.start_s, item.end_s, item.speaker_id),
            )
            best = max(
                (speaker.matched_profile for speaker in items if speaker.matched_profile is not None),
                key=lambda item: item.score,
            )
            grouped.append(
                SpeakerState(
                    speaker_id=speaker_ids[0],
                    speaker_ids=speaker_ids,
                    review_group_id=f"profile:{profile_id}",
                    suggested_name=best.name,
                    matched_profile=best,
                    snippets=snippets,
                )
            )
            JOB_STORE.append_log(
                job_id,
                f"Grouped diarized speakers {', '.join(speaker_ids)} as {best.name} for review",
            )

        return sorted([*grouped, *ungrouped], key=lambda item: item.speaker_id)

    @staticmethod
    def _group_segments_by_speaker(segments: list[DiarizationSegment]) -> dict[str, list[tuple[float, float]]]:
        """! @brief Group segments by speaker.
        @param segments Segment collection processed by the operation.
        @return Dictionary produced by the operation.
        """
        grouped: dict[str, list[tuple[float, float]]] = {}
        for segment in segments:
            grouped.setdefault(segment.speaker_id, []).append((segment.start_s, segment.end_s))
        return grouped

    @staticmethod
    def _select_profile_segments(
        segments: list[tuple[float, float]],
        *,
        max_segments: int = 8,
        min_duration_s: float = 1.5,
    ) -> list[tuple[float, float]]:
        """! @brief Select profile segments.
        @param segments Segment collection processed by the operation.
        @param max_segments Value for max segments.
        @param min_duration_s Value for min duration s.
        @return List produced by the operation.
        """
        if not segments:
            return []
        eligible = [item for item in segments if (item[1] - item[0]) >= min_duration_s]
        ranked = eligible or segments
        ranked = sorted(ranked, key=lambda item: (item[1] - item[0]), reverse=True)
        return ranked[:max_segments]

    @staticmethod
    def _collapse_labeled_segments(
        segments: list[DiarizationSegment],
        speaker_map: dict[str, str],
        max_gap_s: float = 0.6,
    ) -> list[dict[str, object]]:
        """! @brief Collapse labeled segments.
        @param segments Segment collection processed by the operation.
        @param speaker_map Mapping from speaker ids to final speaker names.
        @param max_gap_s Value for max gap s.
        @return List produced by the operation.
        """
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
        """! @brief Plan transcription chunks.
        @param segments Segment collection processed by the operation.
        @param max_gap_s Value for max gap s.
        @param max_chunk_s Maximum chunk duration in seconds.
        @return List produced by the operation.
        """
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
        """! @brief Set stage.
        @param job_id Identifier of the job being processed.
        @param stage_index Value for stage index.
        """
        JOB_STORE.set_stage(job_id, STAGES[stage_index], overall_percent=self._overall(stage_index, 0.0))

    def _start_stage_heartbeat(
        self,
        job_id: str,
        stage_index: int,
        *,
        start_percent: float = 10.0,
        cap_percent: float = 92.0,
        interval_s: float = 0.8,
        assumed_duration_s: float = 90.0,
    ) -> tuple[threading.Event, threading.Thread]:
        """! @brief Start stage heartbeat.
        @param job_id Identifier of the job being processed.
        @param stage_index Value for stage index.
        @param start_percent Value for start percent.
        @param cap_percent Value for cap percent.
        @param interval_s Value for interval s.
        @param assumed_duration_s Value for assumed duration s.
        @return Tuple produced by the operation.
        """
        stop_event = threading.Event()

        def run() -> None:
            """! @brief Run operation.
            """
            started = time.monotonic()
            while not stop_event.wait(interval_s):
                elapsed = time.monotonic() - started
                progress = min(cap_percent, start_percent + (elapsed / max(1.0, assumed_duration_s)) * (cap_percent - start_percent))
                JOB_STORE.set_stage_percent(job_id, progress, overall_percent=self._overall(stage_index, progress))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return stop_event, thread

    def _overall(self, stage_index: int, stage_percent: float) -> float:
        """! @brief Overall operation.
        @param stage_index Value for stage index.
        @param stage_percent Value for stage percent.
        @return float result produced by the operation.
        """
        base = 100 / len(STAGES)
        return min(100.0, (stage_index * base) + (stage_percent / 100.0) * base)

    @staticmethod
    def _ensure_not_cancelled(job_id: str) -> None:
        """! @brief Ensure not cancelled.
        @param job_id Identifier of the job being processed.
        """
        if JOB_STORE.is_cancelled(job_id):
            raise CancelledError("Job cancelled")


ORCHESTRATOR = PipelineOrchestrator()
