from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
from pathlib import Path

from backend.app.config import SETTINGS
from backend.app.job_store import JOB_STORE, ensure_job_artifact_dir, write_json
from backend.app.schemas import JobSpeakerInfo, SpeakerSnippet, SpeakerState
from backend.pipeline.audio import extract_segment, normalize_audio, validate_audio_input
from backend.pipeline.diarization import DiarizationSegment, diarize
from backend.pipeline.formatter import Formatter
from backend.pipeline.snippets import extract_snippets, pick_snippet_ranges
from backend.pipeline.transcription import VoxtralTranscriber, transcribe_segments
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
    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=SETTINGS.max_workers)
        self._futures: dict[str, concurrent.futures.Future[None]] = {}

    def submit(self, job_id: str) -> None:
        future = self._executor.submit(self._run_job, job_id)
        self._futures[job_id] = future

    def cancel(self, job_id: str) -> None:
        JOB_STORE.cancel(job_id)

    def _run_job(self, job_id: str) -> None:
        try:
            JOB_STORE.mark_running(job_id)
            runtime = JOB_STORE.get_runtime(job_id)
            job_dir = ensure_job_artifact_dir(job_id)

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 0)
            JOB_STORE.append_log(job_id, "Stage 1/9: validating and normalizing audio")
            validate_audio_input(runtime.audio_path)
            normalized_audio_path = job_dir / "audio_normalized.wav"
            metadata = normalize_audio(runtime.audio_path, normalized_audio_path, ffmpeg_bin=SETTINGS.ffmpeg_bin)
            write_json(job_dir / "audio_metadata.json", metadata)
            JOB_STORE.set_artifact(job_id, "audio_normalized", normalized_audio_path)
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(0, 100.0))

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 1)
            JOB_STORE.append_log(job_id, "Stage 2/9: voice activity detection")
            speech_regions = detect_speech_regions(normalized_audio_path)
            vad_payload = [asdict(item) for item in speech_regions]
            write_json(job_dir / "vad_regions.json", vad_payload)
            JOB_STORE.set_artifact(job_id, "vad_regions", job_dir / "vad_regions.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(1, 100.0))

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 2)
            JOB_STORE.append_log(job_id, "Stage 3/9: speaker diarization")
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

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 3)
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

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 4)
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

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 5)
            JOB_STORE.append_log(job_id, "Stage 6/9: transcribing diarized segments")
            transcription_dir = ensure_job_artifact_dir(job_id, "transcription_segments")
            segment_jobs: list[dict[str, object]] = []
            transcriber = VoxtralTranscriber(
                SETTINGS.voxtral_binary,
                SETTINGS.voxtral_model_path,
                compute_device=SETTINGS.compute_device,
                cuda_device_id=SETTINGS.cuda_device_id,
                gpu_layers=SETTINGS.voxtral_gpu_layers,
            )
            transcriber_available = transcriber.available()

            segments_for_transcription = self._collapse_labeled_segments(diarization_result.segments, speaker_map)
            if len(segments_for_transcription) != len(diarization_result.segments):
                JOB_STORE.append_log(
                    job_id,
                    (
                        "Post-label diarization consolidation applied: "
                        f"{len(diarization_result.segments)} -> {len(segments_for_transcription)} segments"
                    ),
                )
            if (
                SETTINGS.transcription_max_segments > 0
                and len(segments_for_transcription) > SETTINGS.transcription_max_segments
            ):
                segments_for_transcription = segments_for_transcription[: SETTINGS.transcription_max_segments]
                JOB_STORE.append_log(
                    job_id,
                    f"Transcription segment cap applied: {SETTINGS.transcription_max_segments}",
                )

            for idx, segment in enumerate(segments_for_transcription, start=1):
                padding = 0.2
                start_s = max(0.0, float(segment["start_s"]) - padding)
                end_s = float(segment["end_s"]) + padding
                segment_path = transcription_dir / f"segment_{idx:04d}.wav"
                if transcriber_available:
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

            if transcriber_available:
                JOB_STORE.append_log(
                    job_id,
                    (
                        "ASR runtime detected; transcription uses external binary "
                        f"(compute={transcriber.compute_mode()})"
                    ),
                )
            else:
                JOB_STORE.append_log(
                    job_id,
                    (
                        "ASR runtime unavailable; using fallback transcription. "
                        "Skipping per-segment audio extraction. "
                        f"binary='{SETTINGS.voxtral_binary or '<unset>'}' "
                        f"model='{SETTINGS.voxtral_model_path or '<unset>'}'"
                    ),
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

            transcript_segments = transcribe_segments(transcriber, segment_jobs, progress_callback=_segment_progress)
            fallback_segments = sum(
                1
                for item in transcript_segments
                if str(item.get("text", "")).startswith("[Offline fallback transcript")
            )
            if fallback_segments:
                JOB_STORE.append_log(
                    job_id,
                    f"ASR fallback segments: {fallback_segments}/{len(transcript_segments)}",
                )
                if fallback_segments == len(transcript_segments):
                    JOB_STORE.append_log(
                        job_id,
                        "Warning: ASR produced only fallback text; check ASR binary/model compatibility.",
                    )
            write_json(job_dir / "segments_transcript.json", transcript_segments)
            JOB_STORE.set_artifact(job_id, "segments_transcript", job_dir / "segments_transcript.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(5, 100.0))

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 6)
            JOB_STORE.append_log(job_id, "Stage 7/9: assembling transcript")
            transcript_payload = {
                "speakers": sorted(set(item["speaker_name"] for item in transcript_segments)),
                "segments": transcript_segments,
            }
            write_json(job_dir / "transcript.json", transcript_payload)
            JOB_STORE.set_artifact(job_id, "transcript", job_dir / "transcript.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(6, 100.0))

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 7)
            JOB_STORE.append_log(job_id, "Stage 8/9: formatting minutes of meeting")
            formatter = Formatter(
                command_template=SETTINGS.formatter_command,
                model_path=SETTINGS.formatter_model_path,
                compute_device=SETTINGS.compute_device,
                cuda_device_id=SETTINGS.cuda_device_id,
                gpu_layers=SETTINGS.formatter_gpu_layers,
            )
            title = runtime.title or runtime.audio_path.stem
            speakers = transcript_payload["speakers"]
            markdown, structured, prompt = formatter.build_structured_summary(
                transcript=transcript_segments,
                speakers=speakers,
                title=title,
                template_id=runtime.template_id,
            )
            JOB_STORE.append_log(job_id, f"Formatter mode: {formatter.last_mode}")
            if formatter.last_raw_output:
                (job_dir / "formatter_raw_output.txt").write_text(formatter.last_raw_output, encoding="utf-8")
                JOB_STORE.set_artifact(job_id, "formatter_raw_output", job_dir / "formatter_raw_output.txt")
            (job_dir / "mom.md").write_text(markdown, encoding="utf-8")
            write_json(job_dir / "mom_structured.json", structured)
            (job_dir / "formatter_prompt.txt").write_text(prompt, encoding="utf-8")
            JOB_STORE.set_artifact(job_id, "mom_markdown", job_dir / "mom.md")
            JOB_STORE.set_artifact(job_id, "mom_structured", job_dir / "mom_structured.json")
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(7, 100.0))

            self._ensure_not_cancelled(job_id)
            self._set_stage(job_id, 8)
            JOB_STORE.append_log(job_id, "Stage 9/9: export completed")
            export_dir = ensure_job_artifact_dir(job_id, "export")
            export_path = export_dir / "mom.md"
            export_path.write_text((job_dir / "mom.md").read_text(encoding="utf-8"), encoding="utf-8")
            JOB_STORE.set_artifact(job_id, "export_markdown", export_path)
            JOB_STORE.set_stage_percent(job_id, 100.0, overall_percent=self._overall(8, 100.0))

            JOB_STORE.mark_completed(job_id)
            JOB_STORE.append_log(job_id, "Job completed successfully")
        except CancelledError as exc:
            JOB_STORE.append_log(job_id, str(exc))
            JOB_STORE.cancel(job_id)
        except Exception as exc:  # pragma: no cover - this is fallback guard
            JOB_STORE.mark_failed(job_id, str(exc))
            JOB_STORE.append_log(job_id, f"Job failed: {exc}")

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
