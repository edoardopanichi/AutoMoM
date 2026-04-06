from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import copy
import math
import warnings
from typing import Callable

import numpy as np
import soundfile as sf

from backend.pipeline.compute import resolve_torch_device
from backend.pipeline.vad import SpeechRegion


CHUNKED_LOCAL_DIARIZATION_THRESHOLD_S = 20.0 * 60.0
TARGET_LOCAL_DIARIZATION_CHUNK_S = 20.0 * 60.0
CHUNK_OVERLAP_S = 30.0
CHUNK_SNAP_WINDOW_S = 15.0
STITCH_MATCH_THRESHOLD = 0.82
STITCH_AMBIGUITY_MARGIN = 0.03
MAX_GLOBAL_SPEAKER_EMBEDDINGS = 5


@dataclass
class DiarizationSegment:
    speaker_id: str
    start_s: float
    end_s: float
    confidence: float | None = None


@dataclass
class DiarizationResult:
    segments: list[DiarizationSegment]
    speaker_count: int
    mode: str = "heuristic"
    details: str | None = None
    chunk_plan: list[dict[str, object]] | None = None
    stitching_debug: dict[str, object] | None = None

    def to_json(self) -> list[dict[str, object]]:
        return [asdict(item) for item in self.segments]



def diarize(
    audio_path: Path,
    speech_regions: list[SpeechRegion],
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    max_chunk_s: float = 18.0,
    backend: str = "auto",
    model_path: Path | None = None,
    pipeline_path: str | None = None,
    embedding_model: str | None = None,
    compute_device: str = "auto",
    cuda_device_id: int = 0,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> DiarizationResult:
    normalized_min = int(min_speakers) if min_speakers is not None else 0
    normalized_max = int(max_speakers) if max_speakers is not None else 0
    min_speakers = normalized_min if normalized_min > 0 else None
    max_speakers = normalized_max if normalized_max > 0 else None
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        min_speakers, max_speakers = max_speakers, min_speakers
    normalized_backend = (backend or "auto").strip().lower()
    if normalized_backend not in {"auto", "heuristic", "pyannote", "embedding"}:
        raise ValueError(f"Unsupported diarization backend: {backend}")

    if normalized_backend == "auto":
        normalized_backend = "pyannote"

    if normalized_backend == "pyannote":
        pyannote_result, error = _diarize_with_pyannote(
            audio_path=audio_path,
            speech_regions=speech_regions,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            model_path=model_path,
            pipeline_path=pipeline_path,
            compute_device=compute_device,
            cuda_device_id=cuda_device_id,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
        )
        if pyannote_result is not None:
            return pyannote_result
        raise RuntimeError(_pyannote_error_message(error, model_path=model_path, pipeline_path=pipeline_path))

    if normalized_backend == "embedding":
        embedding_result, error = _diarize_with_embeddings(
            audio_path=audio_path,
            speech_regions=speech_regions,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            max_chunk_s=max_chunk_s,
            model_ref=embedding_model,
            compute_device=compute_device,
            cuda_device_id=cuda_device_id,
        )
        if embedding_result is not None:
            return embedding_result
        raise RuntimeError(_embedding_error_message(error))

    return _diarize_heuristic(
        audio_path=audio_path,
        speech_regions=speech_regions,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        max_chunk_s=max_chunk_s,
        details="heuristic_backend_selected",
    )


def _pyannote_error_message(
    error: str | None,
    *,
    model_path: Path | None,
    pipeline_path: str | None,
) -> str:
    pipeline_ref = (pipeline_path or "").strip()
    if not pipeline_ref and model_path is not None:
        pipeline_ref = str(model_path)
    reason = error or "unknown_pyannote_error"
    if reason == "pyannote_pipeline_not_configured":
        return (
            "Diarization model/pipeline is not configured. "
            "Fix: set AUTOMOM_DIARIZATION_PIPELINE (or AUTOMOM_DIARIZATION_MODEL) "
            "to a valid local pyannote pipeline YAML (e.g. .../config.yaml)."
        )
    if reason.startswith("pyannote_import_error"):
        return (
            f"Pyannote diarization runtime is unavailable ({reason}). "
            "Fix: install pyannote.audio and torch dependencies, then retry."
        )
    if reason.startswith("pyannote_runtime_error"):
        if "outofmemoryerror" in reason.lower() or "out_of_memory" in reason.lower():
            return (
                f"Pyannote diarization ran out of memory ({reason}). "
                f"Configured pipeline: '{pipeline_ref or '<unset>'}'. "
                "Fix: retry on CPU, reduce concurrent GPU load, or use a shorter recording chunk."
            )
        return (
            f"Pyannote diarization failed at runtime ({reason}). "
            f"Configured pipeline: '{pipeline_ref or '<unset>'}'. "
            "Fix: verify the pipeline files and HF token requirements, then retry."
        )
    if reason.startswith("pyannote_parse_error") or reason == "pyannote_no_segments":
        return (
            f"Pyannote diarization returned unusable output ({reason}). "
            "Fix: verify the input audio and pipeline configuration."
        )
    return (
        f"Pyannote diarization unavailable ({reason}). "
        "Fix: verify AUTOMOM_DIARIZATION_MODEL/AUTOMOM_DIARIZATION_PIPELINE and dependencies."
    )


def _embedding_error_message(error: str | None) -> str:
    reason = error or "unknown_embedding_error"
    return (
        f"Embedding diarization unavailable ({reason}). "
        "Fix: install embedding dependencies and set AUTOMOM_DIARIZATION_EMBEDDING_MODEL to a valid model."
    )


def _diarize_with_pyannote(
    audio_path: Path,
    speech_regions: list[SpeechRegion],
    min_speakers: int | None,
    max_speakers: int | None,
    model_path: Path | None,
    compute_device: str,
    cuda_device_id: int,
    pipeline_path: str | None = None,
    embedding_model: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[DiarizationResult | None, str | None]:
    pipeline_ref = _resolve_pyannote_pipeline_ref(model_path, pipeline_path)
    if not pipeline_ref:
        return None, "pyannote_pipeline_not_configured"

    try:
        import torch
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"pyannote\.audio\.core\.io",
            )
            from pyannote.audio import Pipeline
    except Exception as exc:
        return None, f"pyannote_import_error:{exc.__class__.__name__}"

    try:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
        info = sf.info(str(audio_path))
        total_duration_s = float(info.frames) / max(1, int(info.samplerate))
        if total_duration_s > CHUNKED_LOCAL_DIARIZATION_THRESHOLD_S:
            return _diarize_with_pyannote_in_chunks(
                audio_path=audio_path,
                speech_regions=speech_regions,
                total_duration_s=total_duration_s,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                pipeline_ref=pipeline_ref,
                embedding_model=resolve_profile_embedding_model_ref(
                    model_path=model_path,
                    pipeline_path=pipeline_path,
                    embedding_model=embedding_model,
                ),
                compute_device=compute_device,
                cuda_device_id=cuda_device_id,
                token=token,
                progress_callback=progress_callback,
            )

        pipeline = copy.deepcopy(_load_pyannote_pipeline(pipeline_ref, token))
        target_device = resolve_torch_device(compute_device, cuda_device_id)
        if progress_callback is not None:
            progress_callback({"phase": "loading", "detail": "Running full-recording diarization"})

        input_payload = _read_audio_payload(audio_path)
        pipeline_kwargs: dict[str, int] = {}
        if min_speakers is not None:
            pipeline_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            pipeline_kwargs["max_speakers"] = max_speakers

        diarization, active_device = _run_pyannote_pipeline(
            pipeline,
            input_payload,
            pipeline_kwargs,
            torch,
            target_device=target_device,
        )
    except Exception as exc:
        return None, f"pyannote_runtime_error:{exc.__class__.__name__}"

    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
    raw_segments: list[DiarizationSegment] = []
    try:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            raw_segments.append(
                DiarizationSegment(
                    speaker_id=str(speaker),
                    start_s=float(turn.start),
                    end_s=float(turn.end),
                    confidence=None,
                )
            )
    except Exception as exc:
        return None, f"pyannote_parse_error:{exc.__class__.__name__}"

    if not raw_segments:
        return None, "pyannote_no_segments"

    merged = _merge_segments(raw_segments, max_gap_s=0.25)
    remapped = _remap_speakers(merged)
    return (
        DiarizationResult(
            segments=remapped,
            speaker_count=len({segment.speaker_id for segment in remapped}),
            mode="pyannote",
            details=f"pyannote_pipeline:device:{active_device}",
        ),
        None,
    )


def _diarize_with_pyannote_in_chunks(
    *,
    audio_path: Path,
    speech_regions: list[SpeechRegion],
    total_duration_s: float,
    min_speakers: int | None,
    max_speakers: int | None,
    pipeline_ref: str,
    embedding_model: str,
    compute_device: str,
    cuda_device_id: int,
    token: str | None,
    progress_callback: Callable[[dict[str, object]], None] | None,
) -> tuple[DiarizationResult | None, str | None]:
    try:
        import torch
    except Exception as exc:
        return None, f"pyannote_import_error:{exc.__class__.__name__}"

    chunk_plan = _plan_chunked_diarization(
        speech_regions=speech_regions,
        total_duration_s=total_duration_s,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "chunk_plan",
                "detail": f"Chunked diarization enabled ({len(chunk_plan)} chunks)",
                "chunk_count": len(chunk_plan),
                "processed_s": 0.0,
                "total_s": total_duration_s,
            }
        )

    pipeline = copy.deepcopy(_load_pyannote_pipeline(pipeline_ref, token))
    target_device = resolve_torch_device(compute_device, cuda_device_id)
    pipeline_kwargs: dict[str, int] = {}
    if min_speakers is not None:
        pipeline_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pipeline_kwargs["max_speakers"] = max_speakers

    global_bank: dict[str, list[np.ndarray]] = {}
    speaker_order: list[str] = []
    debug_matches: list[dict[str, object]] = []
    all_segments: list[DiarizationSegment] = []
    processed_owned_s = 0.0

    try:
        inference, _ = _load_embedding_inference(
            embedding_model,
            token,
            target_device,
            cuda_device_id,
        )
    except Exception as exc:
        return None, f"embedding_model_load_error:{exc.__class__.__name__}"

    for index, chunk in enumerate(chunk_plan, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "chunk",
                    "detail": f"Chunk {index}/{len(chunk_plan)}",
                    "chunk_index": index,
                    "chunk_count": len(chunk_plan),
                    "processed_s": processed_owned_s,
                    "total_s": total_duration_s,
                }
            )

        input_payload = _read_audio_payload(
            audio_path,
            start_s=float(chunk["audio_start_s"]),
            end_s=float(chunk["audio_end_s"]),
        )
        diarization, active_device = _run_pyannote_pipeline(
            pipeline,
            input_payload,
            pipeline_kwargs,
            torch,
            target_device=target_device,
        )
        annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
        raw_segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            global_start = float(chunk["audio_start_s"]) + float(turn.start)
            global_end = float(chunk["audio_start_s"]) + float(turn.end)
            raw_segments.append(
                DiarizationSegment(
                    speaker_id=str(speaker),
                    start_s=global_start,
                    end_s=global_end,
                    confidence=None,
                )
            )

        owned_segments = _filter_segments_to_owned_window(
            raw_segments,
            own_start_s=float(chunk["own_start_s"]),
            own_end_s=float(chunk["own_end_s"]),
        )
        merged_owned = _merge_segments(owned_segments, max_gap_s=0.25)
        representative = _build_chunk_speaker_embeddings(
            segments=merged_owned,
            audio_path=audio_path,
            inference=inference,
            sample_rate=int(input_payload["sample_rate"]),
            chunk_audio_start_s=float(chunk["audio_start_s"]),
            chunk_audio_end_s=float(chunk["audio_end_s"]),
        )
        local_to_global, match_debug = _assign_chunk_speakers_to_global(
            representative,
            global_bank,
            speaker_order,
        )
        debug_matches.append(
            {
                "chunk_index": index,
                "device": active_device,
                "matches": match_debug,
            }
        )
        for segment in merged_owned:
            segment.speaker_id = local_to_global.get(segment.speaker_id, segment.speaker_id)
        all_segments.extend(merged_owned)
        processed_owned_s += float(chunk["own_end_s"]) - float(chunk["own_start_s"])
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "chunk_complete",
                    "detail": f"Chunk {index}/{len(chunk_plan)} completed",
                    "chunk_index": index,
                    "chunk_count": len(chunk_plan),
                    "processed_s": processed_owned_s,
                    "total_s": total_duration_s,
                }
            )

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "finalizing",
                "detail": "Finalizing global speaker stitching",
                "processed_s": total_duration_s,
                "total_s": total_duration_s,
            }
        )

    merged = _merge_segments(all_segments, max_gap_s=0.25)
    remapped = _remap_speakers(merged)
    return (
        DiarizationResult(
            segments=remapped,
            speaker_count=len({segment.speaker_id for segment in remapped}),
            mode="pyannote",
            details=f"pyannote_chunked:chunks:{len(chunk_plan)}",
            chunk_plan=chunk_plan,
            stitching_debug={"matches": debug_matches},
        ),
        None,
    )


def _read_audio_payload(audio_path: Path, start_s: float | None = None, end_s: float | None = None) -> dict[str, object]:
    with sf.SoundFile(str(audio_path)) as handle:
        sample_rate = int(handle.samplerate)
        total_frames = len(handle)
        start_frame = 0 if start_s is None else max(0, int(start_s * sample_rate))
        end_frame = total_frames if end_s is None else min(total_frames, int(end_s * sample_rate))
        handle.seek(start_frame)
        audio = handle.read(max(0, end_frame - start_frame), dtype="float32", always_2d=False)

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        waveform = np.ascontiguousarray(audio.T)
    else:
        waveform = np.ascontiguousarray(np.atleast_2d(audio))

    import torch

    return {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}


def _plan_chunked_diarization(
    *,
    speech_regions: list[SpeechRegion],
    total_duration_s: float,
) -> list[dict[str, object]]:
    chunk_count = max(1, int(math.ceil(total_duration_s / TARGET_LOCAL_DIARIZATION_CHUNK_S)))
    own_chunk_s = total_duration_s / chunk_count
    silence_points = _silence_boundary_points(speech_regions, total_duration_s)
    boundaries = [0.0]
    for index in range(1, chunk_count):
        target = own_chunk_s * index
        boundaries.append(_snap_boundary(target, silence_points, total_duration_s))
    boundaries.append(total_duration_s)
    boundaries = _monotonic_boundaries(boundaries, total_duration_s)

    chunks: list[dict[str, object]] = []
    for index in range(chunk_count):
        own_start = boundaries[index]
        own_end = boundaries[index + 1]
        chunks.append(
            {
                "chunk_index": index + 1,
                "own_start_s": own_start,
                "own_end_s": own_end,
                "audio_start_s": max(0.0, own_start - CHUNK_OVERLAP_S),
                "audio_end_s": min(total_duration_s, own_end + CHUNK_OVERLAP_S),
            }
        )
    return chunks


def _silence_boundary_points(speech_regions: list[SpeechRegion], total_duration_s: float) -> list[float]:
    if not speech_regions:
        return []
    points: list[float] = []
    previous_end = 0.0
    for region in speech_regions:
        if region.start_s > previous_end:
            points.append((previous_end + region.start_s) / 2.0)
        previous_end = max(previous_end, region.end_s)
    if previous_end < total_duration_s:
        points.append((previous_end + total_duration_s) / 2.0)
    return points


def _snap_boundary(target_s: float, silence_points: list[float], total_duration_s: float) -> float:
    candidates = [point for point in silence_points if abs(point - target_s) <= CHUNK_SNAP_WINDOW_S]
    if not candidates:
        return min(total_duration_s, max(0.0, target_s))
    return min(candidates, key=lambda point: abs(point - target_s))


def _monotonic_boundaries(boundaries: list[float], total_duration_s: float) -> list[float]:
    normalized = [0.0]
    for value in boundaries[1:-1]:
        normalized.append(min(total_duration_s, max(normalized[-1] + 1.0, value)))
    normalized.append(total_duration_s)
    return normalized


def _filter_segments_to_owned_window(
    segments: list[DiarizationSegment],
    *,
    own_start_s: float,
    own_end_s: float,
) -> list[DiarizationSegment]:
    kept: list[DiarizationSegment] = []
    for segment in segments:
        midpoint = (float(segment.start_s) + float(segment.end_s)) / 2.0
        if midpoint < own_start_s or midpoint > own_end_s:
            continue
        clipped_start = max(float(segment.start_s), own_start_s)
        clipped_end = min(float(segment.end_s), own_end_s)
        if clipped_end <= clipped_start:
            continue
        kept.append(
            DiarizationSegment(
                speaker_id=segment.speaker_id,
                start_s=clipped_start,
                end_s=clipped_end,
                confidence=segment.confidence,
            )
        )
    return kept


def _build_chunk_speaker_embeddings(
    *,
    segments: list[DiarizationSegment],
    audio_path: Path,
    inference,
    sample_rate: int,
    chunk_audio_start_s: float,
    chunk_audio_end_s: float,
) -> dict[str, np.ndarray]:
    if not segments:
        return {}

    with sf.SoundFile(str(audio_path)) as handle:
        start_frame = max(0, int(chunk_audio_start_s * sample_rate))
        end_frame = min(len(handle), int(chunk_audio_end_s * sample_rate))
        handle.seek(start_frame)
        audio = handle.read(max(0, end_frame - start_frame), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    else:
        mono = np.asarray(audio, dtype=np.float32)

    speaker_segments: dict[str, list[DiarizationSegment]] = {}
    for segment in segments:
        speaker_segments.setdefault(segment.speaker_id, []).append(segment)

    result: dict[str, np.ndarray] = {}
    for speaker_id, items in speaker_segments.items():
        embeddings: list[np.ndarray] = []
        ranked = sorted(items, key=lambda item: item.end_s - item.start_s, reverse=True)[:3]
        for item in ranked:
            rel_start = max(0, int((float(item.start_s) - chunk_audio_start_s) * sample_rate))
            rel_end = max(rel_start + 1, int((float(item.end_s) - chunk_audio_start_s) * sample_rate))
            clip = mono[rel_start:rel_end]
            if clip.size < int(sample_rate * 0.5):
                continue
            embeddings.append(_compute_embedding_from_clip(clip, sample_rate, inference))
        if embeddings:
            result[speaker_id] = _normalize_embedding(np.vstack(embeddings).mean(axis=0))
    return result


def _compute_embedding_from_clip(audio: np.ndarray, sample_rate: int, inference) -> np.ndarray:
    import torch

    waveform = np.ascontiguousarray(np.atleast_2d(audio))
    embedding = inference({"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate})
    return _normalize_embedding(np.asarray(embedding, dtype=np.float32).reshape(-1))


def _assign_chunk_speakers_to_global(
    representative: dict[str, np.ndarray],
    global_bank: dict[str, list[np.ndarray]],
    speaker_order: list[str],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    mapping: dict[str, str] = {}
    debug_rows: list[dict[str, object]] = []
    used_globals: set[str] = set()

    scored_locals: list[tuple[str, float, str | None, float | None]] = []
    for local_speaker, embedding in representative.items():
        best_global = None
        best_score = -1.0
        second_score = -1.0
        for global_speaker, bank in global_bank.items():
            score = max(float(np.dot(embedding, item)) for item in bank)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_global = global_speaker
            elif score > second_score:
                second_score = score
        scored_locals.append((local_speaker, best_score, best_global, second_score))

    for local_speaker, best_score, best_global, second_score in sorted(scored_locals, key=lambda item: item[1], reverse=True):
        chosen = None
        matched_existing = False
        if (
            best_global is not None
            and best_global not in used_globals
            and best_score >= STITCH_MATCH_THRESHOLD
            and (second_score < 0 or (best_score - second_score) > STITCH_AMBIGUITY_MARGIN)
        ):
            chosen = best_global
            matched_existing = True
        if chosen is None:
            chosen = f"GLOBAL_{len(speaker_order)}"
            speaker_order.append(chosen)
            global_bank.setdefault(chosen, [])
        used_globals.add(chosen)
        mapping[local_speaker] = chosen
        bank = global_bank.setdefault(chosen, [])
        bank.append(representative[local_speaker])
        if len(bank) > MAX_GLOBAL_SPEAKER_EMBEDDINGS:
            del bank[:-MAX_GLOBAL_SPEAKER_EMBEDDINGS]
        debug_rows.append(
            {
                "local_speaker_id": local_speaker,
                "assigned_global_speaker_id": chosen,
                "best_score": None if best_score < 0 else round(best_score, 4),
                "runner_up_score": None if second_score < 0 else round(second_score, 4),
                "matched_existing": matched_existing,
            }
        )
    return mapping, debug_rows


@lru_cache(maxsize=2)
def _load_pyannote_pipeline(pipeline_ref: str, token: str | None):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        from pyannote.audio import Pipeline

    kwargs: dict[str, object] = {}
    if token:
        kwargs["use_auth_token"] = token
    return Pipeline.from_pretrained(pipeline_ref, **kwargs)


def _run_pyannote_pipeline(
    pipeline,
    input_payload: dict[str, object],
    pipeline_kwargs: dict[str, int],
    torch_module,
    *,
    target_device: str,
) -> tuple[object, str]:
    active_device = _move_pyannote_pipeline(pipeline, torch_module, target_device)
    try:
        return _invoke_pyannote_pipeline(pipeline, input_payload, pipeline_kwargs), active_device
    except Exception as exc:
        if active_device == "cuda" and _is_cuda_oom(exc):
            try:
                if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
                    torch_module.cuda.empty_cache()
            except Exception:
                pass
            active_device = _move_pyannote_pipeline(pipeline, torch_module, "cpu")
            return _invoke_pyannote_pipeline(pipeline, input_payload, pipeline_kwargs), active_device
        raise


def _move_pyannote_pipeline(pipeline, torch_module, target_device: str) -> str:
    active_device = target_device
    try:
        pipeline.to(torch_module.device(target_device))
    except Exception:
        if target_device == "cuda":
            try:
                pipeline.to(torch_module.device("cpu"))
                active_device = "cpu"
            except Exception:
                pass
    return active_device


def _invoke_pyannote_pipeline(pipeline, input_payload: dict[str, object], pipeline_kwargs: dict[str, int]):
    try:
        return pipeline(input_payload, **pipeline_kwargs)
    except TypeError:
        return pipeline(input_payload)


def _is_cuda_oom(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "out of memory" in message or "cuda out of memory" in message


def _diarize_with_embeddings(
    audio_path: Path,
    speech_regions: list[SpeechRegion],
    min_speakers: int | None,
    max_speakers: int | None,
    max_chunk_s: float,
    model_ref: str | None,
    compute_device: str,
    cuda_device_id: int,
) -> tuple[DiarizationResult | None, str | None]:
    try:
        import torch
    except Exception as exc:
        return None, f"embedding_import_error:{exc.__class__.__name__}"

    resolved_model_ref = (model_ref or os.getenv("AUTOMOM_DIARIZATION_EMBEDDING_MODEL", "")).strip()
    if not resolved_model_ref:
        resolved_model_ref = "pyannote/wespeaker-voxceleb-resnet34-LM"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    target_device = resolve_torch_device(compute_device, cuda_device_id)
    try:
        inference, active_device = _load_embedding_inference(
            resolved_model_ref,
            token,
            target_device,
            cuda_device_id,
        )
    except Exception as exc:
        return None, f"embedding_model_load_error:{exc.__class__.__name__}"

    try:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        return None, f"embedding_audio_read_error:{exc.__class__.__name__}"

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        waveform = np.ascontiguousarray(audio.T)
    else:
        waveform = np.ascontiguousarray(np.atleast_2d(audio))
    waveform_t = torch.from_numpy(waveform)

    chunks = _build_chunks(
        speech_regions=speech_regions,
        max_chunk_s=max_chunk_s,
        total_duration_s=float(waveform_t.shape[1]) / max(1, int(sample_rate)),
    )
    if not chunks:
        return None, "embedding_no_chunks"

    valid_chunks: list[tuple[float, float]] = []
    features: list[np.ndarray] = []
    for start_s, end_s in chunks:
        start_idx = max(0, int(start_s * sample_rate))
        end_idx = min(int(end_s * sample_rate), waveform_t.shape[1])
        if end_idx - start_idx <= 1:
            continue
        clip = waveform_t[:, start_idx:end_idx]
        try:
            embedding = inference({"waveform": clip, "sample_rate": int(sample_rate)})
        except Exception:
            continue

        embedding_array = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding_array.size == 0 or not np.all(np.isfinite(embedding_array)):
            continue
        features.append(embedding_array)
        valid_chunks.append((start_s, end_s))

    if not valid_chunks or not features:
        return None, "embedding_no_segments"

    feature_matrix = np.vstack(features)
    effective_max = len(feature_matrix) if max_speakers is None else max(1, min(max_speakers, len(feature_matrix)))
    min_allowed = 1 if min_speakers is None else max(1, min(min_speakers, effective_max))
    speaker_count = max(min_allowed, _estimate_speaker_count(feature_matrix, max_speakers=effective_max))
    labels = _cluster(feature_matrix, speaker_count)

    raw_segments: list[DiarizationSegment] = []
    for idx, (start_s, end_s) in enumerate(valid_chunks):
        raw_segments.append(
            DiarizationSegment(
                speaker_id=f"SPEAKER_{int(labels[idx])}",
                start_s=float(start_s),
                end_s=float(end_s),
                confidence=0.82,
            )
        )

    merged = _merge_segments(raw_segments)
    remapped = _remap_speakers(merged)
    return (
        DiarizationResult(
            segments=remapped,
            speaker_count=len({segment.speaker_id for segment in remapped}),
            mode="embedding",
            details=f"embedding_model:{resolved_model_ref};device:{active_device}",
        ),
        None,
    )


def resolve_profile_embedding_model_ref(
    *,
    model_path: Path | None = None,
    pipeline_path: str | None = None,
    embedding_model: str | None = None,
) -> str:
    pipeline_ref = _resolve_pyannote_pipeline_ref(model_path, pipeline_path)
    if pipeline_ref:
        pipeline_dir = Path(pipeline_ref).expanduser().resolve().parent
        embedded_model = pipeline_dir / "embedding"
        if embedded_model.exists():
            return str(embedded_model)
    normalized = (embedding_model or os.getenv("AUTOMOM_DIARIZATION_EMBEDDING_MODEL", "")).strip()
    if normalized:
        return normalized
    return "pyannote/wespeaker-voxceleb-resnet34-LM"


def pyannote_audio_version() -> str:
    try:
        return importlib_metadata.version("pyannote.audio")
    except Exception:
        return "unknown"


def compute_profile_embedding(
    audio_path: Path,
    *,
    model_ref: str,
    compute_device: str = "auto",
    cuda_device_id: int = 0,
    segments: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    import torch

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    target_device = resolve_torch_device(compute_device, cuda_device_id)
    inference, _ = _load_embedding_inference(model_ref, token, target_device, cuda_device_id)
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        waveform = np.ascontiguousarray(audio.T)
    else:
        waveform = np.ascontiguousarray(np.atleast_2d(audio))
    waveform_t = torch.from_numpy(waveform)

    windows = segments or [(0.0, float(waveform_t.shape[1]) / max(1, int(sample_rate)))]
    embeddings: list[np.ndarray] = []
    for start_s, end_s in windows:
        start_idx = max(0, int(start_s * sample_rate))
        end_idx = min(int(end_s * sample_rate), waveform_t.shape[1])
        if end_idx - start_idx <= 1:
            continue
        clip = waveform_t[:, start_idx:end_idx]
        embedding = inference({"waveform": clip, "sample_rate": int(sample_rate)})
        embedding_array = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding_array.size == 0 or not np.all(np.isfinite(embedding_array)):
            continue
        embeddings.append(embedding_array)
    if not embeddings:
        raise RuntimeError("Profile embedding generation returned no usable vectors.")
    stacked = np.vstack(embeddings)
    return _normalize_embedding(stacked.mean(axis=0))


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


@lru_cache(maxsize=3)
def _load_embedding_inference(
    model_ref: str,
    token: str | None,
    target_device: str,
    cuda_device_id: int,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        import torch
        from pyannote.audio import Inference, Model

    kwargs: dict[str, object] = {}
    if token:
        kwargs["token"] = token

    embedding_model = Model.from_pretrained(model_ref, **kwargs)
    active_device = target_device
    if target_device == "cuda":
        try:
            embedding_model = embedding_model.to(torch.device(f"cuda:{max(0, int(cuda_device_id))}"))
        except Exception:
            active_device = "cpu"
    if active_device == "cpu":
        try:
            embedding_model = embedding_model.to(torch.device("cpu"))
        except Exception:
            pass

    try:
        inference = Inference(
            embedding_model,
            window="whole",
            device=torch.device("cpu" if active_device == "cpu" else f"cuda:{max(0, int(cuda_device_id))}"),
        )
        return inference, active_device
    except TypeError:
        inference = Inference(embedding_model, window="whole")
        if hasattr(inference, "to"):
            try:
                inference.to(torch.device("cpu" if active_device == "cpu" else f"cuda:{max(0, int(cuda_device_id))}"))
            except Exception:
                if active_device == "cuda":
                    active_device = "cpu"
        return inference, active_device
    except Exception:
        if active_device != "cuda":
            raise
        active_device = "cpu"
        try:
            inference = Inference(embedding_model, window="whole", device=torch.device("cpu"))
            return inference, active_device
        except TypeError:
            inference = Inference(embedding_model, window="whole")
            if hasattr(inference, "to"):
                try:
                    inference.to(torch.device("cpu"))
                except Exception:
                    pass
            return inference, active_device


def _resolve_pyannote_pipeline_ref(model_path: Path | None, pipeline_path: str | None = None) -> str | None:
    explicit = (pipeline_path or "").strip()
    if explicit:
        return explicit

    env_explicit = os.getenv("AUTOMOM_DIARIZATION_PIPELINE", "").strip()
    if env_explicit:
        return env_explicit

    if model_path is None:
        return None

    candidates = []
    if model_path.exists():
        if model_path.is_dir():
            return str(model_path)
        if model_path.suffix.lower() in {".yaml", ".yml"}:
            return str(model_path)
        candidates.append(model_path.parent / "config.yaml")
        candidates.append(model_path.parent / "pipeline.yaml")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _diarize_heuristic(
    audio_path: Path,
    speech_regions: list[SpeechRegion],
    min_speakers: int | None,
    max_speakers: int | None,
    max_chunk_s: float,
    details: str | None = None,
) -> DiarizationResult:
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)

    if not speech_regions:
        duration_s = len(audio) / sample_rate
        return DiarizationResult(
            segments=[DiarizationSegment(speaker_id="SPEAKER_0", start_s=0.0, end_s=duration_s, confidence=0.4)],
            speaker_count=1,
            mode="heuristic",
            details=details,
        )

    chunks = _build_chunks(
        speech_regions=speech_regions,
        max_chunk_s=max_chunk_s,
        total_duration_s=len(audio) / sample_rate,
    )

    features = np.array([_segment_features(audio, sample_rate, start, end) for start, end in chunks], dtype=np.float32)
    effective_max = len(features) if max_speakers is None else max(1, min(max_speakers, len(features)))
    min_allowed = 1 if min_speakers is None else max(1, min(min_speakers, effective_max))
    speaker_count = max(min_allowed, _estimate_speaker_count(features, max_speakers=effective_max))
    labels = _cluster(features, speaker_count)

    raw_segments: list[DiarizationSegment] = []
    for idx, (start_s, end_s) in enumerate(chunks):
        raw_segments.append(
            DiarizationSegment(
                speaker_id=f"SPEAKER_{int(labels[idx])}",
                start_s=float(start_s),
                end_s=float(end_s),
                confidence=0.7,
            )
        )

    merged = _merge_segments(raw_segments)
    remapped = _remap_speakers(merged)

    return DiarizationResult(
        segments=remapped,
        speaker_count=len({segment.speaker_id for segment in remapped}),
        mode="heuristic",
        details=details,
    )


def _remap_speakers(segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
    unique_speakers = sorted({segment.speaker_id for segment in segments})
    mapping = {speaker: f"SPEAKER_{idx}" for idx, speaker in enumerate(unique_speakers)}
    for segment in segments:
        segment.speaker_id = mapping[segment.speaker_id]
    return segments



def _build_chunks(
    speech_regions: list[SpeechRegion],
    max_chunk_s: float,
    total_duration_s: float,
) -> list[tuple[float, float]]:
    if not speech_regions:
        return [(0.0, total_duration_s)]

    chunks: list[tuple[float, float]] = []
    for region in speech_regions:
        cursor = region.start_s
        while cursor < region.end_s:
            end = min(region.end_s, cursor + max_chunk_s)
            chunks.append((cursor, end))
            cursor = end
    return chunks


def _segment_features(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start_idx = int(start_s * sample_rate)
    end_idx = int(end_s * sample_rate)
    segment = audio[max(0, start_idx) : max(1, end_idx)]
    if segment.size == 0:
        return np.zeros(4, dtype=np.float32)

    energy = float(np.sqrt(np.mean(np.square(segment))) + 1e-8)
    zero_cross = float(np.mean(np.abs(np.diff(np.sign(segment))) > 0))
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)
    if spectrum.sum() == 0:
        centroid = 0.0
        bandwidth = 0.0
    else:
        centroid = float((freqs * spectrum).sum() / spectrum.sum())
        bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * spectrum).sum() / spectrum.sum()))

    return np.array([energy, zero_cross, centroid / 5000.0, bandwidth / 5000.0], dtype=np.float32)



def _estimate_speaker_count(features: np.ndarray, max_speakers: int) -> int:
    if len(features) <= 1:
        return 1

    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    max_k = min(max_speakers, max(2, len(features)))
    best_k = 1
    best_adjusted_score = -1.0
    for k in range(2, max_k + 1):
        labels = _cluster(features, k)
        raw_score = _silhouette(labels, distances)
        _, cluster_sizes = np.unique(labels, return_counts=True)
        singleton_ratio = float((cluster_sizes == 1).sum()) / max(1, len(features))

        # Favor compact, stable clusterings and avoid over-fragmenting into many tiny clusters.
        adjusted_score = raw_score - (0.015 * (k - 1)) - (0.35 * singleton_ratio)
        if adjusted_score > best_adjusted_score:
            best_adjusted_score = adjusted_score
            best_k = k

    if best_adjusted_score < 0.08:
        return min(2, len(features))
    return best_k



def _cluster(features: np.ndarray, k: int, iterations: int = 25) -> np.ndarray:
    if len(features) < k:
        return np.arange(len(features))

    rng = np.random.default_rng(42)
    indices = rng.choice(len(features), size=k, replace=False)
    centroids = features[indices].copy()

    labels = np.zeros(len(features), dtype=int)
    for _ in range(iterations):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for idx in range(k):
            members = features[labels == idx]
            if len(members) == 0:
                centroids[idx] = features[rng.integers(0, len(features))]
            else:
                centroids[idx] = members.mean(axis=0)

    return labels



def _silhouette(labels: np.ndarray, distances: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return -1.0

    values: list[float] = []
    for idx in range(len(labels)):
        same = labels == labels[idx]
        other_labels = [label for label in unique_labels if label != labels[idx]]

        a = float(distances[idx][same].mean()) if same.sum() > 1 else 0.0
        b = min(float(distances[idx][labels == label].mean()) for label in other_labels)
        denom = max(a, b)
        if denom == 0:
            values.append(0.0)
        else:
            values.append((b - a) / denom)

    return float(np.mean(values))



def _merge_segments(segments: list[DiarizationSegment], max_gap_s: float = 0.6) -> list[DiarizationSegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: (item.start_s, item.end_s))
    merged: list[DiarizationSegment] = [ordered[0]]

    for current in ordered[1:]:
        previous = merged[-1]
        if current.speaker_id == previous.speaker_id and current.start_s - previous.end_s <= max_gap_s:
            previous.end_s = current.end_s
            if previous.confidence is not None and current.confidence is not None:
                previous.confidence = (previous.confidence + current.confidence) / 2
            continue
        merged.append(current)

    return merged



def merge_transcript_segments(
    segments: list[dict[str, object]],
    max_gap_s: float | None = None,
) -> list[dict[str, object]]:
    if not segments:
        return []
    merged: list[dict[str, object]] = [segments[0].copy()]
    for segment in segments[1:]:
        current = merged[-1]
        same_speaker = segment["speaker_name"] == current["speaker_name"]
        if max_gap_s is None:
            should_merge = same_speaker
        else:
            small_gap = float(segment["start_s"]) - float(current["end_s"]) <= max_gap_s
            should_merge = same_speaker and small_gap
        if should_merge:
            current["end_s"] = segment["end_s"]
            current["text"] = (str(current["text"]).rstrip() + " " + str(segment["text"]).lstrip()).strip()
        else:
            merged.append(segment.copy())
    return merged
