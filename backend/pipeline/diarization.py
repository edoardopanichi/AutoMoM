from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import copy
import warnings

import numpy as np
import soundfile as sf

from backend.pipeline.compute import resolve_torch_device
from backend.pipeline.vad import SpeechRegion


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
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            model_path=model_path,
            pipeline_path=pipeline_path,
            compute_device=compute_device,
            cuda_device_id=cuda_device_id,
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
    min_speakers: int | None,
    max_speakers: int | None,
    model_path: Path | None,
    compute_device: str,
    cuda_device_id: int,
    pipeline_path: str | None = None,
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

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    kwargs: dict[str, object] = {}
    if token:
        kwargs["use_auth_token"] = token

    try:
        pipeline = copy.deepcopy(_load_pyannote_pipeline(pipeline_ref, token))
        target_device = resolve_torch_device(compute_device, cuda_device_id)

        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            waveform = np.ascontiguousarray(audio.T)
        else:
            waveform = np.ascontiguousarray(np.atleast_2d(audio))
        input_payload = {"waveform": torch.from_numpy(waveform), "sample_rate": int(sample_rate)}

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
