from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

import numpy as np
import soundfile as sf

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
    max_speakers: int = 20,
    max_chunk_s: float = 18.0,
    backend: str = "auto",
    model_path: Path | None = None,
    embedding_model: str | None = None,
) -> DiarizationResult:
    normalized_backend = (backend or "auto").strip().lower()
    if normalized_backend not in {"auto", "heuristic", "pyannote", "embedding"}:
        normalized_backend = "auto"

    pyannote_error: str | None = None
    if normalized_backend in {"auto", "pyannote"}:
        pyannote_result, error = _diarize_with_pyannote(
            audio_path=audio_path,
            max_speakers=max_speakers,
            model_path=model_path,
        )
        if pyannote_result is not None:
            return pyannote_result
        pyannote_error = error
        if normalized_backend == "pyannote":
            return _diarize_heuristic(
                audio_path=audio_path,
                speech_regions=speech_regions,
                max_speakers=max_speakers,
                max_chunk_s=max_chunk_s,
                details=f"pyannote_forced_fallback: {error}",
            )

    embedding_error: str | None = None
    if normalized_backend in {"auto", "embedding"}:
        embedding_result, error = _diarize_with_embeddings(
            audio_path=audio_path,
            speech_regions=speech_regions,
            max_speakers=max_speakers,
            max_chunk_s=max_chunk_s,
            model_ref=embedding_model,
        )
        if embedding_result is not None:
            return embedding_result
        embedding_error = error
        if normalized_backend == "embedding":
            return _diarize_heuristic(
                audio_path=audio_path,
                speech_regions=speech_regions,
                max_speakers=max_speakers,
                max_chunk_s=max_chunk_s,
                details=f"embedding_forced_fallback: {error}",
            )

    details = None
    if normalized_backend == "auto":
        parts = []
        if pyannote_error:
            parts.append(f"pyannote={pyannote_error}")
        if embedding_error:
            parts.append(f"embedding={embedding_error}")
        details = "auto_fallback: " + ", ".join(parts) if parts else "auto_fallback"
    elif normalized_backend == "heuristic":
        details = "heuristic_backend_selected"

    return _diarize_heuristic(
        audio_path=audio_path,
        speech_regions=speech_regions,
        max_speakers=max_speakers,
        max_chunk_s=max_chunk_s,
        details=details,
    )


def _diarize_with_pyannote(
    audio_path: Path,
    max_speakers: int,
    model_path: Path | None,
) -> tuple[DiarizationResult | None, str | None]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:
        return None, f"pyannote_import_error:{exc.__class__.__name__}"

    pipeline_ref = _resolve_pyannote_pipeline_ref(model_path)
    if not pipeline_ref:
        return None, "pyannote_pipeline_not_configured"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    kwargs: dict[str, object] = {}
    if token:
        kwargs["use_auth_token"] = token

    try:
        pipeline = Pipeline.from_pretrained(pipeline_ref, **kwargs)
        try:
            pipeline.to(torch.device("cpu"))
        except Exception:
            # Some pipeline variants do not expose .to()
            pass

        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            waveform = np.ascontiguousarray(audio.T)
        else:
            waveform = np.ascontiguousarray(np.atleast_2d(audio))
        input_payload = {"waveform": torch.from_numpy(waveform), "sample_rate": int(sample_rate)}

        try:
            diarization = pipeline(input_payload, min_speakers=1, max_speakers=max_speakers)
        except TypeError:
            diarization = pipeline(input_payload)
    except Exception as exc:
        return None, f"pyannote_runtime_error:{exc.__class__.__name__}"

    raw_segments: list[DiarizationSegment] = []
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
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
            details="pyannote_pipeline",
        ),
        None,
    )


def _diarize_with_embeddings(
    audio_path: Path,
    speech_regions: list[SpeechRegion],
    max_speakers: int,
    max_chunk_s: float,
    model_ref: str | None,
) -> tuple[DiarizationResult | None, str | None]:
    try:
        import torch
        from pyannote.audio import Inference, Model
    except Exception as exc:
        return None, f"embedding_import_error:{exc.__class__.__name__}"

    resolved_model_ref = (model_ref or os.getenv("AUTOMOM_DIARIZATION_EMBEDDING_MODEL", "")).strip()
    if not resolved_model_ref:
        resolved_model_ref = "pyannote/wespeaker-voxceleb-resnet34-LM"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    kwargs: dict[str, object] = {}
    if token:
        kwargs["token"] = token

    try:
        embedding_model = Model.from_pretrained(resolved_model_ref, **kwargs)
        inference = Inference(embedding_model, window="whole")
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
    speaker_count = _estimate_speaker_count(feature_matrix, max_speakers=max_speakers)
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
            details=f"embedding_model:{resolved_model_ref}",
        ),
        None,
    )


def _resolve_pyannote_pipeline_ref(model_path: Path | None) -> str | None:
    explicit = os.getenv("AUTOMOM_DIARIZATION_PIPELINE", "").strip()
    if explicit:
        return explicit

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
    max_speakers: int,
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
    speaker_count = _estimate_speaker_count(features, max_speakers=max_speakers)
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

    max_k = min(max_speakers, max(2, len(features)))
    best_k = 1
    best_score = -1.0
    for k in range(2, max_k + 1):
        labels = _cluster(features, k)
        score = _silhouette(features, labels)
        if score > best_score:
            best_score = score
            best_k = k

    if best_score < 0.08:
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



def _silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) <= 1:
        return -1.0

    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    values = []
    for idx in range(len(features)):
        same = labels == labels[idx]
        other_labels = [label for label in set(labels.tolist()) if label != labels[idx]]

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
    max_gap_s: float = 1.0,
) -> list[dict[str, object]]:
    if not segments:
        return []
    merged: list[dict[str, object]] = [segments[0].copy()]
    for segment in segments[1:]:
        current = merged[-1]
        same_speaker = segment["speaker_name"] == current["speaker_name"]
        small_gap = float(segment["start_s"]) - float(current["end_s"]) <= max_gap_s
        if same_speaker and small_gap:
            current["end_s"] = segment["end_s"]
            current["text"] = (str(current["text"]).rstrip() + " " + str(segment["text"]).lstrip()).strip()
        else:
            merged.append(segment.copy())
    return merged
