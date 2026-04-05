from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.pipeline.audio import extract_segment
from backend.pipeline.diarization import DiarizationSegment


@dataclass
class Snippet:
    speaker_id: str
    path: Path
    start_s: float
    end_s: float



def pick_snippet_ranges(
    segments: list[DiarizationSegment],
    per_speaker: int = 3,
    min_len_s: float = 3.0,
    max_len_s: float = 8.0,
    *,
    audio_path: Path | None = None,
    min_gap_s: float = 4.0,
) -> dict[str, list[tuple[float, float]]]:
    by_speaker: dict[str, list[DiarizationSegment]] = defaultdict(list)
    for segment in segments:
        by_speaker[segment.speaker_id].append(segment)

    audio_data: np.ndarray | None = None
    sample_rate: int | None = None
    if audio_path is not None and audio_path.exists():
        loaded_audio, loaded_rate = sf.read(str(audio_path), always_2d=False)
        audio_data = np.asarray(loaded_audio, dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        sample_rate = int(loaded_rate)

    selected: dict[str, list[tuple[float, float]]] = {}
    for speaker_id, speaker_segments in by_speaker.items():
        scored = sorted(
            (_score_candidate(item, min_len_s, max_len_s, audio_data, sample_rate) for item in speaker_segments),
            key=lambda item: item["score"],
            reverse=True,
        )
        ranges: list[tuple[float, float]] = []
        for candidate in scored:
            if len(ranges) >= per_speaker:
                break
            if candidate["duration"] < min_len_s:
                continue
            start_s = float(candidate["start_s"])
            end_s = float(candidate["end_s"])
            if any(min(abs(start_s - prev_start), abs(end_s - prev_end)) < min_gap_s for prev_start, prev_end in ranges):
                continue
            ranges.append((start_s, end_s))
        if not ranges and scored:
            best = scored[0]
            ranges.append((float(best["start_s"]), float(best["end_s"])))
        selected[speaker_id] = ranges

    return selected


def _score_candidate(
    segment: DiarizationSegment,
    min_len_s: float,
    max_len_s: float,
    audio_data: np.ndarray | None,
    sample_rate: int | None,
) -> dict[str, float]:
    duration = max(0.0, float(segment.end_s) - float(segment.start_s))
    target_len = min(max_len_s, max(min_len_s, 5.0))
    if duration > max_len_s:
        midpoint = (float(segment.start_s) + float(segment.end_s)) / 2.0
        start_s = max(float(segment.start_s), midpoint - (max_len_s / 2.0))
        end_s = min(float(segment.end_s), start_s + max_len_s)
        start_s = max(float(segment.start_s), end_s - max_len_s)
    else:
        start_s = float(segment.start_s)
        end_s = float(segment.end_s)

    energy_score = 0.0
    if audio_data is not None and sample_rate is not None and audio_data.size:
        start_idx = max(0, int(start_s * sample_rate))
        end_idx = max(start_idx, int(end_s * sample_rate))
        window = audio_data[start_idx:end_idx]
        if window.size:
            rms = float(np.sqrt(np.mean(np.square(window))) + 1e-8)
            energy_score = min(1.0, max(0.0, rms * 8.0))

    confidence = 0.5 if segment.confidence is None else float(segment.confidence)
    duration_score = min(1.0, max(0.0, min(duration, target_len) / target_len))
    score = (duration_score * 0.45) + (energy_score * 0.35) + (confidence * 0.20)
    return {
        "score": score,
        "duration": duration,
        "start_s": start_s,
        "end_s": end_s,
    }



def extract_snippets(
    source_audio_path: Path,
    output_dir: Path,
    snippet_ranges: dict[str, list[tuple[float, float]]],
    ffmpeg_bin: str = "ffmpeg",
) -> list[Snippet]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snippets: list[Snippet] = []
    for speaker_id, ranges in snippet_ranges.items():
        for index, (start_s, end_s) in enumerate(ranges, start=1):
            snippet_path = output_dir / f"{speaker_id}_{index}.wav"
            extract_segment(source_audio_path, snippet_path, start_s=start_s, end_s=end_s, ffmpeg_bin=ffmpeg_bin)
            snippets.append(Snippet(speaker_id=speaker_id, path=snippet_path, start_s=start_s, end_s=end_s))
    return snippets
