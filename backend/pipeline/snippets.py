from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
) -> dict[str, list[tuple[float, float]]]:
    by_speaker: dict[str, list[DiarizationSegment]] = defaultdict(list)
    for segment in segments:
        by_speaker[segment.speaker_id].append(segment)

    selected: dict[str, list[tuple[float, float]]] = {}
    for speaker_id, speaker_segments in by_speaker.items():
        ordered = sorted(speaker_segments, key=lambda item: (item.end_s - item.start_s), reverse=True)
        ranges: list[tuple[float, float]] = []
        for segment in ordered:
            if len(ranges) >= per_speaker:
                break
            duration = segment.end_s - segment.start_s
            if duration < min_len_s:
                continue
            end = min(segment.end_s, segment.start_s + max_len_s)
            ranges.append((segment.start_s, end))
        if not ranges and ordered:
            ranges.append((ordered[0].start_s, min(ordered[0].end_s, ordered[0].start_s + max_len_s)))
        selected[speaker_id] = ranges

    return selected



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
