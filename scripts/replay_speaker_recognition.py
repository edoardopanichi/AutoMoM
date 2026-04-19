from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline.orchestrator import PipelineOrchestrator


class _Runtime:
    api_config = None
    local_diarization_model_id = "pyannote-community-1"


class _Snippet:
    def __init__(self, *, speaker_id: str, path: Path, start_s: float, end_s: float) -> None:
        self.speaker_id = speaker_id
        self.path = path
        self.start_s = start_s
        self.end_s = end_s


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _segments_by_speaker(diarization_path: Path) -> dict[str, list[tuple[float, float]]]:
    rows = _load_json(diarization_path)
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        speaker_id = str(row.get("speaker_id") or row.get("speaker") or "").strip()
        if not speaker_id:
            continue
        grouped.setdefault(speaker_id, []).append((float(row["start_s"]), float(row["end_s"])))
    return grouped


def _snippets(snippets_path: Path) -> list[_Snippet]:
    rows = _load_json(snippets_path)
    snippets: list[_Snippet] = []
    for row in rows:
        snippets.append(
            _Snippet(
                speaker_id=str(row["speaker_id"]),
                path=Path(str(row["path"])),
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
            )
        )
    return snippets


def replay(job_dir: Path, expected_names: list[str]) -> dict[str, object]:
    orchestrator = PipelineOrchestrator()
    info = orchestrator._build_speaker_info(
        _Runtime(),
        job_dir.name,
        job_dir / "audio_normalized.wav",
        _segments_by_speaker(job_dir / "diarization.json"),
        _snippets(job_dir / "snippets.json"),
    )
    matched_names = sorted(
        {
            speaker.matched_profile.name
            for speaker in info.speakers
            if speaker.matched_profile is not None and speaker.matched_profile.status == "matched"
        }
    )
    missing = sorted(set(expected_names) - set(matched_names))
    return {
        "job_id": job_dir.name,
        "detected_speakers": info.detected_speakers,
        "review_cards": len(info.speakers),
        "matched_names": matched_names,
        "expected_names": expected_names,
        "missing": missing,
        "passed": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay saved-profile recognition for an existing job directory.")
    parser.add_argument("job_dir", type=Path)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        help="Expected recognized speaker name. Repeat for multiple names.",
    )
    args = parser.parse_args()

    result = replay(args.job_dir, args.expect)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
