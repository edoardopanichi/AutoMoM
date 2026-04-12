#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS
from backend.pipeline.formatter import _estimate_tokens, validate_markdown_output
from backend.pipeline.template_manager import TEMPLATE_MANAGER


NOTE_SECTIONS = {
    "topics": "Topics",
    "decisions": "Decisions and Conclusions",
    "actions": "Action Items",
    "open": "Open Questions",
    "risks": "Risks",
}

AGENDA_RANGES = [
    ("Agenda and Qualico development agreement", range(1, 2)),
    ("Conditional use 2601: Winnipeg Dig and Demolition at 216 Jean-Marc", range(2, 3)),
    ("Conditional use 2602: second dwelling near aggregate operations", range(3, 5)),
    ("Conditional use 25-05: Ridgeland Colony livestock expansion, TRC, water, manure, and biosecurity", range(5, 17)),
    ("Subdivision 4189-25-7850: industrial property at Springfield Road and Oxford Street", range(17, 18)),
    ("Subdivision 4189-25-7853: farmstead yard site for Oakwood Dairy Farms", range(18, 19)),
]

WEAK_ACTION_PREFIXES = (
    "confirm whether",
    "clarify whether",
    "whether ",
    "how ",
    "what ",
    "why ",
    "does ",
    "is there ",
)

NOISE_MARKERS = (
    "director of medan",
    "mover and seconder assigned",
    "speaker_",
    "note:",
    "contradiction resolved",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministically stitch agenda-range chunk notes into xlab MoM.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--notes-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix/map_reduce_formatter_v2")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_title(job_dir: Path, explicit_title: str) -> str:
    if explicit_title.strip():
        return explicit_title.strip()
    runtime_path = job_dir / "job_runtime.json"
    if runtime_path.exists():
        title = str((_load_json(runtime_path) or {}).get("title") or "").strip()
        if title:
            return title
    return job_dir.name


def _load_speakers(job_dir: Path) -> list[str]:
    payload = _load_json(job_dir / "transcript.json")
    speakers = payload.get("speakers") if isinstance(payload, dict) else []
    return [str(item) for item in speakers if str(item).strip()]


def _parse_note_file(path: Path) -> dict[str, list[str]]:
    parsed = {key: [] for key in NOTE_SECTIONS}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"^###\s+(.+?)\s*$", line)
        if heading:
            normalized = heading.group(1).strip().lower()
            current_key = None
            for key, label in NOTE_SECTIONS.items():
                if normalized == label.lower():
                    current_key = key
                    break
            continue
        if current_key is None:
            continue
        item = re.sub(r"^[-*]\s+", "", line).strip()
        if item and item.lower() != "none":
            parsed[current_key].append(_clean(item))
    return parsed


def _clean(item: str) -> str:
    return re.sub(r"\s+", " ", item).strip().rstrip(".")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = _clean(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _keep(item: str) -> bool:
    lowered = item.lower()
    return not any(marker in lowered for marker in NOISE_MARKERS)


def _filter_actions(items: list[str]) -> list[str]:
    filtered = []
    for item in items:
        lowered = item.lower()
        if lowered.startswith(WEAK_ACTION_PREFIXES):
            continue
        if not _keep(item):
            continue
        filtered.append(item)
    return _dedupe(filtered)


def _filter_decisions(label: str, items: list[str]) -> list[str]:
    has_later_ridgeland_resolution = any("public hearing is officially closed with resolution passed" in item.lower() for item in items)
    filtered = []
    for item in items:
        lowered = item.lower()
        if not _keep(item):
            continue
        if has_later_ridgeland_resolution and ("no formal decision" in lowered or "no decision has been made" in lowered or "defer" in lowered):
            continue
        if "2602" in label and "eight conditions" in lowered:
            continue
        filtered.append(item)
    return _dedupe(filtered)


def _filter_open(items: list[str]) -> list[str]:
    return _dedupe([item for item in items if _keep(item) and not item.lower().startswith("none")])


def _filter_risks(items: list[str]) -> list[str]:
    return _dedupe([item for item in items if _keep(item) and not item.lower().startswith("none")])


def _first_matching(items: list[str], patterns: tuple[str, ...]) -> str:
    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    for item in items:
        lowered = item.lower()
        if any(pattern in lowered for pattern in lowered_patterns):
            return item
    return items[0] if items else "None"


def _build_participants(job_dir: Path, speakers: list[str]) -> list[str]:
    transcript = (job_dir / "transcript.json").read_text(encoding="utf-8").lower()
    candidates = [
        ("Deputy Mayor Glenfield", ("glenfield",)),
        ("Mayor Terry", ("mayor terry",)),
        ("Councillor Kaczynski/Kuzinski", ("kaczynski", "kuzinski")),
        ("Councillor Miller", ("councilor miller", "councillor miller")),
        ("Councillor Warren", ("councilor warren", "councillor warren")),
        ("Mr. Draper", ("mr. draper",)),
        ("Ms. Holly Irvin-Nott", ("holly irvin", "irvin-nott")),
        ("Mr. Wohlman/Wolman", ("wohlman", "wolman")),
        ("Jason Tucker", ("jason tucker",)),
        ("Matthew Braun and Nicole Duma, Oakwood Dairy Farms Limited", ("matthew braun", "nicole duma", "oakwood dairy")),
        ("Applicants, municipal staff, provincial reviewers, and public delegations", ("applicant", "public hearing", "technical review")),
    ]
    participants = [name for name, needles in candidates if any(needle in transcript for needle in needles)]
    aliases = [speaker for speaker in speakers if not speaker.startswith("SPEAKER_")]
    if aliases:
        participants.append("Auto-assigned speaker aliases in transcript: " + ", ".join(aliases))
    return _dedupe(participants)


def _render_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "None"


def main() -> int:
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    notes_dir = args.notes_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else job_dir / "agenda_range_stitch"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    note_files = {int(match.group(1)): path for path in notes_dir.glob("map_*_notes.md") if (match := re.search(r"map_(\d+)_notes\.md$", path.name))}
    title = _infer_title(job_dir, args.title)
    speakers = _load_speakers(job_dir)
    participants = _build_participants(job_dir, speakers)

    overview: list[str] = []
    decisions: list[str] = []
    actions: list[str] = []
    open_points: list[str] = []
    risks: list[str] = []
    group_summaries: list[dict[str, object]] = []

    for label, chunk_range in AGENDA_RANGES:
        parsed = {key: [] for key in NOTE_SECTIONS}
        chunks = []
        for index in chunk_range:
            path = note_files.get(index)
            if path is None:
                continue
            chunks.append(index)
            chunk_parsed = _parse_note_file(path)
            for key, values in chunk_parsed.items():
                parsed[key].extend(values)

        if "2602" in label:
            topic = _first_matching(parsed["topics"], ("2602", "second dwelling"))
        elif "25-05" in label:
            topic = _first_matching(parsed["topics"], ("ridgeland", "livestock", "laying flock"))
        elif "7850" in label:
            topic = _first_matching(parsed["topics"], ("4189257850", "springfield road", "oxford street"))
        elif "7853" in label:
            topic = _first_matching(parsed["topics"], ("4189", "farmstead", "oakwood dairy"))
        else:
            topic = parsed["topics"][0] if parsed["topics"] else label
        overview.append(f"{label}: {topic}")

        group_decisions = _filter_decisions(label, parsed["decisions"])
        group_actions = _filter_actions(parsed["actions"])
        group_open = _filter_open(parsed["open"])
        group_risks = _filter_risks(parsed["risks"])
        decisions.extend(group_decisions)
        actions.extend(group_actions)
        open_points.extend(group_open)
        risks.extend(group_risks)
        group_summaries.append(
            {
                "label": label,
                "chunks": chunks,
                "decisions": len(group_decisions),
                "actions": len(group_actions),
                "open_points": len(group_open),
                "risks": len(group_risks),
            }
        )

    mom = "\n\n".join(
        [
            f"### Title: {title}",
            f"#### Participants:\n{_render_bullets(participants)}",
            f"#### Concise Overview:\n{_render_bullets(_dedupe(overview))}",
            f"#### TODO's:\n{_render_bullets(_dedupe(actions))}",
            f"#### CONCLUSIONS:\n{_render_bullets(_dedupe(decisions))}",
            f"#### DECISION/OPEN POINTS:\n{_render_bullets(_dedupe(open_points))}",
            f"#### RISKS:\n{_render_bullets(_dedupe(risks))}",
        ]
    )
    template = TEMPLATE_MANAGER.load(args.template_id)
    evaluation = {
        "validation": validate_markdown_output(mom, template.sections),
        "estimated_tokens": _estimate_tokens(mom),
        "groups": group_summaries,
    }
    (output_dir / "mom_agenda_range_stitched.md").write_text(mom + "\n", encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(json.dumps({"validation": evaluation["validation"], "estimated_tokens": evaluation["estimated_tokens"]}, indent=2))
    return 0 if bool(evaluation["validation"]["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
