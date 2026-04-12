#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS
from backend.pipeline.formatter import _estimate_tokens, validate_markdown_output
from backend.pipeline.template_manager import TEMPLATE_MANAGER


SECTION_MAP = {
    "topics": "Topics",
    "decisions": "Decisions and Conclusions",
    "actions": "Action Items",
    "open": "Open Questions",
    "risks": "Risks",
}

SUMMARY_TOPICS = [
    (
        "Agenda and Qualico development agreement",
        ("agenda", "qualico", "development agreement"),
    ),
    (
        "Conditional use 2601: Winnipeg Dig and Demolition at 216 Jean-Marc",
        ("2601", "dig and demolition", "216 jean-marc", "chicago auto"),
    ),
    (
        "Conditional use 2602: second dwelling near aggregate operations",
        ("2602", "second dwelling", "aggregate", "oakwood road"),
    ),
    (
        "Conditional use 25-05: Ridgeland Colony livestock expansion",
        ("ridgeland", "livestock", "broiler", "laying flock", "animal units", "richland road"),
    ),
    (
        "TRC, water rights, domestic water use, and provincial review",
        ("trc", "water", "domestic", "license", "aquifer"),
    ),
    (
        "Manure management, biosecurity, avian flu, and Cooks Creek concerns",
        ("manure", "avian", "h5n1", "biosecurity", "cooks creek", "lagoons"),
    ),
    (
        "Subdivision 4189-25-7850: industrial property at Springfield Road and Oxford Street",
        ("4189257850", "springfield road", "oxford street", "auto recycling"),
    ),
    (
        "Subdivision 4189-25-7853: farmstead yard site for Oakwood Dairy Farms",
        ("4189-25-7853", "oakwood dairy", "farmstead", "194.2 acres"),
    ),
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

PROCEDURAL_ACTION_MARKERS = (
    "mover and seconder",
    "provide name and address",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministically stitch map notes into the xlab MoM template.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--notes-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix/map_reduce_formatter_v2")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-overview", type=int, default=18)
    parser.add_argument("--max-actions", type=int, default=35)
    parser.add_argument("--max-decisions", type=int, default=35)
    parser.add_argument("--max-open", type=int, default=30)
    parser.add_argument("--max-risks", type=int, default=30)
    return parser.parse_args()


def _load_json(path: Path):
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
    parsed = {key: [] for key in SECTION_MAP}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"^###\s+(.+?)\s*$", line)
        if heading:
            normalized_heading = heading.group(1).strip().lower()
            current_key = None
            for key, label in SECTION_MAP.items():
                if normalized_heading == label.lower():
                    current_key = key
                    break
            continue
        if current_key is None:
            continue
        item = re.sub(r"^[-*]\s+", "", line).strip()
        item = item.rstrip()
        if item and item.lower() != "none":
            parsed[current_key].append(item)
    return parsed


def _clean_item(item: str) -> str:
    item = re.sub(r"\s+", " ", item).strip()
    item = item.rstrip(" \t")
    return item


def _dedupe(items: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = _clean_item(item)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if len(result) >= limit:
            break
    return result


def _contains_any(item: str, needles: tuple[str, ...]) -> bool:
    lowered = item.lower()
    return any(needle in lowered for needle in needles)


def _best_keyword_match(items: list[str], keywords: tuple[str, ...]) -> str:
    best_item = ""
    best_score = 0
    for item in items:
        lowered = item.lower()
        matches = [keyword for keyword in keywords if keyword in lowered]
        if not matches:
            continue
        specificity = sum(len(keyword) for keyword in matches)
        score = len(matches) * 1000 + specificity
        if score > best_score:
            best_score = score
            best_item = item
    return best_item


def _build_overview(collected: dict[str, list[str]], *, limit: int) -> list[str]:
    candidates = collected["topics"] + collected["decisions"]
    overview: list[str] = []
    for label, keywords in SUMMARY_TOPICS:
        matched = _best_keyword_match(candidates, keywords)
        if matched:
            overview.append(f"{label}: {matched}")
        else:
            overview.append(label)
    if len(overview) < limit:
        overview.extend(collected["topics"])
    return _dedupe(overview, limit=limit)


def _coverage_select(items: list[str], *, limit: int) -> list[str]:
    selected: list[str] = []
    for _label, keywords in SUMMARY_TOPICS:
        matched = _best_keyword_match(items, keywords)
        if matched:
            selected.append(matched)
    selected.extend(items)
    return _dedupe(selected, limit=limit)


def _filter_actions(items: list[str], *, limit: int) -> list[str]:
    filtered: list[str] = []
    for item in items:
        cleaned = _clean_item(item)
        lowered = cleaned.lower()
        if lowered.startswith(WEAK_ACTION_PREFIXES):
            continue
        if any(marker in lowered for marker in PROCEDURAL_ACTION_MARKERS):
            continue
        filtered.append(cleaned)
    return _coverage_select(filtered, limit=limit)


def _filter_decisions(items: list[str], *, limit: int) -> list[str]:
    filtered: list[str] = []
    for item in items:
        lowered = item.lower()
        if "no formal decision made" in lowered:
            continue
        if "director of medan" in lowered:
            continue
        filtered.append(item)
    return _coverage_select(filtered, limit=limit)


def _build_participants(job_dir: Path, speakers: list[str]) -> list[str]:
    transcript_path = job_dir / "transcript.json"
    raw_transcript = transcript_path.read_text(encoding="utf-8").lower() if transcript_path.exists() else ""
    raw_notes = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in sorted((job_dir / "map_reduce_formatter_v2").glob("map_*_notes.md"))
    ).lower()
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
    text = raw_transcript + "\n" + raw_notes
    participants = [name for name, needles in candidates if any(needle in text for needle in needles)]
    aliases = [speaker for speaker in speakers if not speaker.startswith("SPEAKER_")]
    if aliases:
        participants.append("Auto-assigned speaker aliases in transcript: " + ", ".join(aliases))
    return _dedupe(participants, limit=20)


def _render_bullets(items: list[str]) -> str:
    if not items:
        return "None"
    return "\n".join(f"- {item}" for item in items)


def main() -> int:
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    notes_dir = args.notes_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else notes_dir / "stitched_from_map_notes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = {key: [] for key in SECTION_MAP}
    note_files = sorted(notes_dir.glob("map_*_notes.md"))
    if not note_files:
        raise FileNotFoundError(f"No map note files found in {notes_dir}")
    for path in note_files:
        parsed = _parse_note_file(path)
        for key, items in parsed.items():
            collected[key].extend(items)

    speakers = _load_speakers(job_dir)
    participants = _build_participants(job_dir, speakers)

    overview = _build_overview(collected, limit=args.max_overview)
    actions = _filter_actions(collected["actions"], limit=args.max_actions)
    decisions = _filter_decisions(collected["decisions"], limit=args.max_decisions)
    open_points = _coverage_select(collected["open"], limit=args.max_open)
    risks = _coverage_select(collected["risks"], limit=args.max_risks)
    title = _infer_title(job_dir, args.title)

    mom = "\n\n".join(
        [
            f"### Title: {title}",
            f"#### Participants:\n{_render_bullets(participants)}",
            f"#### Concise Overview:\n{_render_bullets(overview)}",
            f"#### TODO's:\n{_render_bullets(actions)}",
            f"#### CONCLUSIONS:\n{_render_bullets(decisions)}",
            f"#### DECISION/OPEN POINTS:\n{_render_bullets(open_points)}",
            f"#### RISKS:\n{_render_bullets(risks)}",
        ]
    )
    template = TEMPLATE_MANAGER.load(args.template_id)
    evaluation = {
        "validation": validate_markdown_output(mom, template.sections),
        "estimated_tokens": _estimate_tokens(mom),
        "note_files": len(note_files),
        "counts": {
            "overview": len(overview),
            "actions": len(actions),
            "decisions": len(decisions),
            "open_points": len(open_points),
            "risks": len(risks),
        },
    }
    (output_dir / "mom_stitched.md").write_text(mom + "\n", encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(json.dumps(evaluation, indent=2))
    return 0 if bool(evaluation["validation"]["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
