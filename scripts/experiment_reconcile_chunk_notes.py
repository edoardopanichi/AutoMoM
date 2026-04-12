#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS
from backend.pipeline.formatter import _estimate_tokens, _extract_model_text, validate_markdown_output
from backend.pipeline.template_manager import TEMPLATE_MANAGER


NOTE_SECTIONS = {
    "topics": "Topics",
    "decisions": "Decisions and Conclusions",
    "actions": "Action Items",
    "open": "Open Questions",
    "risks": "Risks",
}

AGENDA_GROUPS = [
    {
        "label": "Agenda and Qualico development agreement",
        "keywords": ("agenda", "qualico", "development agreement", "david joplin", "signed agreement"),
        "min_score": 2,
    },
    {
        "label": "Conditional use 2601: Winnipeg Dig and Demolition at 216 Jean-Marc",
        "keywords": ("2601", "dig and demolition", "216 jean-marc", "chicago auto", "derelict vehicles"),
        "min_score": 1,
    },
    {
        "label": "Conditional use 2602: second dwelling near aggregate operations",
        "keywords": ("2602", "second dwelling", "aggregate", "25022 oakwood", "private water well"),
        "min_score": 1,
    },
    {
        "label": "Conditional use 25-05: Ridgeland Colony livestock expansion",
        "keywords": ("ridgeland", "livestock", "broiler", "laying flock", "animal units", "richland road"),
        "min_score": 2,
    },
    {
        "label": "TRC, water rights, domestic water use, and provincial review",
        "keywords": ("trc", "water rights", "domestic water", "water use", "license", "aquifer", "meter"),
        "min_score": 2,
    },
    {
        "label": "Manure management, biosecurity, avian flu, and Cooks Creek concerns",
        "keywords": ("manure", "avian", "h5n1", "biosecurity", "cooks creek", "lagoon", "food scraps"),
        "min_score": 2,
    },
    {
        "label": "Subdivision 4189-25-7850: industrial property at Springfield Road and Oxford Street",
        "keywords": ("4189257850", "springfield road", "oxford street", "auto recycling", "auto wrecking"),
        "min_score": 2,
    },
    {
        "label": "Subdivision 4189-25-7853: farmstead yard site for Oakwood Dairy Farms",
        "keywords": ("4189-25-7853", "oakwood dairy", "farmstead", "194.2 acres", "matthew braun", "nicole duma"),
        "min_score": 1,
    },
]

AGENDA_RANGES = [
    ("Agenda and Qualico development agreement", range(1, 2)),
    ("Conditional use 2601: Winnipeg Dig and Demolition at 216 Jean-Marc", range(2, 3)),
    ("Conditional use 2602: second dwelling near aggregate operations", range(3, 5)),
    ("Conditional use 25-05: Ridgeland Colony livestock expansion, TRC, water, manure, and biosecurity", range(5, 17)),
    ("Subdivision 4189-25-7850: industrial property at Springfield Road and Oxford Street", range(17, 18)),
    ("Subdivision 4189-25-7853: farmstead yard site for Oakwood Dairy Farms", range(18, 19)),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile chunk-level notes by agenda item, then stitch an xlab MoM.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--notes-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix/map_reduce_formatter_v2")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-host", default="")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grouping", choices=("keyword", "range"), default="range")
    return parser.parse_args()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


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


def _infer_model(job_dir: Path, explicit_model: str) -> str:
    if explicit_model.strip():
        return explicit_model.strip()
    summary_path = job_dir / "job_summary.json"
    if summary_path.exists():
        payload = _load_json(summary_path)
        execution = payload.get("execution") if isinstance(payload, dict) else None
        if isinstance(execution, dict):
            formatter = execution.get("formatter")
            if isinstance(formatter, dict) and formatter.get("model"):
                return str(formatter["model"]).strip()
    return SETTINGS.formatter_ollama_model


def _load_speakers(job_dir: Path) -> list[str]:
    payload = _load_json(job_dir / "transcript.json")
    speakers = payload.get("speakers") if isinstance(payload, dict) else []
    return [str(item) for item in speakers if str(item).strip()]


def _clean_text(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _call_ollama(
    *,
    host: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    num_ctx: int,
    num_predict: int,
    temperature: float,
    timeout_s: int,
) -> tuple[str, dict[str, object]]:
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature},
    }
    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"Ollama HTTP error: {body}") from exc
    raw = json.loads(body)
    return _clean_text(_extract_model_text(str(raw.get("response", "")), "")), raw


def _match_score(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _parse_note_file(path: Path) -> tuple[int, str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"##\s+Transcript chunk\s+(\d+)/(\d+)", text)
    return (int(match.group(1)) if match else 0), text


def _group_notes(note_files: list[Path]) -> list[dict[str, object]]:
    chunks = [_parse_note_file(path) for path in note_files]
    chunks_by_index = {index: text for index, text in chunks}
    grouped: list[dict[str, object]] = []
    for group in AGENDA_GROUPS:
        keywords = tuple(group["keywords"])
        min_score = int(group.get("min_score", 1))
        matches = [(index, text) for index, text in chunks if _match_score(text, keywords) >= min_score]
        grouped.append(
            {
                "label": group["label"],
                "keywords": keywords,
                "min_score": min_score,
                "chunks": matches,
                "notes": "\n\n".join(text for _index, text in matches),
            }
        )
    return grouped


def _group_notes_by_range(note_files: list[Path]) -> list[dict[str, object]]:
    chunks = [_parse_note_file(path) for path in note_files]
    chunks_by_index = {index: text for index, text in chunks}
    grouped: list[dict[str, object]] = []
    for label, chunk_range in AGENDA_RANGES:
        matches = [(index, chunks_by_index[index]) for index in chunk_range if index in chunks_by_index]
        grouped.append(
            {
                "label": label,
                "chunks": matches,
                "notes": "\n\n".join(text for _index, text in matches),
            }
        )
    return grouped


def _reconcile_prompts(title: str, speakers: list[str], group_label: str, notes: str) -> tuple[str, str]:
    system = (
        "You reconcile chronological chunk-level meeting notes for one agenda item.\n"
        "Return concise Markdown only. Do not write the final MoM. Do not invent facts.\n"
        "Resolve contradictions explicitly:\n"
        "- Later explicit decisions override earlier proposals, questions, or tentative statements.\n"
        "- Put final decisions only under Decisions and Conclusions.\n"
        "- Put unresolved conflicts, missing facts, or uncertainty under Open Questions.\n"
        "- If an earlier proposal was later changed, either omit it or mention the final state only.\n"
        "- Preserve important conditions and action owners when supported.\n\n"
        "Use these headings exactly:\n"
        "### Topics\n### Decisions and Conclusions\n### Action Items\n### Open Questions\n### Risks\n"
        "If a heading has no content, write None."
    )
    user = (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n"
        f"Agenda item to reconcile: {group_label}\n\n"
        "Chronological chunk notes for this agenda item:\n"
        f"{notes}\n"
    )
    return system, user


def _parse_reconciled_notes(text: str) -> dict[str, list[str]]:
    parsed = {key: [] for key in NOTE_SECTIONS}
    current_key: str | None = None
    for raw_line in text.splitlines():
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
            parsed[current_key].append(re.sub(r"\s+", " ", item))
    return parsed


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", item).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _render_bullets(items: list[str]) -> str:
    if not items:
        return "None"
    return "\n".join(f"- {item}" for item in items)


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


def _stitch_mom(job_dir: Path, title: str, speakers: list[str], group_outputs: list[tuple[str, str]]) -> str:
    overview: list[str] = []
    decisions: list[str] = []
    actions: list[str] = []
    open_points: list[str] = []
    risks: list[str] = []
    for label, text in group_outputs:
        parsed = _parse_reconciled_notes(text)
        topic = parsed["topics"][0] if parsed["topics"] else label
        overview.append(f"{label}: {topic}")
        decisions.extend(parsed["decisions"])
        actions.extend(parsed["actions"])
        open_points.extend(parsed["open"])
        risks.extend(parsed["risks"])

    participants = _build_participants(job_dir, speakers)
    return "\n\n".join(
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


def main() -> int:
    _load_dotenv(ROOT_DIR / ".env")
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    notes_dir = args.notes_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else job_dir / f"reconcile_chunk_notes_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    note_files = sorted(notes_dir.glob("map_*_notes.md"))
    if not note_files:
        raise FileNotFoundError(f"No map note files found in {notes_dir}")
    title = _infer_title(job_dir, args.title)
    model = _infer_model(job_dir, args.model)
    host = (args.ollama_host or SETTINGS.ollama_host).rstrip("/")
    speakers = _load_speakers(job_dir)
    grouped_notes = _group_notes_by_range(note_files) if args.grouping == "range" else _group_notes(note_files)

    records: list[dict[str, object]] = []
    group_outputs: list[tuple[str, str]] = []
    for index, group in enumerate(grouped_notes, start=1):
        label = str(group["label"])
        notes = str(group["notes"])
        chunk_numbers = [chunk_index for chunk_index, _text in group["chunks"]]  # type: ignore[index]
        if not notes.strip():
            output = "### Topics\nNone\n\n### Decisions and Conclusions\nNone\n\n### Action Items\nNone\n\n### Open Questions\nNone\n\n### Risks\nNone"
            group_outputs.append((label, output))
            continue
        system_prompt, user_prompt = _reconcile_prompts(title, speakers, label, notes)
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        prompt_path = output_dir / f"reconcile_{index:02d}_{slug}_prompt.txt"
        raw_path = output_dir / f"reconcile_{index:02d}_{slug}_raw.json"
        notes_path = output_dir / f"reconcile_{index:02d}_{slug}.md"
        prompt_path.write_text(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", encoding="utf-8")
        print(f"[reconcile {index}/{len(grouped_notes)}] {label} chunks={chunk_numbers}", flush=True)
        started = time.monotonic()
        output, raw = _call_ollama(
            host=host,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        seconds = time.monotonic() - started
        raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        notes_path.write_text(output + "\n", encoding="utf-8")
        group_outputs.append((label, output))
        records.append(
            {
                "label": label,
                "chunks": chunk_numbers,
                "seconds": seconds,
                "prompt_tokens_estimate": _estimate_tokens(system_prompt + "\n\n" + user_prompt),
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
                "prompt_eval_duration": raw.get("prompt_eval_duration"),
                "eval_duration": raw.get("eval_duration"),
            }
        )

    mom = _stitch_mom(job_dir, title, speakers, group_outputs)
    template = TEMPLATE_MANAGER.load(args.template_id)
    evaluation = {
        "validation": validate_markdown_output(mom, template.sections),
        "estimated_tokens": _estimate_tokens(mom),
        "groups": len(group_outputs),
        "records": records,
    }
    (output_dir / "mom_reconciled.md").write_text(mom + "\n", encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(json.dumps({"validation": evaluation["validation"], "estimated_tokens": evaluation["estimated_tokens"]}, indent=2))
    return 0 if bool(evaluation["validation"]["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
