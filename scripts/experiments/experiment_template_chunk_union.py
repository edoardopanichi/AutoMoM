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
from backend.app.schemas import TemplateSection
from backend.pipeline.formatter import _estimate_tokens, _extract_model_text, validate_markdown_output
from backend.pipeline.template_manager import TEMPLATE_MANAGER


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Template-section chunk union and polish experiment.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-host", default="")
    parser.add_argument("--chunk-token-target", type=int, default=3600)
    parser.add_argument("--first-chunk-token-target", type=int, default=0)
    parser.add_argument("--chunk-num-ctx", type=int, default=8192)
    parser.add_argument("--polish-num-ctx", type=int, default=16384)
    parser.add_argument("--chunk-num-predict", type=int, default=900)
    parser.add_argument("--polish-num-predict", type=int, default=2200)
    parser.add_argument("--chunk-user-mode", choices=("constrained", "minimal"), default="constrained")
    parser.add_argument("--polish-mode", choices=("all", "section"), default="all")
    parser.add_argument("--reuse-chunks", action="store_true")
    parser.add_argument("--rerun-chunks", default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def _parse_index_set(raw: str) -> set[int]:
    result: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        result.add(int(item))
    return result


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


def _load_transcript(job_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    payload = _load_json(job_dir / "transcript.json")
    segments = []
    for index, raw in enumerate(payload.get("segments") or [], start=1):
        if not isinstance(raw, dict):
            continue
        speaker_id = str(raw.get("speaker_id") or raw.get("speaker") or f"SPEAKER_{index:04d}")
        speaker_name = str(raw.get("speaker_name") or raw.get("speaker") or speaker_id)
        text = str(raw.get("text") or raw.get("transcript") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "start_s": float(raw.get("start_s") or raw.get("start") or 0.0),
                "end_s": float(raw.get("end_s") or raw.get("end") or 0.0),
                "text": text,
            }
        )
    speakers = [str(item) for item in payload.get("speakers") or [] if str(item).strip()]
    return segments, speakers


def _chunk_transcript(segments: list[dict[str, object]], token_target: int) -> list[list[dict[str, object]]]:
    chunks: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_tokens = 0
    for segment in segments:
        tokens = _estimate_tokens(str(segment.get("text") or "")) + 20
        if current and current_tokens + tokens > token_target:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(segment)
        current_tokens += tokens
    if current:
        chunks.append(current)
    return chunks


def _chunk_transcript_with_first_target(
    segments: list[dict[str, object]],
    *,
    first_token_target: int,
    token_target: int,
) -> list[list[dict[str, object]]]:
    if first_token_target <= 0 or first_token_target >= token_target or not segments:
        return _chunk_transcript(segments, token_target)
    first_chunk: list[dict[str, object]] = []
    first_tokens = 0
    split_index = 0
    for index, segment in enumerate(segments):
        tokens = _estimate_tokens(str(segment.get("text") or "")) + 20
        if first_chunk and first_tokens + tokens > first_token_target:
            split_index = index
            break
        first_chunk.append(segment)
        first_tokens += tokens
    else:
        return [first_chunk]
    return [first_chunk] + _chunk_transcript(segments[split_index:], token_target)


def _render_segments(segments: list[dict[str, object]]) -> str:
    lines = []
    for item in segments:
        text = str(item.get("text") or "").strip()
        if text:
            lines.append(f"[{float(item.get('start_s') or 0.0):08.2f}s] {item['speaker_name']}: {text}")
    return "\n".join(lines)


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


def _is_deterministic_section(section: TemplateSection) -> bool:
    normalized = section.heading.lower()
    return "title" in normalized or "participant" in normalized


def _section_rules(sections: list[TemplateSection]) -> str:
    return "\n".join(
        f"{index}. {section.heading}{' <content>' if section.allow_prefix else ''}"
        for index, section in enumerate(sections, start=1)
    )


def _chunk_prompts(
    template_prompt: str,
    sections: list[TemplateSection],
    title: str,
    speakers: list[str],
    chunk_index: int,
    chunk_count: int,
    chunk_text: str,
    user_mode: str,
) -> tuple[str, str]:
    system = (
        f"{template_prompt}\n\n"
        f"Required section order:\n{_section_rules(sections)}"
    )
    user_header = (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n"
        f"Transcript chunk: chunk_{chunk_index:02d}/{chunk_count}\n\n"
    )
    if user_mode == "minimal":
        user = f"{user_header}{chunk_text}\n"
    else:
        user = (
            f"{user_header}"
            "Important chunk-local constraints:\n"
            "- This is only one chronological chunk of a longer meeting.\n"
            "- Use only the transcript text below.\n"
            "- If multiple agenda items appear, mention each one briefly in the relevant template sections.\n"
            "- Do not create TODOs unless the chunk explicitly assigns or requires an action.\n"
            "- Do not call something a council decision unless the transcript contains an explicit motion, vote, chair ruling, or council decision.\n"
            "- If a public speaker requests deferral, extra evidence, or a condition, record it as that speaker's concern/request unless council explicitly adopts it.\n"
            "- If the chunk ends before the item outcome is clear, mark the outcome as unresolved within this chunk rather than final.\n"
            "- Preserve pending/not signed/deferred/approved wording exactly when it appears.\n\n"
            f"{chunk_text}\n"
        )
    return system, user


def _polish_prompts(
    template_prompt: str,
    sections: list[TemplateSection],
    title: str,
    speakers: list[str],
    chunk_union: str,
) -> tuple[str, str]:
    system = (
        "You are given a chunk-labelled union of partial Minutes of Meeting sections.\n"
        "Each partial section was produced from one transcript chunk using the selected meeting template.\n"
        "The labels chunk_01, chunk_02, etc. show chronological order; higher chunk numbers happened later in the meeting.\n"
        "Your job is to polish the union into one coherent, non-redundant MoM body while keeping the same section headings and order.\n"
        "When two chunks conflict, a later explicit decision overrides an earlier tentative or provisional statement.\n"
        "Do not invent new facts. Do not add inferred TODOs. Preserve important distinctions such as pending, not signed, deferred, approved, or changed.\n"
        "Return ONLY the required sections listed below. Title and Participants are intentionally excluded and will be filled deterministically.\n\n"
        f"Required section order:\n{_section_rules(sections)}"
    )
    user = (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n\n"
        "Chunk-labelled section union to polish:\n"
        f"{chunk_union}\n"
    )
    return system, user


def _section_polish_prompts(
    section: TemplateSection,
    title: str,
    speakers: list[str],
    section_union: str,
) -> tuple[str, str]:
    section_name = section.heading.lower()
    extra_rules = ""
    if "todo" in section_name:
        extra_rules = (
            "\nFor TODO/action sections: keep only concrete actions assigned or accepted by council, staff, or an applicant. "
            "Do not turn public objections, requests, or desired conditions into TODOs unless the meeting explicitly adopts them. "
            "Remove TODOs that a later chunk shows were already resolved, incorporated, or superseded."
        )
    elif "decision" in section_name or "open" in section_name:
        extra_rules = (
            "\nFor decision/open point sections: state the final status of each agenda item. "
            "If an earlier chunk says an item was deferred, proposed, or pending, but a later chunk says it was approved, passed, or incorporated, "
            "do not list the earlier state as a current decision/open point. You may mention it only as an earlier concern that was later superseded."
        )
    elif "overview" in section_name or "conclusion" in section_name:
        extra_rules = (
            "\nFor narrative sections: present the meeting in chronological order, but make the final outcome clear when later chunks resolve earlier uncertainty."
        )
    system = (
        "You are given one chunk-labelled Minutes of Meeting section assembled from chronological transcript chunks.\n"
        "The labels chunk_01, chunk_02, etc. show chronological order; higher chunk numbers happened later in the meeting.\n"
        "Polish this single section into one coherent, non-redundant section.\n"
        "When chunks conflict, a later explicit decision overrides an earlier tentative, proposed, or provisional statement.\n"
        "Do not preserve both sides of a conflict as current facts; resolve the section to the final known status.\n"
        "Do not invent new facts. Do not add inferred TODOs. Preserve important distinctions such as pending, not signed, deferred, approved, or changed.\n"
        f"{extra_rules}\n"
        f"Return ONLY this exact section heading and its body:\n{section.heading}"
    )
    user = (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n\n"
        "Chunk-labelled section to polish:\n"
        f"{section_union}\n"
    )
    return system, user


def _normalize_sections(text: str, sections: list[TemplateSection]) -> str:
    outputs = []
    for index, section in enumerate(sections):
        heading = re.escape(section.heading)
        next_headings = [re.escape(item.heading) for item in sections[index + 1 :]]
        if next_headings:
            pattern = rf"(?ms)^{heading}\s*(.*?)(?=^({'|'.join(next_headings)})\s*|\Z)"
        else:
            pattern = rf"(?ms)^{heading}\s*(.*)\Z"
        match = re.search(pattern, text)
        body = match.group(1).strip() if match else "None"
        if not body:
            body = "None"
        outputs.append(f"{section.heading}\n{body}" if not section.allow_prefix else f"{section.heading} {body}")
    return "\n\n".join(outputs)


def _parse_section_bodies(text: str, sections: list[TemplateSection]) -> dict[str, list[str]]:
    normalized = _normalize_sections(text, sections)
    result: dict[str, list[str]] = {}
    for index, section in enumerate(sections):
        start = normalized.find(section.heading)
        if start < 0:
            result[section.heading] = []
            continue
        body_start = start + len(section.heading)
        next_positions = [normalized.find(next_section.heading, body_start) for next_section in sections[index + 1 :]]
        next_positions = [pos for pos in next_positions if pos >= 0]
        body = normalized[body_start : min(next_positions) if next_positions else len(normalized)].strip()
        lines = []
        for raw_line in body.splitlines():
            line = re.sub(r"^[-*]\s+", "", raw_line.strip()).strip()
            if line and line.lower() != "none":
                lines.append(line)
        result[section.heading] = lines
    return result


def _section_slice(text: str, section: TemplateSection, sections: list[TemplateSection]) -> str:
    normalized = _normalize_sections(text, sections)
    index = sections.index(section)
    start = normalized.find(section.heading)
    if start < 0:
        return f"{section.heading}\nNone"
    next_positions = [normalized.find(next_section.heading, start + len(section.heading)) for next_section in sections[index + 1 :]]
    next_positions = [pos for pos in next_positions if pos >= 0]
    return normalized[start : min(next_positions) if next_positions else len(normalized)].strip()


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return safe[:60] or "section"


def _build_chunk_union(chunk_outputs: list[tuple[int, str]], sections: list[TemplateSection]) -> str:
    bodies_by_section: dict[str, list[str]] = {section.heading: [] for section in sections}
    for chunk_index, output in chunk_outputs:
        parsed = _parse_section_bodies(output, sections)
        for section in sections:
            for line in parsed[section.heading]:
                bodies_by_section[section.heading].append(f"- chunk_{chunk_index:02d}: {line}")
    rendered = []
    for section in sections:
        body = "\n".join(bodies_by_section[section.heading]) if bodies_by_section[section.heading] else "None"
        rendered.append(f"{section.heading}\n{body}")
    return "\n\n".join(rendered)


def _deterministic_section(section: TemplateSection, title: str, speakers: list[str]) -> str:
    if "title" in section.heading.lower():
        return f"{section.heading} {title}" if section.allow_prefix else f"{section.heading}\n{title}"
    if "participant" in section.heading.lower():
        return f"{section.heading}\n" + "\n".join(f"- {speaker}" for speaker in speakers)
    return f"{section.heading}\nNone"


def _assemble_final(
    *,
    template_sections: list[TemplateSection],
    generated_sections: list[TemplateSection],
    generated_markdown: str,
    title: str,
    speakers: list[str],
) -> str:
    generated = _normalize_sections(generated_markdown, generated_sections)
    pieces = []
    for section in template_sections:
        if _is_deterministic_section(section):
            pieces.append(_deterministic_section(section, title, speakers))
        else:
            pattern = rf"(?ms)^{re.escape(section.heading)}.*?(?=^#|\Z)"
            match = re.search(pattern, generated)
            pieces.append(match.group(0).strip() if match else f"{section.heading}\nNone")
    return "\n\n".join(pieces)


def main() -> int:
    _load_dotenv(ROOT_DIR / ".env")
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else job_dir / f"template_chunk_union_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    template = TEMPLATE_MANAGER.load(args.template_id)
    generated_sections = [section for section in template.sections if not _is_deterministic_section(section)]
    transcript, speakers = _load_transcript(job_dir)
    title = _infer_title(job_dir, args.title)
    model = _infer_model(job_dir, args.model)
    host = (args.ollama_host or SETTINGS.ollama_host).rstrip("/")
    chunks = _chunk_transcript_with_first_target(
        transcript,
        first_token_target=args.first_chunk_token_target,
        token_target=args.chunk_token_target,
    )
    rerun_chunks = _parse_index_set(args.rerun_chunks)
    chunk_outputs: list[tuple[int, str]] = []
    records: list[dict[str, object]] = []

    for index, chunk in enumerate(chunks, start=1):
        system_prompt, user_prompt = _chunk_prompts(
            template.prompt_block,
            generated_sections,
            title,
            speakers,
            index,
            len(chunks),
            _render_segments(chunk),
            args.chunk_user_mode,
        )
        prompt_path = output_dir / f"chunk_{index:02d}_prompt.txt"
        raw_path = output_dir / f"chunk_{index:02d}_raw.json"
        output_path = output_dir / f"chunk_{index:02d}_sections.md"
        prompt_path.write_text(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", encoding="utf-8")
        if args.reuse_chunks and output_path.exists() and index not in rerun_chunks:
            normalized = output_path.read_text(encoding="utf-8").strip()
            chunk_outputs.append((index, normalized))
            records.append({"name": f"chunk_{index:02d}", "reused": True})
            print(f"[chunk {index}/{len(chunks)}] reused {output_path.name}", flush=True)
            continue
        print(f"[chunk {index}/{len(chunks)}] model={model}", flush=True)
        started = time.monotonic()
        output, raw = _call_ollama(
            host=host,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_ctx=args.chunk_num_ctx,
            num_predict=args.chunk_num_predict,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        normalized = _normalize_sections(output, generated_sections)
        raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        output_path.write_text(normalized + "\n", encoding="utf-8")
        chunk_outputs.append((index, normalized))
        records.append(
            {
                "name": f"chunk_{index:02d}",
                "seconds": time.monotonic() - started,
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
            }
        )

    union = _build_chunk_union(chunk_outputs, generated_sections)
    (output_dir / "chunk_section_union.md").write_text(union + "\n", encoding="utf-8")

    if len(chunks) > 1:
        if args.polish_mode == "section":
            polished_sections = []
            for section in generated_sections:
                section_union = _section_slice(union, section, generated_sections)
                system_prompt, user_prompt = _section_polish_prompts(section, title, speakers, section_union)
                safe = _safe_name(section.heading)
                (output_dir / f"polish_{safe}_prompt.txt").write_text(
                    f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                    encoding="utf-8",
                )
                print(f"[polish] merging {section.heading}", flush=True)
                started = time.monotonic()
                section_polished, raw = _call_ollama(
                    host=host,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    num_ctx=args.polish_num_ctx,
                    num_predict=args.polish_num_predict,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                )
                (output_dir / f"polish_{safe}_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
                normalized_section = _normalize_sections(section_polished, [section])
                (output_dir / f"polish_{safe}.md").write_text(normalized_section + "\n", encoding="utf-8")
                polished_sections.append(normalized_section)
                records.append(
                    {
                        "name": f"polish_{safe}",
                        "seconds": time.monotonic() - started,
                        "prompt_eval_count": raw.get("prompt_eval_count"),
                        "eval_count": raw.get("eval_count"),
                    }
                )
            polished = "\n\n".join(polished_sections)
        else:
            system_prompt, user_prompt = _polish_prompts(template.prompt_block, generated_sections, title, speakers, union)
            (output_dir / "polish_prompt.txt").write_text(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", encoding="utf-8")
            print("[polish] merging chunk-labelled sections", flush=True)
            started = time.monotonic()
            polished, raw = _call_ollama(
                host=host,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                num_ctx=args.polish_num_ctx,
                num_predict=args.polish_num_predict,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
            )
            (output_dir / "polish_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
            records.append(
                {
                    "name": "polish",
                    "seconds": time.monotonic() - started,
                    "prompt_eval_count": raw.get("prompt_eval_count"),
                    "eval_count": raw.get("eval_count"),
                }
            )
    else:
        polished = union

    normalized_polished = _normalize_sections(polished, generated_sections)
    final_mom = _assemble_final(
        template_sections=template.sections,
        generated_sections=generated_sections,
        generated_markdown=normalized_polished,
        title=title,
        speakers=speakers,
    )
    evaluation = {
        "validation": validate_markdown_output(final_mom, template.sections),
        "estimated_tokens": _estimate_tokens(final_mom),
        "chunk_count": len(chunks),
        "records": records,
    }
    (output_dir / "polished_sections.md").write_text(normalized_polished + "\n", encoding="utf-8")
    (output_dir / "mom_template_chunk_union.md").write_text(final_mom + "\n", encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(json.dumps({"validation": evaluation["validation"], "estimated_tokens": evaluation["estimated_tokens"], "chunk_count": len(chunks)}, indent=2))
    return 0 if bool(evaluation["validation"]["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
