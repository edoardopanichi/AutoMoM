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
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class CallRecord:
    name: str
    seconds: float
    prompt_tokens_estimate: int
    prompt_eval_count: int | None
    eval_count: int | None
    prompt_path: str
    raw_path: str


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map/reduce long-transcript formatter experiment.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-host", default="")
    parser.add_argument("--chunk-token-target", type=int, default=2200)
    parser.add_argument("--map-num-ctx", type=int, default=8192)
    parser.add_argument("--reduce-num-ctx", type=int, default=16384)
    parser.add_argument("--map-num-predict", type=int, default=900)
    parser.add_argument("--reduce-num-predict", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_transcript(job_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    payload = _load_json(job_dir / "transcript.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError(f"Unsupported transcript payload: {job_dir / 'transcript.json'}")
    segments = [_normalize_segment(item, index) for index, item in enumerate(payload["segments"], start=1)]
    speakers = [str(item) for item in payload.get("speakers") or [] if str(item).strip()]
    if not speakers:
        speakers = sorted({str(item["speaker_name"]) for item in segments if str(item.get("speaker_name") or "").strip()})
    return segments, speakers


def _normalize_segment(raw: dict[str, Any], index: int) -> dict[str, object]:
    fallback_speaker = f"SPEAKER_{index:04d}"
    speaker_id = str(raw.get("speaker_id") or raw.get("speaker") or fallback_speaker)
    speaker_name = str(raw.get("speaker_name") or raw.get("speaker") or speaker_id)
    text = str(raw.get("text") or raw.get("transcript") or raw.get("utterance") or "").strip()
    return {
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "start_s": float(raw.get("start_s") or raw.get("start") or 0.0),
        "end_s": float(raw.get("end_s") or raw.get("end") or 0.0),
        "text": text,
    }


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


def _render_segments(segments: list[dict[str, object]]) -> str:
    lines = []
    for item in segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{float(item.get('start_s') or 0.0):08.2f}s] {item['speaker_name']}: {text}")
    return "\n".join(lines)


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


def _clean_text(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _map_prompt(title: str, speakers: list[str], chunk_index: int, chunk_count: int, chunk_text: str) -> tuple[str, str]:
    system = (
        "Extract neutral factual notes from this transcript chunk for later Minutes of Meeting generation.\n"
        "Return concise Markdown only. Do not write the final MoM. Do not invent facts.\n"
        "Use these headings exactly:\n"
        "### Topics\n### Decisions and Conclusions\n### Action Items\n### Open Questions\n### Risks\n### Useful Evidence\n"
        "If a heading has no content, write None."
    )
    user = (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n"
        f"Transcript chunk: {chunk_index}/{chunk_count}\n\n"
        f"{chunk_text}\n"
    )
    return system, user


def _reduce_prompt(
    template_prompt: str,
    title: str,
    speakers: list[str],
    notes_text: str,
    section: TemplateSection,
) -> tuple[str, str]:
    system = (
        f"{template_prompt}\n\n"
        "You are generating one section of the final Minutes of Meeting from chunk-level factual notes.\n"
        f"Generate ONLY this required section: {section.heading}\n"
        f"The first line must start exactly with: {section.heading}\n"
        "Do not include any other required section heading. Use only the notes. If unsupported, write None."
    )
    user = (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n"
        f"Requested section: {section.heading}\n\n"
        "Chunk-level factual notes:\n"
        f"{notes_text}\n"
    )
    return system, user


def _normalize_section(text: str, section: TemplateSection) -> str:
    heading = section.heading
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return f"{heading} None" if section.allow_prefix else f"{heading}\nNone"
    start = next((idx for idx, line in enumerate(lines) if line.startswith(heading)), -1)
    if start >= 0:
        lines = lines[start:]
        kept = []
        for idx, line in enumerate(lines):
            if idx > 0 and re.match(r"^#{2,4}\s+", line):
                break
            kept.append(line)
        return "\n".join(kept).strip()
    body = "\n".join(lines)
    return f"{heading} {body}" if section.allow_prefix else f"{heading}\n{body}"


def _record_call(
    records: list[CallRecord],
    *,
    name: str,
    started: float,
    system_prompt: str,
    user_prompt: str,
    raw: dict[str, object],
    prompt_path: Path,
    raw_path: Path,
) -> None:
    records.append(
        CallRecord(
            name=name,
            seconds=time.monotonic() - started,
            prompt_tokens_estimate=_estimate_tokens(f"{system_prompt}\n\n{user_prompt}"),
            prompt_eval_count=_optional_int(raw.get("prompt_eval_count")),
            eval_count=_optional_int(raw.get("eval_count")),
            prompt_path=str(prompt_path),
            raw_path=str(raw_path),
        )
    )


def main() -> int:
    _load_dotenv(ROOT_DIR / ".env")
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else job_dir / f"map_reduce_formatter_experiment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    template = TEMPLATE_MANAGER.load(args.template_id)
    transcript, speakers = _load_transcript(job_dir)
    title = _infer_title(job_dir, args.title)
    model = _infer_model(job_dir, args.model)
    host = (args.ollama_host or SETTINGS.ollama_host).rstrip("/")
    chunks = _chunk_transcript(transcript, args.chunk_token_target)
    records: list[CallRecord] = []
    notes: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        system_prompt, user_prompt = _map_prompt(title, speakers, index, len(chunks), _render_segments(chunk))
        prompt_path = output_dir / f"map_{index:02d}_prompt.txt"
        raw_path = output_dir / f"map_{index:02d}_raw.json"
        notes_path = output_dir / f"map_{index:02d}_notes.md"
        prompt_path.write_text(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", encoding="utf-8")
        print(f"[map {index}/{len(chunks)}] model={model}", flush=True)
        started = time.monotonic()
        output, raw = _call_ollama(
            host=host,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_ctx=args.map_num_ctx,
            num_predict=args.map_num_predict,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        chunk_notes = f"## Transcript chunk {index}/{len(chunks)}\n{output}".strip()
        notes_path.write_text(chunk_notes + "\n", encoding="utf-8")
        notes.append(chunk_notes)
        _record_call(
            records,
            name=f"map_{index:02d}",
            started=started,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw=raw,
            prompt_path=prompt_path,
            raw_path=raw_path,
        )

    notes_text = "\n\n".join(notes)
    (output_dir / "combined_chunk_notes.md").write_text(notes_text + "\n", encoding="utf-8")

    section_outputs: list[str] = []
    for index, section in enumerate(template.sections, start=1):
        system_prompt, user_prompt = _reduce_prompt(template.prompt_block, title, speakers, notes_text, section)
        prompt_path = output_dir / f"reduce_{index:02d}_{_slug(section.heading)}_prompt.txt"
        raw_path = output_dir / f"reduce_{index:02d}_{_slug(section.heading)}_raw.json"
        section_path = output_dir / f"reduce_{index:02d}_{_slug(section.heading)}.md"
        prompt_path.write_text(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", encoding="utf-8")
        print(f"[reduce {index}/{len(template.sections)}] {section.heading}", flush=True)
        started = time.monotonic()
        output, raw = _call_ollama(
            host=host,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_ctx=args.reduce_num_ctx,
            num_predict=args.reduce_num_predict,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        normalized = _normalize_section(output, section)
        section_path.write_text(normalized + "\n", encoding="utf-8")
        section_outputs.append(normalized)
        _record_call(
            records,
            name=f"reduce_{index:02d}_{_slug(section.heading)}",
            started=started,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw=raw,
            prompt_path=prompt_path,
            raw_path=raw_path,
        )

    mom = "\n\n".join(section_outputs).strip() + "\n"
    evaluation = {
        "validation": validate_markdown_output(mom, template.sections),
        "estimated_tokens": _estimate_tokens(mom),
        "transcript_segments": len(transcript),
        "chunk_count": len(chunks),
    }
    (output_dir / "mom_map_reduce.md").write_text(mom, encoding="utf-8")
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "job_dir": str(job_dir),
                "template_id": args.template_id,
                "model": model,
                "ollama_host": host,
                "title": title,
                "speakers": speakers,
                "chunk_token_target": args.chunk_token_target,
                "map_num_ctx": args.map_num_ctx,
                "reduce_num_ctx": args.reduce_num_ctx,
                "records": [asdict(record) for record in records],
                "evaluation": evaluation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output_dir={output_dir}")
    print(json.dumps(evaluation, indent=2))
    return 0 if bool(evaluation["validation"]["valid"]) else 1


def _optional_int(raw: object) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "section"


if __name__ == "__main__":
    raise SystemExit(main())
