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
class SectionRun:
    heading: str
    seconds: float
    prompt_tokens_estimate: int
    prompt_eval_count: int | None
    eval_count: int | None
    raw_output_path: str
    prompt_path: str
    normalized_output: str


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
    parser = argparse.ArgumentParser(
        description="Experiment with section-by-section full-transcript formatter calls for one long job."
    )
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-host", default="")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_transcript(job_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    transcript_path = job_dir / "transcript.json"
    payload = _load_json(transcript_path)
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        speakers = [str(item) for item in payload.get("speakers") or [] if str(item).strip()]
        segments = [_normalize_segment(item, index) for index, item in enumerate(payload["segments"], start=1)]
    else:
        segments_path = job_dir / "segments_transcript.json"
        segments = [_normalize_segment(item, index) for index, item in enumerate(_load_json(segments_path), start=1)]
        speakers = []
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
        formatter = payload.get("formatter") if isinstance(payload, dict) else None
        if isinstance(formatter, dict) and formatter.get("model"):
            return str(formatter["model"]).strip()
        execution = payload.get("execution") if isinstance(payload, dict) else None
        if isinstance(execution, dict):
            nested = execution.get("formatter")
            if isinstance(nested, dict) and nested.get("model"):
                return str(nested["model"]).strip()
    return SETTINGS.formatter_ollama_model


def _render_transcript(segments: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for item in segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start_s = float(item.get("start_s") or 0.0)
        speaker = str(item.get("speaker_name") or item.get("speaker_id") or "Speaker")
        lines.append(f"[{start_s:08.2f}s] {speaker}: {text}")
    return "\n".join(lines)


def _system_prompt_for_section(template_prompt: str, section: TemplateSection) -> str:
    heading = section.heading
    title_rule = " The title content must stay on the same line as the heading." if section.allow_prefix else ""
    return (
        f"{template_prompt}\n\n"
        "You are running an experimental section-by-section formatter pass.\n"
        f"Generate ONLY this required section: {heading}\n"
        f"Return Markdown only. The first line must start exactly with: {heading}{title_rule}\n"
        "Do not include any other required section heading.\n"
        "Use only facts supported by the transcript. If the transcript does not support content for this section, write None.\n"
        "Be concise, but include all important facts for this section."
    )


def _user_prompt_for_section(
    *,
    title: str,
    speakers: list[str],
    transcript_text: str,
    section: TemplateSection,
) -> str:
    return (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n"
        f"Requested section: {section.heading}\n\n"
        "Full transcript:\n"
        f"{transcript_text}\n"
    )


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
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
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
    parsed = json.loads(body)
    return _clean_model_text(_extract_model_text(str(parsed.get("response", "")), "")), parsed


def _clean_model_text(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_section_output(text: str, section: TemplateSection) -> str:
    heading = section.heading
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return f"{heading} None" if section.allow_prefix else f"{heading}\nNone"
    first_heading_index = next((idx for idx, line in enumerate(lines) if line.startswith(heading)), -1)
    if first_heading_index >= 0:
        lines = lines[first_heading_index:]
        body: list[str] = []
        for idx, line in enumerate(lines):
            if idx > 0 and re.match(r"^#{2,4}\s+", line):
                break
            body.append(line)
        return "\n".join(body).strip()
    body_text = "\n".join(lines).strip()
    if section.allow_prefix:
        return f"{heading} {body_text}"
    return f"{heading}\n{body_text}"


def _evaluate_markdown(markdown: str, sections: list[TemplateSection]) -> dict[str, object]:
    validation = validate_markdown_output(markdown, sections)
    repeated_headings = {
        section.heading: len(re.findall(rf"(?m)^{re.escape(section.heading)}", markdown))
        for section in sections
    }
    return {
        "validation": validation,
        "repeated_headings": repeated_headings,
        "has_placeholder_none": "None" in markdown,
        "estimated_tokens": _estimate_tokens(markdown),
    }


def main() -> int:
    _load_dotenv(ROOT_DIR / ".env")
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else job_dir / f"sectionwise_formatter_experiment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    template = TEMPLATE_MANAGER.load(args.template_id)
    sections = template.sections
    transcript, speakers = _load_transcript(job_dir)
    transcript_text = _render_transcript(transcript)
    title = _infer_title(job_dir, args.title)
    model = _infer_model(job_dir, args.model)
    host = (args.ollama_host or SETTINGS.ollama_host).rstrip("/")

    section_outputs: list[str] = []
    runs: list[SectionRun] = []
    for index, section in enumerate(sections, start=1):
        system_prompt = _system_prompt_for_section(template.prompt_block, section)
        user_prompt = _user_prompt_for_section(
            title=title,
            speakers=speakers,
            transcript_text=transcript_text,
            section=section,
        )
        prompt_path = output_dir / f"{index:02d}_{_slug(section.heading)}_prompt.txt"
        raw_path = output_dir / f"{index:02d}_{_slug(section.heading)}_raw.json"
        prompt_path.write_text(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", encoding="utf-8")

        print(f"[{index}/{len(sections)}] {section.heading} model={model} num_ctx={args.num_ctx}", flush=True)
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
        normalized = _normalize_section_output(output, section)
        section_outputs.append(normalized)
        runs.append(
            SectionRun(
                heading=section.heading,
                seconds=seconds,
                prompt_tokens_estimate=_estimate_tokens(f"{system_prompt}\n\n{user_prompt}"),
                prompt_eval_count=_optional_int(raw.get("prompt_eval_count")),
                eval_count=_optional_int(raw.get("eval_count")),
                raw_output_path=str(raw_path),
                prompt_path=str(prompt_path),
                normalized_output=normalized,
            )
        )
        (output_dir / f"{index:02d}_{_slug(section.heading)}_normalized.md").write_text(normalized, encoding="utf-8")

    mom = "\n\n".join(section_outputs).strip() + "\n"
    evaluation = _evaluate_markdown(mom, sections)
    (output_dir / "mom_sectionwise.md").write_text(mom, encoding="utf-8")
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "job_dir": str(job_dir),
                "template_id": args.template_id,
                "model": model,
                "ollama_host": host,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "temperature": args.temperature,
                "title": title,
                "speakers": speakers,
                "transcript_segments": len(transcript),
                "transcript_tokens_estimate": _estimate_tokens(transcript_text),
                "runs": [asdict(run) for run in runs],
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
