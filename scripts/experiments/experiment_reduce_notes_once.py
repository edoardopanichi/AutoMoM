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
    parser = argparse.ArgumentParser(description="Reduce existing chunk notes into one xlab MoM.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument(
        "--notes-path",
        type=Path,
        default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix/map_reduce_formatter_v2/combined_chunk_notes.md",
    )
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-host", default="")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=int, default=1200)
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
        text = re.sub(r"^```(?:markdown)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _build_prompts(template_id: str, title: str, speakers: list[str], notes: str) -> tuple[str, str]:
    template = TEMPLATE_MANAGER.load(template_id)
    ordered = "\n".join(
        f"{index}. {section.heading}{' <content>' if section.allow_prefix else ''}"
        for index, section in enumerate(template.sections, start=1)
    )
    system = (
        f"{template.prompt_block}\n\n"
        "You are generating the final Minutes of Meeting from chunk-level factual notes extracted from the transcript.\n"
        "Use only these notes. Preserve important decisions, conditions, action items, open questions, and risks.\n"
        "Avoid copying every detail; synthesize across chunks.\n\n"
        f"Required section order:\n{ordered}\n\n"
        "Return markdown only. Use exactly the required headings and order. If a section has no content, write exactly None."
    )
    user = (
        f"Meeting title hint: {title}\n"
        f"Known speakers: {', '.join(speakers)}\n\n"
        "Chunk-level factual notes:\n"
        f"{notes}\n"
    )
    return system, user


def main() -> int:
    _load_dotenv(ROOT_DIR / ".env")
    args = _parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    notes_path = args.notes_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else job_dir / f"single_reduce_formatter_experiment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    notes = notes_path.read_text(encoding="utf-8")
    title = _infer_title(job_dir, args.title)
    speakers = _load_speakers(job_dir)
    model = _infer_model(job_dir, args.model)
    host = (args.ollama_host or SETTINGS.ollama_host).rstrip("/")
    system_prompt, user_prompt = _build_prompts(args.template_id, title, speakers, notes)
    (output_dir / "prompt.txt").write_text(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", encoding="utf-8")

    print(f"model={model} num_ctx={args.num_ctx} notes_tokens_estimate={_estimate_tokens(notes)}", flush=True)
    started = time.monotonic()
    markdown, raw = _call_ollama(
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
    template = TEMPLATE_MANAGER.load(args.template_id)
    evaluation = {
        "validation": validate_markdown_output(markdown, template.sections),
        "estimated_tokens": _estimate_tokens(markdown),
        "seconds": seconds,
        "prompt_eval_count": raw.get("prompt_eval_count"),
        "eval_count": raw.get("eval_count"),
        "prompt_eval_duration": raw.get("prompt_eval_duration"),
        "eval_duration": raw.get("eval_duration"),
    }
    (output_dir / "mom_single_reduce.md").write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(json.dumps(evaluation, indent=2))
    return 0 if bool(evaluation["validation"]["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
