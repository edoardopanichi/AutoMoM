#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS, ensure_directories
from backend.pipeline.formatter import (
    FORMATTER_CHUNK_DURATION_S,
    FORMATTER_CHUNK_TOKEN_TARGET,
    Formatter,
    _chunk_time_bounds,
    _chunk_transcript_by_time_and_tokens,
    _estimate_tokens,
    _extract_model_text,
    _render_transcript_lines,
)
from backend.tests.replay_formatter_from_transcript import _load_transcript_payload


@dataclass(frozen=True)
class CallStats:
    index: int
    kind: str
    seconds: float
    prompt_chars: int
    response_chars: int
    thinking_chars: int
    done_reason: str
    prompt_eval_count: int | None
    eval_count: int | None
    mode: str


class ExperimentFormatter(Formatter):
    def __init__(
        self,
        *,
        output_dir: Path,
        think: bool | str | None,
        chunk_think: bool | str | None,
        final_think: bool | str | None,
        num_ctx: int | None,
        num_predict: int | None,
        chunk_num_predict: int | None,
        final_num_predict: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.think = think
        self.chunk_think = chunk_think
        self.final_think = final_think
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.chunk_num_predict = chunk_num_predict
        self.final_num_predict = final_num_predict
        self.temperature = temperature
        self.calls: list[CallStats] = []

    def run_model(self, prompt: str, *, system_prompt: str = "") -> dict[str, object] | None:
        system_prompt = system_prompt or (
            "Write concise, professional markdown minutes of meeting in English. "
            "Return only the final markdown document."
        )
        call_index = len(self.calls) + 1
        kind = "chunk" if "Summarize one chronological meeting transcript chunk" in system_prompt else "final"
        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }
        if self.think is not None:
            payload["think"] = self.think
        if kind == "chunk" and self.chunk_think is not None:
            payload["think"] = self.chunk_think
        if kind == "final" and self.final_think is not None:
            payload["think"] = self.final_think
        options: dict[str, Any] = {}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if kind == "chunk" and self.chunk_num_predict is not None:
            options["num_predict"] = self.chunk_num_predict
        if kind == "final" and self.final_num_predict is not None:
            options["num_predict"] = self.final_num_predict
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if options:
            payload["options"] = options

        call_dir = self.output_dir / f"call_{call_index:02d}_{kind}"
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        (call_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")
        (call_dir / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        request = urllib.request.Request(
            url=f"{self.ollama_host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            self.last_mode = "heuristic_ollama_http_error"
            (call_dir / "stderr.txt").write_text(self.last_stderr, encoding="utf-8")
            return None
        except TimeoutError as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            self.last_mode = "heuristic_ollama_timeout"
            (call_dir / "stderr.txt").write_text(self.last_stderr, encoding="utf-8")
            return None
        except urllib.error.URLError as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            reason = getattr(exc, "reason", None)
            self.last_mode = (
                "heuristic_ollama_timeout"
                if isinstance(reason, (TimeoutError, socket.timeout))
                else "heuristic_ollama_unavailable"
            )
            (call_dir / "stderr.txt").write_text(self.last_stderr, encoding="utf-8")
            return None

        seconds = time.time() - started
        self.last_stdout = body
        self.last_stderr = ""
        (call_dir / "raw_response.json").write_text(body, encoding="utf-8")

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            self.last_raw_output = ""
            self.last_mode = "heuristic_ollama_invalid_json"
            return None

        output = _extract_model_text(str(parsed.get("response", "")), "")
        (call_dir / "response.md").write_text(output, encoding="utf-8")
        if parsed.get("thinking"):
            (call_dir / "thinking.txt").write_text(str(parsed.get("thinking")), encoding="utf-8")

        finalized = self._finalize_model_output(output)
        self.calls.append(
            CallStats(
                index=call_index,
                kind=kind,
                seconds=round(seconds, 3),
                prompt_chars=len(system_prompt) + len(prompt),
                response_chars=len(str(parsed.get("response", ""))),
                thinking_chars=len(str(parsed.get("thinking", ""))),
                done_reason=str(parsed.get("done_reason") or ""),
                prompt_eval_count=parsed.get("prompt_eval_count"),
                eval_count=parsed.get("eval_count"),
                mode=self.last_mode,
            )
        )
        _write_json(self.output_dir / "calls_completed.json", {"calls": [call.__dict__ for call in self.calls]})
        return finalized


def _parse_think(raw: str) -> bool | str | None:
    value = raw.strip().lower()
    if value in {"", "omit", "none"}:
        return None
    if value in {"false", "0", "no", "off"}:
        return False
    if value in {"true", "1", "yes", "on"}:
        return True
    return raw.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay AutoMoM formatter with configurable Ollama options.")
    parser.add_argument("--job-dir", type=Path, default=SETTINGS.jobs_dir / "2026-04-12-12:36-super_long_fix")
    parser.add_argument("--template-id", default="xlab_MoM")
    parser.add_argument("--title", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-host", default=SETTINGS.ollama_host)
    parser.add_argument("--think", default="omit", help="omit, true, false, low, medium, or high")
    parser.add_argument("--chunk-think", default="", help="Override --think for chunk-summary calls")
    parser.add_argument("--final-think", default="", help="Override --think for final-render calls")
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--chunk-num-predict", type=int, default=None)
    parser.add_argument("--final-num-predict", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _command_snapshot(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return {"available": False, "error": f"command not found: {command[0]}"}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _write_run_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    job_dir: Path,
    title: str,
    transcript: list[dict[str, object]],
    speakers: list[str],
) -> None:
    chunks = _chunk_transcript_by_time_and_tokens(
        transcript,
        duration_s=FORMATTER_CHUNK_DURATION_S,
        token_target=FORMATTER_CHUNK_TOKEN_TARGET,
    )
    chunk_rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        start_s, end_s = _chunk_time_bounds(chunk)
        chunk_rows.append(
            {
                "chunk_index": index,
                "segment_count": len(chunk),
                "start_s": start_s,
                "end_s": end_s,
                "estimated_tokens": _estimate_tokens(_render_transcript_lines(chunk, include_timestamps=True)),
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "job_dir": str(job_dir),
        "output_dir": str(output_dir),
        "title": title,
        "template_id": args.template_id,
        "model": args.model,
        "ollama_host": args.ollama_host,
        "think": args.think,
        "chunk_think": args.chunk_think,
        "final_think": args.final_think,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "chunk_num_predict": args.chunk_num_predict,
        "final_num_predict": args.final_num_predict,
        "temperature": args.temperature,
        "timeout_s": args.timeout_s,
        "transcript": {
            "segment_count": len(transcript),
            "speaker_count": len(speakers),
            "speakers": speakers,
            "estimated_tokens": _estimate_tokens(_render_transcript_lines(transcript, include_timestamps=True)),
            "chunk_count": len(chunks),
            "chunks": chunk_rows,
        },
        "system_before": {
            "nvidia_smi": _command_snapshot(["nvidia-smi"]),
            "ollama_ps": _command_snapshot(["ollama", "ps"]),
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)


def main() -> int:
    args = _parse_args()
    ensure_directories()
    job_dir = args.job_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript, speakers = _load_transcript_payload(job_dir / "transcript.json")
    title = args.title.strip() or job_dir.name
    _write_run_manifest(
        args=args,
        output_dir=output_dir,
        job_dir=job_dir,
        title=title,
        transcript=transcript,
        speakers=speakers,
    )

    formatter = ExperimentFormatter(
        output_dir=output_dir,
        think=_parse_think(args.think),
        chunk_think=_parse_think(args.chunk_think) if args.chunk_think.strip() else None,
        final_think=_parse_think(args.final_think) if args.final_think.strip() else None,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        chunk_num_predict=args.chunk_num_predict,
        final_num_predict=args.final_num_predict,
        temperature=args.temperature,
        ollama_host=args.ollama_host.rstrip("/"),
        ollama_model=args.model,
        timeout_s=args.timeout_s,
    )
    started = time.time()
    error = ""
    result = None
    try:
        result = formatter.write_model_output_to_mom(
            transcript=transcript,
            speakers=speakers,
            title=title,
            template_id=args.template_id,
            output_path=output_dir / "mom.md",
        )
    except Exception as exc:
        error = str(exc)
        (output_dir / "error.txt").write_text(error, encoding="utf-8")

    if result is not None:
        (output_dir / "formatter_system_prompt.txt").write_text(result.system_prompt, encoding="utf-8")
        (output_dir / "formatter_user_prompt.txt").write_text(result.user_prompt, encoding="utf-8")
        (output_dir / "formatter_validation.json").write_text(json.dumps(result.validation, indent=2), encoding="utf-8")
        (output_dir / "formatter_reduced_notes.json").write_text(
            json.dumps(result.reduced_notes, indent=2),
            encoding="utf-8",
        )
        (output_dir / "mom_structured.json").write_text(json.dumps(result.structured, indent=2), encoding="utf-8")
        if formatter.last_raw_output:
            (output_dir / "formatter_raw_output.txt").write_text(formatter.last_raw_output, encoding="utf-8")
        if formatter.last_stdout:
            (output_dir / "formatter_stdout.txt").write_text(formatter.last_stdout, encoding="utf-8")
        if formatter.last_stderr:
            (output_dir / "formatter_stderr.txt").write_text(formatter.last_stderr, encoding="utf-8")

    summary = {
        "ok": result is not None,
        "error": error,
        "model": args.model,
        "think": args.think,
        "chunk_think": args.chunk_think,
        "final_think": args.final_think,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "chunk_num_predict": args.chunk_num_predict,
        "final_num_predict": args.final_num_predict,
        "temperature": args.temperature,
        "elapsed_s": round(time.time() - started, 3),
        "formatter_mode": formatter.last_mode,
        "markdown_chars": len(result.markdown) if result is not None else 0,
        "reduced_note_count": len(result.reduced_notes) if result is not None else 0,
        "validation": result.validation if result is not None else None,
        "calls": [call.__dict__ for call in formatter.calls],
        "system_after": {
            "nvidia_smi": _command_snapshot(["nvidia-smi"]),
            "ollama_ps": _command_snapshot(["ollama", "ps"]),
        },
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
