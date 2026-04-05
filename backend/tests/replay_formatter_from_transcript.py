#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

"""! Standalone formatter replay utility for debugging stage-8 MoM generation.

This script replays the formatter pipeline using an existing transcript artifact
from a job, without requiring the web GUI runtime.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_dotenv(path: Path) -> None:
    """! Load basic KEY=VALUE entries from a dotenv file into process env.

    Existing environment variables are kept unchanged so explicit shell exports
    still take precedence over `.env`.

    @param path Dotenv file path.
    @return None
    """
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


_load_dotenv(ROOT_DIR / ".env")

from backend.app.config import SETTINGS, ensure_directories
from backend.pipeline.formatter import Formatter
from backend.pipeline.template_manager import TEMPLATE_MANAGER, TemplateManager


# Support both the canonical filename and the typo variant requested by users.
TRANSCRIPT_CANDIDATES = ("transcript.json", "transcipt.json")


def _find_transcript_file(job_dir: Path) -> Path | None:
    """! Locate transcript artifact inside a job directory.

    @param job_dir Job folder path.
    @return Transcript path when found, otherwise `None`.
    """
    for name in TRANSCRIPT_CANDIDATES:
        candidate = job_dir / name
        if candidate.exists():
            return candidate
    return None


def _latest_job_with_transcript(jobs_dir: Path) -> Path:
    """! Pick the most recent job directory that has a transcript artifact.

    Recency is based on transcript file mtime because this reflects the end of
    stage 7 and keeps replay target selection deterministic.

    @param jobs_dir Root `data/jobs` directory.
    @return Latest job folder with transcript data.
    @throws FileNotFoundError If no replayable job exists.
    """
    candidates: list[tuple[float, Path]] = []
    for item in jobs_dir.iterdir():
        if not item.is_dir():
            continue
        transcript_path = _find_transcript_file(item)
        if transcript_path is None:
            continue
        candidates.append((transcript_path.stat().st_mtime, item))
    if not candidates:
        raise FileNotFoundError(f"No job folder with transcript found in: {jobs_dir}")
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def _parse_args() -> argparse.Namespace:
    """! Parse CLI options for replay selection and output controls.

    @return Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replay AutoMoM formatter stage from existing transcriber output in a job transcript JSON. "
            "This runs the same formatter pipeline path as stage 8 (no web GUI required)."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--job-id", help="Job ID under data/jobs (for example: 144b7eca-...)")
    group.add_argument("--job-dir", type=Path, help="Direct path to an existing job directory")
    group.add_argument("--transcript-path", type=Path, help="Direct path to transcript JSON")
    parser.add_argument("--template-id", default="default", help="Template ID used for formatter prompt")
    parser.add_argument(
        "--title",
        help=(
            "Meeting title for prompt assembly. If omitted, script tries previous formatter_user_prompt.txt title "
            "then falls back to job folder name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder for replay artifacts. Defaults to <job_dir>/formatter_replay_<utc_timestamp>/",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the reconstructed formatter prompt to stdout",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """! Resolve transcript/job inputs based on CLI flags.

    @param args Parsed CLI arguments.
    @return Tuple `(transcript_path, job_dir_or_none)`.
    @throws FileNotFoundError If provided paths are invalid or transcript is missing.
    """
    if args.transcript_path:
        transcript_path = args.transcript_path.expanduser().resolve()
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        return transcript_path, None

    if args.job_id:
        job_dir = (SETTINGS.jobs_dir / args.job_id).resolve()
    elif args.job_dir:
        job_dir = args.job_dir.expanduser().resolve()
    else:
        job_dir = _latest_job_with_transcript(SETTINGS.jobs_dir.resolve())

    if not job_dir.exists() or not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    transcript_path = _find_transcript_file(job_dir)
    if transcript_path is None:
        candidate_names = ", ".join(TRANSCRIPT_CANDIDATES)
        raise FileNotFoundError(f"No transcript file ({candidate_names}) in {job_dir}")
    return transcript_path, job_dir


def _as_float(raw: Any, *, default: float = 0.0) -> float:
    """! Convert unknown numeric-like input to float with fallback.

    @param raw Raw value to parse.
    @param default Fallback value when parsing fails.
    @return Parsed float or fallback.
    """
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _normalize_transcript_segment(raw: dict[str, Any], idx: int) -> dict[str, object]:
    """! Normalize one transcript segment into formatter-compatible structure.

    @param raw Source segment payload from transcript JSON.
    @param idx 1-based segment index, used for deterministic fallback IDs.
    @return Normalized segment dictionary.
    """
    fallback_id = f"SPEAKER_{idx:04d}"
    speaker_id = str(raw.get("speaker_id") or raw.get("speaker") or fallback_id)
    speaker_name = str(raw.get("speaker_name") or raw.get("speaker") or speaker_id)
    text = str(raw.get("text") or raw.get("transcript") or raw.get("utterance") or "").strip()
    start_s = _as_float(raw.get("start_s", raw.get("start", 0.0)))
    end_s = _as_float(raw.get("end_s", raw.get("end", start_s)), default=start_s)
    return {
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "start_s": start_s,
        "end_s": end_s,
        "text": text,
    }


def _load_transcript_payload(transcript_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    """! Load and normalize transcript payload from a replay source file.

    Supported input shapes:
    - object with `segments` list (standard AutoMoM artifact)
    - object with `transcript` list
    - top-level segment list

    @param transcript_path Transcript JSON file path.
    @return Tuple `(normalized_segments, speakers)`.
    @throws ValueError If transcript format is invalid or empty.
    """
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    raw_segments: list[Any]
    speakers: list[str] = []

    if isinstance(payload, dict):
        maybe_speakers = payload.get("speakers")
        if isinstance(maybe_speakers, list):
            speakers = [str(item) for item in maybe_speakers if str(item).strip()]
        if isinstance(payload.get("segments"), list):
            raw_segments = payload["segments"]
        elif isinstance(payload.get("transcript"), list):
            raw_segments = payload["transcript"]
        else:
            raise ValueError("Transcript payload must contain 'segments' or 'transcript' list")
    elif isinstance(payload, list):
        raw_segments = payload
    else:
        raise ValueError("Transcript JSON must be a list or object")

    normalized = [_normalize_transcript_segment(item, idx) for idx, item in enumerate(raw_segments, start=1)]
    if not normalized:
        raise ValueError(f"Transcript has no segments: {transcript_path}")

    if not speakers:
        speakers = sorted({str(item["speaker_name"]) for item in normalized})
    return normalized, speakers


def _parse_title_from_existing_prompt(job_dir: Path) -> str | None:
    """! Extract `Title:` value from an existing formatter prompt artifact.

    @param job_dir Job directory path.
    @return Parsed title or `None` when missing.
    """
    prompt_path = job_dir / "formatter_user_prompt.txt"
    if not prompt_path.exists():
        return None
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Title:"):
            parsed = line.partition(":")[2].strip()
            if parsed:
                return parsed
    return None


def _resolve_title(args: argparse.Namespace, transcript_path: Path, job_dir: Path | None) -> str:
    """! Resolve replay title using CLI override, prior prompt, or fallback.

    Fallback order mirrors practical runtime debugging needs:
    explicit CLI title -> previous prompt title -> job folder -> transcript stem.

    @param args Parsed CLI arguments.
    @param transcript_path Transcript source path.
    @param job_dir Optional job directory.
    @return Title string for prompt assembly.
    """
    if args.title:
        return args.title.strip()
    if job_dir is not None:
        prompt_title = _parse_title_from_existing_prompt(job_dir)
        if prompt_title:
            return prompt_title
        return job_dir.name
    return transcript_path.stem


def _resolve_output_dir(args: argparse.Namespace, transcript_path: Path, job_dir: Path | None) -> Path:
    """! Compute replay artifact output directory.

    @param args Parsed CLI arguments.
    @param transcript_path Transcript source path.
    @param job_dir Optional job directory.
    @return Absolute output directory path.
    """
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    base_dir = job_dir if job_dir is not None else transcript_path.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (base_dir / f"formatter_replay_{timestamp}").resolve()


def _write_text(path: Path, text: str) -> None:
    """! Write UTF-8 text file, creating parent directories when needed.

    @param path Output file path.
    @param text Text payload.
    @return None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """! Write JSON artifact with stable indentation.

    @param path Output file path.
    @param payload JSON-serializable dictionary payload.
    @return None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_formatter_debug_artifacts(output_dir: Path, formatter: Formatter, *, system_prompt: str, user_prompt: str) -> None:
    """! Persist replay debug artifacts aligned with stage-8 outputs.

    @param output_dir Replay output directory.
    @param formatter Formatter instance with captured streams/mode.
    @param system_prompt Reconstructed formatter system prompt.
    @param user_prompt Reconstructed formatter user prompt.
    @return None
    """
    _write_text(output_dir / "formatter_system_prompt.txt", system_prompt)
    _write_text(output_dir / "formatter_user_prompt.txt", user_prompt)
    if formatter.last_stdout:
        _write_text(output_dir / "formatter_stdout.txt", formatter.last_stdout)
    if formatter.last_stderr:
        _write_text(output_dir / "formatter_stderr.txt", formatter.last_stderr)
    if formatter.last_raw_output:
        _write_text(output_dir / "formatter_raw_output.txt", formatter.last_raw_output)


def main() -> int:
    """! Execute formatter replay from transcript artifact to `mom.md`.

    Reuses the same formatter call path as orchestrator stage 8:
    `Formatter.write_model_output_to_mom(...)`.

    @return POSIX exit code: `0` success, `1` formatter failure.
    """
    args = _parse_args()
    ensure_directories()
    TemplateManager()

    transcript_path, job_dir = _resolve_paths(args)
    transcript, speakers = _load_transcript_payload(transcript_path)
    title = _resolve_title(args, transcript_path, job_dir)
    output_dir = _resolve_output_dir(args, transcript_path, job_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mom_path = output_dir / "mom.md"

    formatter = Formatter(
        command_template=SETTINGS.formatter_command if SETTINGS.formatter_backend == "command" else "",
        model_path=SETTINGS.formatter_model_path if SETTINGS.formatter_backend == "command" else "",
        ollama_host=SETTINGS.ollama_host,
        ollama_model=SETTINGS.formatter_ollama_model,
        timeout_s=SETTINGS.formatter_timeout_s,
    )

    formatter_error: str | None = None
    structured: dict[str, object] | None = None
    system_prompt = ""
    user_prompt = ""
    try:
        # Same stage-8 function call used by the orchestrator path.
        formatter_result = formatter.write_model_output_to_mom(
            transcript=transcript,
            speakers=speakers,
            title=title,
            template_id=args.template_id,
            output_path=mom_path,
        )
        structured = formatter_result.structured
        system_prompt = formatter_result.system_prompt
        user_prompt = formatter_result.user_prompt
    except RuntimeError as exc:
        formatter_error = str(exc)
        bundle = TEMPLATE_MANAGER.build_formatter_request(args.template_id, transcript, speakers, title)
        system_prompt = bundle.system_prompt
        user_prompt = bundle.user_prompt

    _write_formatter_debug_artifacts(output_dir, formatter, system_prompt=system_prompt, user_prompt=user_prompt)
    if structured is not None:
        _write_json(output_dir / "mom_structured.json", structured)

    print(f"transcript_path={transcript_path}")
    print(f"job_dir={job_dir if job_dir is not None else 'n/a'}")
    print(f"output_dir={output_dir}")
    print(f"title={title}")
    print(f"template_id={args.template_id}")
    print(f"speakers={', '.join(speakers)}")
    print(f"formatter_mode={formatter.last_mode}")

    if args.print_prompt:
        print("\n--- formatter_system_prompt.txt ---")
        print(system_prompt)
        print("\n--- formatter_user_prompt.txt ---")
        print(user_prompt)

    if formatter_error is not None:
        print(f"error={formatter_error}")
        if not mom_path.exists():
            print("mom.md was not generated because formatter output was empty.")
        return 1

    print(f"mom_path={mom_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
