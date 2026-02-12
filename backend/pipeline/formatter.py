from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import SETTINGS
from backend.pipeline.compute import should_enable_native_gpu
from backend.pipeline.template_manager import TEMPLATE_MANAGER


ACTION_OWNER_PATTERN = re.compile(r"(?P<owner>[A-Z][a-z]+) will (?P<task>[^.]+)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2}|tomorrow|next week|monday|tuesday|wednesday|thursday|friday)\b", re.IGNORECASE)
LLAMA_FLAGS = ("--single-turn",)
LLAMA_GPU_FLAG_TOKENS = ("-ngl", "--gpu-layers", "--n-gpu-layers")
LLAMA_BINARY_PATTERN = re.compile(r"(^|[\s;|&()])(?P<cmd>(?:[^\s;|&()]+/)?llama-cli)(?=$|[\s;|&()])")
SHELL_WRAPPERS = {"bash", "sh", "zsh"}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNTIME_LOG_PREFIXES = (
    "warning: no usable gpu found",
    "warning: one possible reason is that llama.cpp was compiled without gpu support",
    "warning: consult docs/build.md for compilation instructions",
    "llama_",
    "ggml_",
    "build:",
    "load_tensors:",
    "model_loader:",
    "common_init_from_params:",
    "system_info:",
    "print_info:",
    "error: unknown argument",
    "error: invalid argument",
)


class Formatter:
    def __init__(
        self,
        command_template: str | None = None,
        model_path: str | None = None,
        *,
        compute_device: str = "auto",
        cuda_device_id: int = 0,
        gpu_layers: int = 99,
    ) -> None:
        self.command_template = command_template or ""
        self.model_path = model_path or ""
        self._cuda_device_id = max(0, int(cuda_device_id))
        self._gpu_layers = max(0, int(gpu_layers))
        self._gpu_requested = should_enable_native_gpu(compute_device, self._cuda_device_id)
        self._gpu_retry_disabled = False
        self.last_raw_output: str = ""
        self.last_stdout: str = ""
        self.last_stderr: str = ""
        self.last_mode: str = "heuristic"

    def run_model(self, prompt: str) -> dict[str, object] | None:
        if not self.command_template:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_no_command"
            return None

        if not self.model_path or not Path(self.model_path).exists():
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_missing_model_path"
            return None

        command = self.command_template.format(model=self.model_path)
        use_gpu = self._gpu_requested and not self._gpu_retry_disabled
        args = _ensure_non_interactive_llama_command(
            shlex.split(command),
            use_gpu=use_gpu,
            gpu_layers=self._gpu_layers,
        )
        try:
            process = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=SETTINGS.formatter_timeout_s,
            )
            output = _extract_model_text(process.stdout, process.stderr)
            invocation_failed = _invocation_failed(process)
            if use_gpu and _should_retry_without_gpu(process):
                retry_args = _ensure_non_interactive_llama_command(
                    shlex.split(command),
                    use_gpu=False,
                    gpu_layers=self._gpu_layers,
                )
                retry_process = subprocess.run(
                    retry_args,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=SETTINGS.formatter_timeout_s,
                )
                retry_output = _extract_model_text(retry_process.stdout, retry_process.stderr)
                if retry_output:
                    self._gpu_retry_disabled = True
                    process = retry_process
                    output = retry_output
                    invocation_failed = False
                elif not _invocation_failed(retry_process):
                    self._gpu_retry_disabled = True
                    process = retry_process
                    output = retry_output
                    invocation_failed = False
        except subprocess.TimeoutExpired:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_command_timeout"
            return None
        except (OSError, ValueError):
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_command_error"
            return None

        self.last_stdout = process.stdout or ""
        self.last_stderr = process.stderr or ""
        if not output:
            if invocation_failed:
                self.last_raw_output = ""
                self.last_mode = "heuristic_command_nonzero"
                return None
            self.last_raw_output = ""
            self.last_mode = "heuristic_empty_output"
            return None
        self.last_raw_output = output

        parsed = self._parse_json_output(output)
        if parsed is not None:
            self.last_mode = "model_json"
            return parsed

        if _looks_like_markdown_document(output):
            self.last_mode = "model_markdown"
            return {"_raw_markdown_text": output}

        # Accept plain-text model output and inject it into the structured fallback.
        self.last_mode = "model_plaintext"
        return {"_raw_model_text": output}

    def build_structured_summary(
        self,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
        template_id: str,
    ) -> tuple[str, dict[str, object], str]:
        prompt = TEMPLATE_MANAGER.build_formatter_prompt(template_id, transcript, speakers, title)
        self.run_model(prompt)
        structured = self._heuristic_structuring(transcript, speakers, title)
        markdown = self.last_raw_output if self.last_raw_output else ""
        return markdown, structured, prompt

    @staticmethod
    def _parse_json_output(output: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from fenced code blocks or mixed output.
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            try:
                parsed = json.loads(fenced_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None
        return None

    def _heuristic_structuring(
        self,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
    ) -> dict[str, object]:
        statements = [str(seg["text"]).strip() for seg in transcript if str(seg["text"]).strip()]
        statements = [item for item in statements if not item.startswith("[Offline fallback transcript")]

        discussion_summary = _discussion_bullets(transcript)
        decisions = _keyword_extract(statements, ["decide", "approved", "agreed", "resolution"])
        open_items = _keyword_extract(statements, ["risk", "issue", "question", "blocker"])[:8]
        actions = _extract_actions(statements)

        transcript_md = "\n".join(
            f"- **{seg['speaker_name']}**: {seg['text']}"
            for seg in transcript
        )

        return {
            "title": title,
            "date_time": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "participants": speakers,
            "agenda": ["Inferred from discussion context"],
            "discussion_summary": discussion_summary,
            "decisions": decisions,
            "action_items": actions,
            "open_questions_risks": open_items,
            "transcript_markdown": transcript_md,
        }



def _discussion_bullets(transcript: list[dict[str, object]], top_n: int = 8) -> list[str]:
    counter = Counter(str(item["speaker_name"]) for item in transcript)
    bullets: list[str] = []
    for seg in transcript[: top_n * 2]:
        speaker = str(seg["speaker_name"])
        text = str(seg["text"]).strip()
        if not text:
            continue
        if text.startswith("[Offline fallback transcript"):
            continue
        prefix = f"{speaker}"
        if counter[speaker] > 1:
            bullets.append(f"{prefix}: {text}")
        else:
            bullets.append(text)
        if len(bullets) >= top_n:
            break

    if not bullets:
        bullets = ["No high-confidence transcript content available."]
    return bullets



def _keyword_extract(statements: list[str], keywords: list[str]) -> list[str]:
    found: list[str] = []
    for text in statements:
        lower = text.lower()
        if any(keyword in lower for keyword in keywords):
            found.append(text)
    return found[:10]



def _extract_actions(statements: list[str]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for text in statements:
        match = ACTION_OWNER_PATTERN.search(text)
        if not match:
            continue
        due_date = "Not specified"
        date_match = DATE_PATTERN.search(text)
        if date_match:
            due_date = date_match.group(1)
        actions.append(
            {
                "owner": match.group("owner"),
                "task": match.group("task").strip(),
                "due_date": due_date,
            }
        )
    return actions[:12]


def _ensure_non_interactive_llama_command(
    args: list[str],
    *,
    use_gpu: bool = False,
    gpu_layers: int = 99,
) -> list[str]:
    if not args:
        return args

    if _is_shell_wrapper(args):
        script = args[2]
        missing_flags: list[str] = []
        if "--single-turn" not in script:
            missing_flags.append("--single-turn")
        if use_gpu and gpu_layers > 0 and not any(token in script for token in LLAMA_GPU_FLAG_TOKENS):
            missing_flags.extend(["-ngl", str(gpu_layers)])
        if not missing_flags or "llama-cli" not in script:
            return args
        flags = " ".join(missing_flags)

        def _inject(match: re.Match[str]) -> str:
            return f"{match.group(1)}{match.group('cmd')} {flags}"

        updated_script, replacements = LLAMA_BINARY_PATTERN.subn(_inject, script, count=1)
        if replacements == 0:
            return args
        updated = args.copy()
        updated[2] = updated_script
        return updated

    llama_index = next((idx for idx, token in enumerate(args) if Path(token).name == "llama-cli"), None)
    if llama_index is None:
        return args

    missing_flags: list[str] = []
    if "--single-turn" not in args:
        missing_flags.append("--single-turn")
    if use_gpu and gpu_layers > 0 and not any(flag in args for flag in LLAMA_GPU_FLAG_TOKENS):
        missing_flags.extend(["-ngl", str(gpu_layers)])
    if not missing_flags:
        return args
    return args[: llama_index + 1] + missing_flags + args[llama_index + 1 :]


def _is_shell_wrapper(args: list[str]) -> bool:
    return len(args) >= 3 and Path(args[0]).name in SHELL_WRAPPERS and args[1] in {"-c", "-lc"}


def _invocation_failed(process: subprocess.CompletedProcess[str]) -> bool:
    if process.returncode != 0:
        return True

    stderr = process.stderr.lower()
    if "unknown argument" in stderr or "error:" in stderr:
        return True

    return False


def _should_retry_without_gpu(process: subprocess.CompletedProcess[str]) -> bool:
    if process.returncode != 0:
        return False
    stderr = process.stderr.lower()
    if "unknown argument" not in stderr:
        return False
    return any(token in stderr for token in LLAMA_GPU_FLAG_TOKENS)


def _extract_model_text(stdout: str, stderr: str) -> str:
    raw_stdout = stdout or ""
    if raw_stdout != "":
        return raw_stdout
    raw_stderr = stderr or ""
    if raw_stderr != "":
        return _strip_runtime_logs(raw_stderr)
    return ""


def _strip_runtime_logs(text: str) -> str:
    if not text:
        return ""

    kept_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = ANSI_ESCAPE_RE.sub("", line).strip()
        if not stripped:
            kept_lines.append(line)
            continue
        if _is_runtime_log_line(stripped):
            continue
        kept_lines.append(line)

    cleaned = "".join(kept_lines)
    if cleaned.strip() == "":
        return ""
    return cleaned


def _is_runtime_log_line(stripped_line: str) -> bool:
    lower = stripped_line.lower()
    if lower.startswith("main:"):
        remainder = stripped_line.split(":", 1)[1].strip()
        if _looks_like_structured_content(remainder):
            return False
        return True

    if any(lower.startswith(prefix) for prefix in RUNTIME_LOG_PREFIXES):
        return True
    if lower.startswith("| memory breakdown"):
        return True
    if lower.startswith("|   - host"):
        return True
    if lower.startswith("|   - cpu_repack"):
        return True
    return False


def _looks_like_structured_content(text: str) -> bool:
    if not text:
        return False
    return bool(re.match(r"^(#{1,6}\s+\S+|[-*]\s+\S+|\d+\.\s+\S+|\|.+\|)", text))


def _looks_like_markdown_document(text: str) -> bool:
    if not text:
        return False
    heading_count = len(re.findall(r"(?m)^##?\s+\S+", text))
    return heading_count >= 2
