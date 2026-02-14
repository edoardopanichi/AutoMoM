from __future__ import annotations

import json
import os
import re
import shlex
import select
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pty

from backend.app.config import SETTINGS
from backend.pipeline.compute import should_enable_native_gpu
from backend.pipeline.template_manager import TEMPLATE_MANAGER


ACTION_OWNER_PATTERN = re.compile(
    r"(?P<owner>[A-Z][a-z]+) will (?P<task>[^.]+)", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|tomorrow|next week|monday|tuesday|wednesday|thursday|friday)\b", re.IGNORECASE)
LLAMA_FLAGS = ("--single-turn",)
LLAMA_GPU_FLAG_TOKENS = ("-ngl", "--gpu-layers", "--n-gpu-layers")
LLAMA_BINARY_PATTERN = re.compile(
    r"(^|[\s;|&()])(?P<cmd>(?:[^\s;|&()]+/)?llama-cli)(?=$|[\s;|&()])")
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
MAIN_RUNTIME_REMAINDER_PREFIXES = (
    "build =",
    "build:",
    "seed =",
    "seed:",
    "llama backend init",
    "load the model and apply lora adapter",
    "n_ctx =",
    "n_batch =",
    "n_ubatch =",
    "n_predict =",
    "n_keep =",
    "n_parallel =",
    "n_examples =",
    "interactive mode",
    "sampling",
    "available commands:",
    "/exit",
    "/regen",
    "/clear",
    "/read",
    ">",
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
        self._gpu_requested = should_enable_native_gpu(
            compute_device, self._cuda_device_id)
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
        use_pty = _is_shell_wrapper(args)
        try:
            process = _run_command(
                args=args,
                prompt=prompt,
                timeout_s=SETTINGS.formatter_timeout_s,
                use_pty=use_pty,
            )
            print(f"\n\n Debug: if working the stdout should be here: {process.stdout} \n\n")
            output = _extract_model_text(process.stdout, process.stderr, prompt=prompt)
            invocation_failed = _invocation_failed(process)
            if use_gpu and _should_retry_without_gpu(process):
                retry_args = _ensure_non_interactive_llama_command(
                    shlex.split(command),
                    use_gpu=False,
                    gpu_layers=self._gpu_layers,
                )
                retry_process = _run_command(
                    args=retry_args,
                    prompt=prompt,
                    timeout_s=SETTINGS.formatter_timeout_s,
                    use_pty=use_pty,
                )
                retry_output = _extract_model_text(retry_process.stdout, retry_process.stderr, prompt=prompt)
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
        except subprocess.TimeoutExpired as exc:
            timeout_stdout = _decode_timeout_stream(exc.stdout)
            timeout_stderr = _decode_timeout_stream(exc.stderr)
            output = _extract_model_text(timeout_stdout, timeout_stderr, prompt=prompt)
            self.last_stdout = timeout_stdout
            self.last_stderr = timeout_stderr
            if output:
                self.last_raw_output = output
                parsed = self._parse_json_output(output)
                if parsed is not None:
                    self.last_mode = "model_json"
                    return parsed
                if _looks_like_markdown_document(output):
                    self.last_mode = "model_markdown"
                    return {"_raw_markdown_text": output}
                self.last_mode = "model_plaintext"
                return {"_raw_model_text": output}
            self.last_raw_output = ""
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
        prompt = TEMPLATE_MANAGER.build_formatter_prompt(
            template_id, transcript, speakers, title)
        self.run_model(prompt)
        structured = self._heuristic_structuring(transcript, speakers, title)
        markdown = self.last_raw_output if self.last_raw_output else ""
        return markdown, structured, prompt

    def write_model_output_to_mom(
        self,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
        template_id: str,
        output_path: Path,
    ) -> tuple[dict[str, object], str]:
        markdown, structured, prompt = self.build_structured_summary(
            transcript=transcript,
            speakers=speakers,
            title=title,
            template_id=template_id,
        )
        if not markdown.strip():
            raise RuntimeError(_formatter_failure_message(self.last_mode, self.model_path, self.command_template))
        output_path.write_text(markdown, encoding="utf-8")
        return structured, prompt

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
        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", output, flags=re.DOTALL | re.IGNORECASE)
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
        statements = [str(seg["text"]).strip()
                      for seg in transcript if str(seg["text"]).strip()]
        statements = [item for item in statements if not item.startswith(
            "[Offline fallback transcript")]

        discussion_summary = _discussion_bullets(transcript)
        decisions = _keyword_extract(
            statements, ["decide", "approved", "agreed", "resolution"])
        open_items = _keyword_extract(
            statements, ["risk", "issue", "question", "blocker"])[:8]
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

        updated_script, replacements = LLAMA_BINARY_PATTERN.subn(
            _inject, script, count=1)
        if replacements == 0:
            return args
        updated = args.copy()
        updated[2] = updated_script
        return updated

    llama_index = next((idx for idx, token in enumerate(
        args) if Path(token).name == "llama-cli"), None)
    if llama_index is None:
        return args

    missing_flags: list[str] = []
    if "--single-turn" not in args:
        missing_flags.append("--single-turn")
    if use_gpu and gpu_layers > 0 and not any(flag in args for flag in LLAMA_GPU_FLAG_TOKENS):
        missing_flags.extend(["-ngl", str(gpu_layers)])
    if not missing_flags:
        return args
    return args[: llama_index + 1] + missing_flags + args[llama_index + 1:]


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
    logs = f"{process.stdout}\n{process.stderr}".lower()
    if "unknown argument" not in logs:
        return False
    return any(token in logs for token in LLAMA_GPU_FLAG_TOKENS)


def _extract_model_text(stdout: str, stderr: str, *, prompt: str = "") -> str:
    raw_stdout = stdout or ""
    raw_stderr = stderr or ""
    cleaned_stderr = _strip_runtime_logs(raw_stderr) if raw_stderr else ""
    if raw_stdout != "":
        model_stdout = _extract_llama_generated_text(raw_stdout, prompt=prompt)
        cleaned_stdout = _strip_runtime_logs(model_stdout)
        if cleaned_stdout:
            return cleaned_stdout
        if cleaned_stderr:
            return cleaned_stderr
        if _contains_only_runtime_logs(model_stdout):
            return ""
        return ""
    if raw_stderr != "":
        return cleaned_stderr
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
        remainder_lower = remainder.lower()
        if not remainder:
            return True
        if any(remainder_lower.startswith(prefix) for prefix in MAIN_RUNTIME_REMAINDER_PREFIXES):
            return True
        if any(remainder_lower.startswith(prefix) for prefix in RUNTIME_LOG_PREFIXES):
            return True
        if _looks_like_structured_content(remainder):
            return False
        if _looks_like_narrative_content(remainder):
            return False
        return True

    if any(lower.startswith(prefix) for prefix in RUNTIME_LOG_PREFIXES):
        return True
    if _is_memory_breakdown_table_line(lower):
        return True
    return False


def _looks_like_structured_content(text: str) -> bool:
    if not text:
        return False
    return bool(re.match(r"^(#{1,6}\s+\S+|[-*]\s+\S+|\d+\.\s+\S+|\|.+\|)", text))


def _looks_like_narrative_content(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("prompt:") or lowered.startswith("generation:"):
        return False
    if "=" in text and len(text.split()) <= 5:
        return False
    return bool(re.search(r"[A-Za-z]{3,}", text)) and (" " in text or text.endswith("."))


def _looks_like_markdown_document(text: str) -> bool:
    if not text:
        return False
    heading_count = len(re.findall(r"(?m)^##?\s+\S+", text))
    return heading_count >= 2


def _extract_llama_generated_text(text: str, *, prompt: str = "") -> str:
    if not text:
        print("\n\n Debug: Empty text received for model output extraction. \n\n")
        return ""

    normalized = _normalize_terminal_output(text)
    model_text = normalized

    prompt_marker = "\n> "
    if prompt:
        prompt_idx = model_text.rfind(prompt)
        if prompt_idx >= 0:
            model_text = model_text[prompt_idx + len(prompt):]
            model_text = _drop_leading_spinner_lines(model_text)
        else:
            marker_idx = model_text.rfind(prompt_marker)
            if marker_idx >= 0:
                model_text = model_text[marker_idx + len(prompt_marker):]
                truncated_idx = model_text.find("... (truncated)")
                if truncated_idx >= 0:
                    truncated_end = model_text.find("\n", truncated_idx)
                    if truncated_end >= 0:
                        model_text = model_text[truncated_end + 1:]
                else:
                    prompt_end = model_text.find("\n")
                    if prompt_end >= 0:
                        model_text = model_text[prompt_end + 1:]
                model_text = _drop_leading_spinner_lines(model_text)
    else:
        marker_idx = model_text.rfind(prompt_marker)
        if marker_idx >= 0:
            prompt_end = model_text.find("\n", marker_idx + len(prompt_marker))
            if prompt_end >= 0:
                model_text = model_text[prompt_end + 1:]
                model_text = _drop_leading_spinner_lines(model_text)

    timing_match = re.search(r"(?m)^\[\s*Prompt:.*$", model_text)
    if timing_match:
        model_text = model_text[: timing_match.start()]

    exiting_match = re.search(r"(?m)^Exiting\.\.\.\s*$", model_text)
    if exiting_match:
        model_text = model_text[: exiting_match.start()]

    return model_text


def _normalize_terminal_output(text: str) -> str:
    if not text:
        return ""

    normalized = ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")
    out_chars: list[str] = []
    for char in normalized:
        if char == "\x08":
            if out_chars:
                out_chars.pop()
            continue
        out_chars.append(char)
    return "".join(out_chars)


def _drop_leading_spinner_lines(text: str) -> str:
    lines = text.splitlines(keepends=True)
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if all(ch in {"|", "/", "-", "\\", "."} for ch in stripped):
            lines.pop(0)
            continue
        break
    return "".join(lines)


def _is_memory_breakdown_table_line(lower_line: str) -> bool:
    if not lower_line.startswith("|"):
        return False
    if "memory breakdown" in lower_line:
        return True
    if all(token in lower_line for token in ("total", "model", "context", "compute", "unaccounted")):
        return True
    if lower_line.startswith("|   - "):
        if any(token in lower_line for token in ("host", "cpu", "cuda", "gpu", "metal", "vulkan", "hip", "opencl", "sycl")):
            return True
    if "=" in lower_line and any(token in lower_line for token in ("model", "context", "compute", "unaccounted", "self", "free", "total")):
        return True
    return False


def _contains_only_runtime_logs(text: str) -> bool:
    saw_content = False
    for line in text.splitlines():
        stripped = ANSI_ESCAPE_RE.sub("", line).strip()
        if not stripped:
            continue
        saw_content = True
        if not _is_runtime_log_line(stripped):
            return False
    return saw_content


def _formatter_failure_message(last_mode: str, model_path: str, command_template: str) -> str:
    if last_mode == "heuristic_no_command":
        return (
            "Formatter command is not configured. "
            "Fix: set AUTOMOM_FORMATTER_COMMAND to a runnable command that reads prompt from stdin."
        )
    if last_mode == "heuristic_missing_model_path":
        return (
            "Formatter model file is missing. "
            f"Expected at '{model_path or '<unset>'}'. "
            "Fix: set AUTOMOM_FORMATTER_MODEL to an existing local model file."
        )
    if last_mode == "heuristic_command_timeout":
        return (
            "Formatter command timed out. "
            "Fix: verify model runtime performance or increase AUTOMOM_FORMATTER_TIMEOUT_S."
        )
    if last_mode == "heuristic_command_error":
        return (
            "Formatter command could not be started. "
            f"Command template: '{command_template or '<unset>'}'. "
            "Fix: verify the executable is installed and command syntax is valid."
        )
    if last_mode == "heuristic_command_nonzero":
        return (
            "Formatter command failed (non-zero exit). "
            "Fix: inspect formatter stderr artifact and verify binary/model compatibility."
        )
    if last_mode == "heuristic_empty_output":
        return (
            "Formatter produced empty output. "
            "Fix: verify the formatter command, model file, and runtime compatibility."
        )
    return (
        f"Formatter output unavailable (mode={last_mode}). "
        "Fix: verify formatter command and model configuration."
    )


def _run_with_pty_capture(
    args: list[str],
    *,
    prompt: str,
    timeout_s: int,
) -> subprocess.CompletedProcess[str] | None:
    master_fd: int | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)
        if process.stdin is not None:
            try:
                process.stdin.write(prompt.encode("utf-8", errors="replace"))
                process.stdin.close()
            except OSError:
                pass

        deadline = time.monotonic() + float(timeout_s)
        chunks: list[bytes] = []
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining == 0.0 and process.poll() is None:
                process.kill()
                raise subprocess.TimeoutExpired(cmd=args, timeout=timeout_s)

            ready, _, _ = select.select(
                [master_fd], [], [], min(0.1, remaining))
            if master_fd in ready:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""
                if data:
                    chunks.append(data)
                elif process.poll() is not None:
                    break

            if process.poll() is not None and master_fd not in ready:
                break

        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                data = b""
            if not data:
                break
            chunks.append(data)

        return subprocess.CompletedProcess(
            args=args,
            returncode=process.wait(),
            stdout=b"".join(chunks).decode("utf-8", errors="replace"),
            stderr="",
        )
    except OSError:
        return None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass


def _run_command(
    *,
    args: list[str],
    prompt: str,
    timeout_s: int,
    use_pty: bool,
) -> subprocess.CompletedProcess[str]:
    if use_pty:
        process = _run_with_pty_capture(args, prompt=prompt, timeout_s=timeout_s)
        if process is not None:
            return process
    return subprocess.run(
        args,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _decode_timeout_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream
