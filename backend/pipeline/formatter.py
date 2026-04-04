from __future__ import annotations

import json
import re
import socket
import shlex
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import SETTINGS
from backend.pipeline.openai_client import OpenAIAPIError, OpenAIClient
from backend.pipeline.template_manager import TEMPLATE_MANAGER

ACTION_OWNER_PATTERN = re.compile(r"(?P<owner>[A-Z][a-z]+) will (?P<task>[^.]+)", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|tomorrow|next week|monday|tuesday|wednesday|thursday|friday)\b",
    re.IGNORECASE,
)
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class Formatter:
    def __init__(
        self,
        command_template: str | None = None,
        model_path: str | None = None,
        ollama_host: str | None = None,
        ollama_model: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        *,
        timeout_s: int | None = None,
        **_: object,
    ) -> None:
        self.command_template = command_template or ""
        self.model_path = model_path or ""
        self.ollama_host = (ollama_host or SETTINGS.ollama_host).rstrip("/")
        self.ollama_model = (ollama_model or SETTINGS.formatter_ollama_model).strip()
        self.openai_api_key = (openai_api_key or "").strip()
        self.openai_model = (openai_model or "").strip()
        self.timeout_s = int(timeout_s or SETTINGS.formatter_timeout_s)
        self.last_raw_output: str = ""
        self.last_stdout: str = ""
        self.last_stderr: str = ""
        self.last_mode: str = "heuristic"

    def run_model(self, prompt: str) -> dict[str, object] | None:
        if self.openai_api_key:
            return self._run_openai_response(prompt)
        if self.command_template:
            return self._run_legacy_command(prompt)
        if not self.ollama_model:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_no_model"
            return None

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        request = urllib.request.Request(
            url=f"{self.ollama_host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            self.last_mode = "heuristic_ollama_http_error"
            return None
        except TimeoutError as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            self.last_mode = "heuristic_ollama_timeout"
            return None
        except urllib.error.URLError as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError) or isinstance(reason, socket.timeout):
                self.last_mode = "heuristic_ollama_timeout"
            else:
                self.last_mode = "heuristic_ollama_unavailable"
            return None
        except Exception as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            self.last_mode = "heuristic_ollama_unavailable"
            return None

        self.last_stdout = body
        self.last_stderr = ""

        try:
            parsed_response = json.loads(body)
        except json.JSONDecodeError:
            self.last_raw_output = ""
            self.last_mode = "heuristic_ollama_invalid_json"
            return None

        output = _extract_model_text(str(parsed_response.get("response", "")), "")
        if not output:
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

        self.last_mode = "model_plaintext"
        return {"_raw_model_text": output}

    def _run_openai_response(self, prompt: str) -> dict[str, object] | None:
        try:
            client = OpenAIClient(self.openai_api_key)
            output = client.generate_text(prompt, model=self.openai_model or "gpt-5-mini", timeout_s=self.timeout_s)
        except (OpenAIAPIError, ValueError) as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            self.last_mode = "heuristic_openai_error"
            return None

        self.last_stdout = output
        self.last_stderr = ""
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

    def _run_legacy_command(self, prompt: str) -> dict[str, object] | None:
        if not self.model_path or not Path(self.model_path).exists():
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_missing_model_path"
            return None
        command = self.command_template.format(model=self.model_path)
        try:
            process = subprocess.run(
                shlex.split(command),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except Exception as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            self.last_mode = "heuristic_command_error"
            return None
        self.last_stdout = process.stdout or ""
        self.last_stderr = process.stderr or ""
        output = _extract_model_text(self.last_stdout, self.last_stderr)
        if not output:
            self.last_mode = "heuristic_command_nonzero" if process.returncode != 0 else "heuristic_empty_output"
            self.last_raw_output = ""
            return None
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
            raise RuntimeError(_formatter_failure_message(self.last_mode, self.ollama_model, self.ollama_host))
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

        transcript_md = "\n".join(f"- **{seg['speaker_name']}**: {seg['text']}" for seg in transcript)

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
        if counter[speaker] > 1:
            bullets.append(f"{speaker}: {text}")
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
        actions.append({"owner": match.group("owner"), "task": match.group("task").strip(), "due_date": due_date})
    return actions[:12]


def _looks_like_markdown_document(text: str) -> bool:
    if not text:
        return False
    heading_count = len(re.findall(r"(?m)^##?\s+\S+", text))
    return heading_count >= 2


def _extract_model_text(stdout: str, stderr: str, *, prompt: str = "") -> str:
    _ = prompt
    raw = stdout if stdout.strip() else stderr
    return _strip_runtime_logs(raw)


def _strip_runtime_logs(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = ANSI_ESCAPE_RE.sub("", line).strip()
        if stripped.startswith("time=") or stripped.startswith("load="):
            continue
        lines.append(line)
    cleaned = "".join(lines)
    if not cleaned.strip():
        return ""
    return cleaned


def _formatter_failure_message(last_mode: str, model_name: str, ollama_host: str) -> str:
    if last_mode == "heuristic_no_model":
        return "Formatter model is not configured. Fix: set AUTOMOM_FORMATTER_OLLAMA_MODEL."
    if last_mode == "heuristic_ollama_http_error":
        return (
            f"Ollama rejected formatter request for model '{model_name}'. "
            "Fix: verify model name and ensure it is pulled."
        )
    if last_mode == "heuristic_ollama_unavailable":
        return (
            f"Ollama is unavailable at '{ollama_host}'. "
            "Fix: start Ollama service and verify AUTOMOM_OLLAMA_HOST."
        )
    if last_mode == "heuristic_ollama_timeout":
        return (
            "Formatter request to Ollama timed out. "
            "Fix: use a smaller model, shorten transcript, or increase AUTOMOM_FORMATTER_TIMEOUT_S."
        )
    if last_mode == "heuristic_ollama_invalid_json":
        return "Ollama returned invalid JSON response. Fix: verify Ollama health and server logs."
    if last_mode == "heuristic_empty_output":
        return "Formatter produced empty output. Fix: try another Ollama model or adjust prompt/template."
    if last_mode == "heuristic_openai_error":
        return "OpenAI formatter request failed. Fix: verify the API key, selected model, and network access."
    return f"Formatter output unavailable (mode={last_mode})."
