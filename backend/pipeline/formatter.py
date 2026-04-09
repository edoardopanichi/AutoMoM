from __future__ import annotations

import json
import re
import socket
import shlex
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import SETTINGS
from backend.app.schemas import TemplateSection
from backend.pipeline.openai_client import OpenAIAPIError, OpenAIClient
from backend.pipeline.subprocess_utils import run_cancellable_subprocess
from backend.pipeline.template_manager import TEMPLATE_MANAGER, FormatterPromptBundle

ACTION_OWNER_PATTERN = re.compile(r"(?P<owner>[A-Z][a-z]+) will (?P<task>[^.]+)", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|tomorrow|next week|monday|tuesday|wednesday|thursday|friday)\b",
    re.IGNORECASE,
)
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
HEADING_RE = re.compile(r"(?m)^(#{2,4}\s+.+)$")
FORMATTER_LONG_INPUT_TOKEN_LIMIT = 12000
FORMATTER_CHUNK_TOKEN_TARGET = 3000
FORMATTER_MAX_ATTEMPTS = 3
DEFAULT_SYSTEM_PROMPT = (
    "Write concise, professional markdown minutes of meeting in English. "
    "Return only the final markdown document."
)


@dataclass(frozen=True)
class FormatterAttempt:
    attempt: int
    mode: str
    valid: bool
    errors: list[str]
    output_preview: str


@dataclass(frozen=True)
class FormatterRunResult:
    markdown: str
    structured: dict[str, object]
    system_prompt: str
    user_prompt: str
    validation: dict[str, object]
    reduced_notes: list[dict[str, object]]


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
        job_id: str | None = None,
        timeout_s: int | None = None,
        **_: object,
    ) -> None:
        """! @brief Initialize the Formatter instance.
        @param command_template Value for command template.
        @param model_path Value for model path.
        @param ollama_host Value for ollama host.
        @param ollama_model Value for ollama model.
        @param openai_api_key Value for openai api key.
        @param openai_model Value for openai model.
        @param job_id Identifier of the job being processed.
        @param timeout_s Timeout in seconds.
        @param _ Value for _.
        """
        self.command_template = command_template or ""
        self.model_path = model_path or ""
        self.ollama_host = (ollama_host or SETTINGS.ollama_host).rstrip("/")
        self.ollama_model = (ollama_model or SETTINGS.formatter_ollama_model).strip()
        self.openai_api_key = (openai_api_key or "").strip()
        self.openai_model = (openai_model or "").strip()
        self.job_id = job_id
        self.timeout_s = int(timeout_s or SETTINGS.formatter_timeout_s)
        self.last_raw_output: str = ""
        self.last_stdout: str = ""
        self.last_stderr: str = ""
        self.last_mode: str = "heuristic"

    def run_model(self, prompt: str, *, system_prompt: str = "") -> dict[str, object] | None:
        """! @brief Run model.
        @param prompt Value for prompt.
        @param system_prompt Value for system prompt.
        @return Dictionary produced by the operation.
        """
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        if self.openai_api_key:
            return self._run_openai_response(prompt, system_prompt=system_prompt)
        if self.command_template:
            return self._run_legacy_command(prompt, system_prompt=system_prompt)
        if not self.ollama_model:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_no_model"
            return None

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
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
        return self._finalize_model_output(output)

    def _run_openai_response(self, prompt: str, *, system_prompt: str = "") -> dict[str, object] | None:
        """! @brief Run openai response.
        @param prompt Value for prompt.
        @param system_prompt Value for system prompt.
        @return Dictionary produced by the operation.
        """
        try:
            client = OpenAIClient(self.openai_api_key)
            output = client.generate_text(
                prompt,
                model=self.openai_model or "gpt-5-mini",
                timeout_s=self.timeout_s,
                instructions=system_prompt,
            )
        except (OpenAIAPIError, ValueError) as exc:
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = str(exc)
            self.last_mode = "heuristic_openai_error"
            return None

        self.last_stdout = output
        self.last_stderr = ""
        return self._finalize_model_output(output)

    def _run_legacy_command(self, prompt: str, *, system_prompt: str = "") -> dict[str, object] | None:
        """! @brief Run legacy command.
        @param prompt Value for prompt.
        @param system_prompt Value for system prompt.
        @return Dictionary produced by the operation.
        """
        if not self.model_path or not Path(self.model_path).exists():
            self.last_raw_output = ""
            self.last_stdout = ""
            self.last_stderr = ""
            self.last_mode = "heuristic_missing_model_path"
            return None
        command = self.command_template.format(model=self.model_path)
        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{prompt}" if system_prompt else prompt
        try:
            process = run_cancellable_subprocess(
                shlex.split(command),
                input_text=full_prompt,
                job_id=self.job_id,
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
        return self._finalize_model_output(output)

    def _finalize_model_output(self, output: str) -> dict[str, object] | None:
        """! @brief Finalize model output.
        @param output Value for output.
        @return Dictionary produced by the operation.
        """
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

    def write_model_output_to_mom(
        self,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
        template_id: str,
        output_path: Path,
    ) -> FormatterRunResult:
        """! @brief Write model output to MoM.
        @param transcript Transcript segments used by the operation.
        @param speakers Speaker names available for the meeting.
        @param title Meeting title associated with the request.
        @param template_id Identifier of the template.
        @param output_path Path to the output file.
        @return Result produced by the operation.
        """
        bundle = TEMPLATE_MANAGER.build_formatter_request(template_id, transcript, speakers, title)
        estimated_tokens = _estimate_tokens(f"{bundle.system_prompt}\n\n{bundle.user_prompt}")
        reduced_notes = _reduce_transcript_notes(transcript, token_target=FORMATTER_CHUNK_TOKEN_TARGET)
        # Strict section validation gets brittle on very large transcripts, so reduce the transcript
        # into structured notes before prompting once the estimated payload crosses the long-input cap.
        reduction_used = bool(bundle.strict_sections and estimated_tokens > FORMATTER_LONG_INPUT_TOKEN_LIMIT)
        active_bundle = (
            TEMPLATE_MANAGER.build_formatter_request(
                template_id,
                _render_reduced_notes_as_transcript(reduced_notes),
                speakers,
                title,
                transcript_label="Structured notes",
            )
            if reduction_used
            else bundle
        )

        attempts: list[FormatterAttempt] = []
        markdown = ""
        validation_result = {"valid": True, "errors": [], "mode": "compatibility"}
        corrective_feedback = ""
        for attempt in range(1, FORMATTER_MAX_ATTEMPTS + 1):
            attempt_user_prompt = active_bundle.user_prompt
            if corrective_feedback:
                # Feed validator errors back into the next attempt so the model can repair heading
                # order and missing sections without changing the transcript content itself.
                attempt_user_prompt = f"{attempt_user_prompt}\n\nCorrection request:\n{corrective_feedback}\n"
            result = self.run_model(attempt_user_prompt, system_prompt=active_bundle.system_prompt)
            if result is None:
                continue
            markdown = (
                str(result.get("_raw_markdown_text") or result.get("_raw_model_text") or self.last_raw_output or "").strip()
            )
            validation_result = validate_markdown_output(markdown, active_bundle.strict_sections)
            attempts.append(
                FormatterAttempt(
                    attempt=attempt,
                    mode=self.last_mode,
                    valid=bool(validation_result["valid"]),
                    errors=list(validation_result["errors"]),
                    output_preview=markdown[:300],
                )
            )
            if validation_result["valid"]:
                break
            corrective_feedback = (
                "Your previous answer did not follow the required template.\n"
                + "\n".join(f"- {error}" for error in validation_result["errors"])
                + "\nReturn the full document again, with corrected headings and order."
            )

        if not markdown.strip():
            raise RuntimeError(_formatter_failure_message(self.last_mode, self.ollama_model, self.ollama_host))
        if active_bundle.strict_sections and not validation_result["valid"]:
            raise RuntimeError(
                "Formatter output did not follow the required template after retries: "
                + "; ".join(validation_result["errors"])
            )

        structured = self._heuristic_structuring(transcript, speakers, title)
        output_path.write_text(markdown, encoding="utf-8")
        validation_payload = {
            "valid": validation_result["valid"],
            "errors": validation_result["errors"],
            "attempts": [attempt.__dict__ for attempt in attempts],
            "strict_template": bool(active_bundle.strict_sections),
            "estimated_tokens": estimated_tokens,
            "reduction_used": reduction_used,
        }
        return FormatterRunResult(
            markdown=markdown,
            structured=structured,
            system_prompt=active_bundle.system_prompt,
            user_prompt=active_bundle.user_prompt,
            validation=validation_payload,
            reduced_notes=reduced_notes if reduction_used else [],
        )

    @staticmethod
    def _parse_json_output(output: str) -> dict[str, object] | None:
        """! @brief Parse json output.
        @param output Value for output.
        @return Dictionary produced by the operation.
        """
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
        """! @brief Heuristic structuring.
        @param transcript Transcript segments used by the operation.
        @param speakers Speaker names available for the meeting.
        @param title Meeting title associated with the request.
        @return Dictionary produced by the operation.
        """
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


def validate_markdown_output(markdown: str, sections: list[TemplateSection]) -> dict[str, object]:
    """! @brief Validate markdown output.
    @param markdown Value for markdown.
    @param sections Value for sections.
    @return Dictionary produced by the operation.
    """
    if not sections:
        return {"valid": True, "errors": [], "headings": []}
    headings = HEADING_RE.findall(markdown)
    blocks = _markdown_heading_blocks(markdown)
    errors: list[str] = []
    last_index = -1
    for section in sections:
        match_index = -1
        for idx, heading in enumerate(headings):
            normalized = heading.strip()
            if section.allow_prefix:
                if normalized.startswith(section.heading):
                    match_index = idx
                    break
            elif normalized == section.heading:
                match_index = idx
                break
        if match_index == -1:
            if section.required:
                errors.append(f"Missing required heading '{section.heading}'")
            continue
        if match_index < last_index:
            errors.append(f"Heading '{section.heading}' is out of order")
        last_index = match_index
        block_body = blocks[match_index][1].strip() if match_index < len(blocks) else ""
        heading_has_inline_content = section.allow_prefix and headings[match_index].strip() != section.heading
        if section.required and not block_body and not heading_has_inline_content:
            errors.append(f"Heading '{section.heading}' has no content; use '{section.empty_value}' when empty")

    required_count = len(sections)
    if len(headings) < required_count:
        errors.append("Output contains fewer headings than required by the template")
    if len(headings) > required_count:
        errors.append("Output contains extra headings outside the template")
    return {"valid": not errors, "errors": errors, "headings": headings}


def _markdown_heading_blocks(markdown: str) -> list[tuple[str, str]]:
    """! @brief Markdown heading blocks.
    @param markdown Value for markdown.
    @return List produced by the operation.
    """
    matches = list(re.finditer(r"(?m)^(#{2,4}\s+.+)$", markdown))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        blocks.append((match.group(1), markdown[start:end]))
    return blocks


def _render_reduced_notes_as_transcript(notes: list[dict[str, object]]) -> list[dict[str, object]]:
    """! @brief Render reduced notes as transcript.
    @param notes Value for notes.
    @return List produced by the operation.
    """
    rendered: list[dict[str, object]] = []
    for idx, note in enumerate(notes, start=1):
        lines = [
            f"Participants: {', '.join(note['participants']) if note['participants'] else 'None'}",
            f"Overview: {' | '.join(note['overview']) if note['overview'] else 'None'}",
            f"TODOs: {' | '.join(note['todos']) if note['todos'] else 'None'}",
            f"Conclusions: {' | '.join(note['conclusions']) if note['conclusions'] else 'None'}",
            f"Open points: {' | '.join(note['open_points']) if note['open_points'] else 'None'}",
            f"Risks: {' | '.join(note['risks']) if note['risks'] else 'None'}",
        ]
        rendered.append(
            {
                "speaker_name": f"Chunk {idx}",
                "text": "\n".join(lines),
                "start_s": float(idx),
                "end_s": float(idx),
            }
        )
    return rendered


def _estimate_tokens(text: str) -> int:
    """! @brief Estimate tokens.
    @param text Value for text.
    @return int result produced by the operation.
    """
    return max(1, len(text) // 4)


def _chunk_transcript(transcript: list[dict[str, object]], token_target: int) -> list[list[dict[str, object]]]:
    """! @brief Chunk transcript.
    @param transcript Transcript segments used by the operation.
    @param token_target Value for token target.
    @return List produced by the operation.
    """
    chunks: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_tokens = 0
    for segment in transcript:
        seg_tokens = _estimate_tokens(str(segment.get("text") or "")) + 16
        if current and current_tokens + seg_tokens > token_target:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(segment)
        current_tokens += seg_tokens
    if current:
        chunks.append(current)
    return chunks


def _reduce_transcript_notes(transcript: list[dict[str, object]], token_target: int) -> list[dict[str, object]]:
    """! @brief Reduce transcript notes.
    @param transcript Transcript segments used by the operation.
    @param token_target Value for token target.
    @return List produced by the operation.
    """
    notes: list[dict[str, object]] = []
    for chunk in _chunk_transcript(transcript, token_target):
        statements = [str(seg["text"]).strip() for seg in chunk if str(seg.get("text") or "").strip()]
        notes.append(
            {
                "participants": sorted({str(seg["speaker_name"]) for seg in chunk if str(seg.get("speaker_name") or "").strip()}),
                "overview": _discussion_bullets(chunk, top_n=4),
                "todos": [f"{item['owner']}: {item['task']} ({item['due_date']})" for item in _extract_actions(statements)[:6]],
                "conclusions": _keyword_extract(statements, ["decide", "approved", "agreed", "conclusion", "resolved"])[:6],
                "open_points": _keyword_extract(statements, ["open", "pending", "question", "follow up", "blocker"])[:6],
                "risks": _keyword_extract(statements, ["risk", "concern", "delay", "issue"])[:6],
            }
        )
    return notes


def _discussion_bullets(transcript: list[dict[str, object]], top_n: int = 8) -> list[str]:
    """! @brief Discussion bullets.
    @param transcript Transcript segments used by the operation.
    @param top_n Value for top n.
    @return List produced by the operation.
    """
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
    """! @brief Keyword extract.
    @param statements Value for statements.
    @param keywords Value for keywords.
    @return List produced by the operation.
    """
    found: list[str] = []
    for text in statements:
        lower = text.lower()
        if any(keyword in lower for keyword in keywords):
            found.append(text)
    return found[:10]


def _extract_actions(statements: list[str]) -> list[dict[str, str]]:
    """! @brief Extract actions.
    @param statements Value for statements.
    @return List produced by the operation.
    """
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
    """! @brief Looks like markdown document.
    @param text Value for text.
    @return True when the requested condition is satisfied; otherwise False.
    """
    if not text:
        return False
    heading_count = len(re.findall(r"(?m)^##?\s+\S+", text))
    return heading_count >= 2


def _extract_model_text(stdout: str, stderr: str, *, prompt: str = "") -> str:
    """! @brief Extract model text.
    @param stdout Value for stdout.
    @param stderr Value for stderr.
    @param prompt Value for prompt.
    @return str result produced by the operation.
    """
    _ = prompt
    raw = stdout if stdout.strip() else stderr
    return _strip_runtime_logs(raw)


def _strip_runtime_logs(text: str) -> str:
    """! @brief Strip runtime logs.
    @param text Value for text.
    @return str result produced by the operation.
    """
    if not text:
        return ""
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = ANSI_ESCAPE_RE.sub("", line).strip()
        if stripped.startswith("time=") or stripped.startswith("load="):
            continue
        normalized_line = re.sub(r"^\s*main:\s*", "", line)
        stripped = ANSI_ESCAPE_RE.sub("", normalized_line).strip()
        if stripped.startswith("error:") and "simulated nonzero exit" in stripped:
            continue
        lines.append(normalized_line)
    cleaned = "".join(lines)
    if not cleaned.strip():
        return ""
    return cleaned


def _formatter_failure_message(last_mode: str, model_name: str, ollama_host: str) -> str:
    """! @brief Formatter failure message.
    @param last_mode Value for last mode.
    @param model_name Value for model name.
    @param ollama_host Value for ollama host.
    @return str result produced by the operation.
    """
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
