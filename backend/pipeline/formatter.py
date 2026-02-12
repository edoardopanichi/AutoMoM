from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline.template_manager import TEMPLATE_MANAGER


ACTION_OWNER_PATTERN = re.compile(r"(?P<owner>[A-Z][a-z]+) will (?P<task>[^.]+)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2}|tomorrow|next week|monday|tuesday|wednesday|thursday|friday)\b", re.IGNORECASE)


class Formatter:
    def __init__(self, command_template: str | None = None, model_path: str | None = None) -> None:
        self.command_template = command_template or ""
        self.model_path = model_path or ""
        self.last_mode: str = "heuristic"

    def run_model(self, prompt: str) -> dict[str, object] | None:
        if not self.command_template:
            self.last_mode = "heuristic_no_command"
            return None

        if not self.model_path or not Path(self.model_path).exists():
            self.last_mode = "heuristic_missing_model_path"
            return None

        command = self.command_template.format(model=self.model_path)
        args = shlex.split(command)
        try:
            process = subprocess.run(args, input=prompt, capture_output=True, text=True)
        except (OSError, ValueError):
            self.last_mode = "heuristic_command_error"
            return None

        if process.returncode != 0:
            self.last_mode = "heuristic_command_nonzero"
            return None

        output = process.stdout.strip()
        if not output:
            self.last_mode = "heuristic_empty_output"
            return None

        parsed = self._parse_json_output(output)
        if parsed is not None:
            self.last_mode = "model_json"
            return parsed

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
        model_output = self.run_model(prompt)
        if model_output is None:
            model_output = self._heuristic_structuring(transcript, speakers, title)
        elif "_raw_model_text" in model_output:
            model_output = self._merge_raw_text_with_template_structure(
                transcript=transcript,
                speakers=speakers,
                title=title,
                raw_text=str(model_output["_raw_model_text"]),
            )

        markdown = TEMPLATE_MANAGER.render(template_id, model_output)
        return markdown, model_output, prompt

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

    def _merge_raw_text_with_template_structure(
        self,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
        raw_text: str,
    ) -> dict[str, object]:
        structured = self._heuristic_structuring(transcript, speakers, title)
        cleaned = raw_text.strip()
        if cleaned:
            structured["executive_summary"] = cleaned[:2500]
            lines = [line.strip("- ").strip() for line in cleaned.splitlines() if line.strip()]
            if lines:
                structured["discussion_summary"] = lines[:8]
        return structured

    def _heuristic_structuring(
        self,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
    ) -> dict[str, object]:
        statements = [str(seg["text"]).strip() for seg in transcript if str(seg["text"]).strip()]
        statements = [item for item in statements if not item.startswith("[Offline fallback transcript")]

        if statements:
            summary = " ".join(statements[:3])
        else:
            summary = "Discussion captured from the meeting audio."

        discussion_summary = _discussion_bullets(transcript)
        decisions = _keyword_extract(statements, ["decide", "approved", "agreed", "resolution"])
        open_items = _keyword_extract(statements, ["risk", "issue", "question", "blocker"])[:8]
        actions = _extract_actions(statements)

        transcript_md = "\n".join(
            f"- [{seg['start_s']:.2f}-{seg['end_s']:.2f}] **{seg['speaker_name']}**: {seg['text']}"
            for seg in transcript
        )

        return {
            "title": title,
            "date_time": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "participants": speakers,
            "executive_summary": summary,
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
