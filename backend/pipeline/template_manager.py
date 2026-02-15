from __future__ import annotations

import json
import re
from pathlib import Path

from backend.app.config import SETTINGS
from backend.app.schemas import TemplateDefinition, TemplateSummary


DEFAULT_TEMPLATE_ID = "default"
DEFAULT_TEMPLATE_NAME = "Default Minutes Template"
TEMPLATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_TEMPLATE_PROMPT = (
    "You are an assistant that writes concise Minutes of Meeting in English. "
    "Use the transcript and speaker list. Return clear bullets, decisions, action items, and risks."
)


class TemplateManager:
    def __init__(self) -> None:
        self._ensure_default_template()

    @staticmethod
    def _validate_template_id(template_id: str) -> str:
        normalized = (template_id or "").strip()
        if not TEMPLATE_ID_PATTERN.fullmatch(normalized):
            raise ValueError("Invalid template_id. Use 1-64 chars: letters, numbers, '.', '_' or '-'")
        return normalized

    @staticmethod
    def _template_meta_path(template_id: str) -> Path:
        safe_id = TemplateManager._validate_template_id(template_id)
        return SETTINGS.templates_dir / f"{safe_id}.json"

    def _ensure_default_template(self) -> None:
        if self._template_meta_path(DEFAULT_TEMPLATE_ID).exists():
            return
        self.save(
            TemplateDefinition(
                template_id=DEFAULT_TEMPLATE_ID,
                name=DEFAULT_TEMPLATE_NAME,
                version="1.0.0",
                description="Default AutoMoM formatter prompt template.",
                prompt_block=DEFAULT_TEMPLATE_PROMPT,
            )
        )

    def list_templates(self) -> list[TemplateSummary]:
        result: list[TemplateSummary] = []
        for path in sorted(SETTINGS.templates_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            result.append(
                TemplateSummary(
                    template_id=payload["template_id"],
                    name=payload["name"],
                    version=payload["version"],
                    description=payload.get("description", ""),
                )
            )
        return result

    def load(self, template_id: str) -> TemplateDefinition:
        meta_path = self._template_meta_path(template_id)
        if not meta_path.exists():
            raise FileNotFoundError(template_id)

        return TemplateDefinition(**json.loads(meta_path.read_text(encoding="utf-8")))

    def save(self, definition: TemplateDefinition) -> None:
        meta_path = self._template_meta_path(definition.template_id)
        SETTINGS.templates_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(definition.model_dump(), indent=2), encoding="utf-8")

    def delete(self, template_id: str) -> None:
        if template_id == DEFAULT_TEMPLATE_ID:
            raise ValueError("Default template cannot be deleted")
        meta_path = self._template_meta_path(template_id)
        if not meta_path.exists():
            raise FileNotFoundError(template_id)
        meta_path.unlink(missing_ok=True)

    def build_formatter_prompt(
        self,
        template_id: str,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
    ) -> str:
        template = self.load(template_id)
        transcript_text = "\n".join(f"{seg['speaker_name']}: {seg['text']}" for seg in transcript)
        prompt = (
            f"{template.prompt_block}\n\n"
            "Requirements:\n"
            "- Final output must be in English.\n"
            "- Capture decisions/action items/open questions when present.\n\n"
            f"Title: {title}\n"
            f"Speakers: {', '.join(speakers)}\n\n"
            f"Transcript:\n{transcript_text}\n"
        )
        return prompt


TEMPLATE_MANAGER = TemplateManager()
