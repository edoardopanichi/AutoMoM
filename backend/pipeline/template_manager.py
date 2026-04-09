from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from backend.app.config import SETTINGS
from backend.app.schemas import TemplateDefinition, TemplateSection, TemplateSummary


DEFAULT_TEMPLATE_ID = "default"
DEFAULT_TEMPLATE_NAME = "Default Minutes Template"
TEMPLATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_TEMPLATE_PROMPT = (
    "You are an assistant that writes concise Minutes of Meeting in English. "
    "Use the transcript and speaker list. Return clear bullets, decisions, action items, and risks."
)
DEFAULT_TEMPLATE_SECTIONS = [
    TemplateSection(heading="### Title:", allow_prefix=True),
    TemplateSection(heading="#### Participants:"),
    TemplateSection(heading="#### Concise Overview:"),
    TemplateSection(heading="#### TODO's:"),
    TemplateSection(heading="#### CONCLUSIONS:"),
    TemplateSection(heading="#### DECISION/OPEN POINTS:"),
    TemplateSection(heading="#### RISKS:"),
]


@dataclass(frozen=True)
class FormatterPromptBundle:
    system_prompt: str
    user_prompt: str
    template: TemplateDefinition
    strict_sections: list[TemplateSection]


class TemplateManager:
    def __init__(self) -> None:
        """! @brief Initialize the TemplateManager instance.
        """
        self._ensure_default_template()

    @staticmethod
    def _validate_template_id(template_id: str) -> str:
        """! @brief Validate template id.
        @param template_id Identifier of the template.
        @return str result produced by the operation.
        """
        normalized = (template_id or "").strip()
        if not TEMPLATE_ID_PATTERN.fullmatch(normalized):
            raise ValueError("Invalid template_id. Use 1-64 chars: letters, numbers, '.', '_' or '-'")
        return normalized

    @staticmethod
    def _template_meta_path(template_id: str) -> Path:
        """! @brief Template meta path.
        @param template_id Identifier of the template.
        @return Path result produced by the operation.
        """
        safe_id = TemplateManager._validate_template_id(template_id)
        return SETTINGS.templates_dir / f"{safe_id}.json"

    def _ensure_default_template(self) -> None:
        """! @brief Ensure default template.
        """
        if self._template_meta_path(DEFAULT_TEMPLATE_ID).exists():
            return
        self.save(
            TemplateDefinition(
                template_id=DEFAULT_TEMPLATE_ID,
                name=DEFAULT_TEMPLATE_NAME,
                version="1.0.0",
                description="Default AutoMoM formatter prompt template.",
                prompt_block=DEFAULT_TEMPLATE_PROMPT,
                sections=DEFAULT_TEMPLATE_SECTIONS,
            )
        )

    def list_templates(self) -> list[TemplateSummary]:
        """! @brief List templates.
        @return List produced by the operation.
        """
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
        """! @brief Load operation.
        @param template_id Identifier of the template.
        @return Result produced by the operation.
        """
        meta_path = self._template_meta_path(template_id)
        if not meta_path.exists():
            raise FileNotFoundError(template_id)

        return TemplateDefinition(**json.loads(meta_path.read_text(encoding="utf-8")))

    def save(self, definition: TemplateDefinition) -> None:
        """! @brief Save operation.
        @param definition Value for definition.
        """
        meta_path = self._template_meta_path(definition.template_id)
        SETTINGS.templates_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(definition.model_dump(), indent=2), encoding="utf-8")

    def delete(self, template_id: str) -> None:
        """! @brief Delete operation.
        @param template_id Identifier of the template.
        """
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
        """! @brief Build formatter prompt.
        @param template_id Identifier of the template.
        @param transcript Transcript segments used by the operation.
        @param speakers Speaker names available for the meeting.
        @param title Meeting title associated with the request.
        @return str result produced by the operation.
        """
        bundle = self.build_formatter_request(template_id, transcript, speakers, title)
        return f"{bundle.system_prompt}\n\n{bundle.user_prompt}"

    def build_formatter_request(
        self,
        template_id: str,
        transcript: list[dict[str, object]],
        speakers: list[str],
        title: str,
        *,
        transcript_label: str = "Transcript",
    ) -> FormatterPromptBundle:
        """! @brief Build formatter request.
        @param template_id Identifier of the template.
        @param transcript Transcript segments used by the operation.
        @param speakers Speaker names available for the meeting.
        @param title Meeting title associated with the request.
        @param transcript_label Value for transcript label.
        @return Result produced by the operation.
        """
        template = self.load(template_id)
        transcript_text = "\n".join(f"{seg['speaker_name']}: {seg['text']}" for seg in transcript)
        sections = template.sections or []
        section_rules = ""
        if sections:
            ordered = "\n".join(
                f"{index}. {section.heading}{' <content>' if section.allow_prefix else ''}"
                for index, section in enumerate(sections, start=1)
            )
            section_rules = (
                "Required section order:\n"
                f"{ordered}\n\n"
                "Validation rules:\n"
                "- Return markdown only.\n"
                "- Keep the exact section order.\n"
                "- Do not rename headings.\n"
                "- If a required section has no content, write exactly 'None'.\n\n"
            )
        system_prompt = (
            f"{template.prompt_block}\n\n"
            f"{section_rules}"
            "Global requirements:\n"
            "- Final output must be in English.\n"
            "- Capture decisions, action items, and open questions when present.\n"
            "- Do not add commentary outside the document.\n"
        )
        user_prompt = (
            f"Title: {title}\n"
            f"Speakers: {', '.join(speakers)}\n\n"
            f"{transcript_label}:\n{transcript_text}\n"
        )
        return FormatterPromptBundle(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            template=template,
            strict_sections=sections,
        )


TEMPLATE_MANAGER = TemplateManager()
