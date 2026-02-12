from __future__ import annotations

from backend.app.schemas import TemplateDefinition
from backend.pipeline.template_manager import TemplateManager


def test_template_rendering(isolated_settings) -> None:
    manager = TemplateManager()
    manager.save(
        TemplateDefinition(
            template_id="brief",
            name="Brief",
            version="1.0.0",
            description="Test template",
            prompt_block="Write concise minutes in English.",
            content="# Summary\n{{ executive_summary }}\n",
        )
    )

    rendered = manager.render("brief", {"executive_summary": "Team aligned on scope."})
    assert "Team aligned on scope." in rendered


def test_formatter_prompt_assembly(isolated_settings) -> None:
    manager = TemplateManager()
    prompt = manager.build_formatter_prompt(
        template_id="default",
        transcript=[
            {
                "speaker_name": "Alice",
                "start_s": 0.0,
                "end_s": 1.2,
                "text": "We decided to ship on Friday.",
            }
        ],
        speakers=["Alice"],
        title="Sprint Review",
    )

    assert "Final output must be in English" in prompt
    assert "Sprint Review" in prompt
    assert "Alice" in prompt
