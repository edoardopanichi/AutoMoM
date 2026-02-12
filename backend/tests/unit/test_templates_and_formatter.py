from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from backend.app.config import SETTINGS
from backend.app.schemas import TemplateDefinition
from backend.pipeline.formatter import Formatter
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


def test_formatter_llama_command_is_forced_non_interactive(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(command, input, capture_output, text, timeout):
        captured["command"] = command
        captured["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout='{"executive_summary":"ok"}', stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(
        command_template='bash -lc "cat >/tmp/prompt; llama-cli -m \\"{model}\\" -n 128 -f /tmp/prompt"',
        model_path=str(model_path),
    )

    result = formatter.run_model("hello")

    assert result == {"executive_summary": "ok"}
    assert formatter.last_mode == "model_json"
    assert captured["timeout"] == SETTINGS.formatter_timeout_s
    script = str(captured["command"][2])  # type: ignore[index]
    assert "--single-turn" in script


def test_formatter_timeout_falls_back_to_heuristic(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="llama-cli", timeout=1.0)

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    result = formatter.run_model("hello")

    assert result is None
    assert formatter.last_mode == "heuristic_command_timeout"
