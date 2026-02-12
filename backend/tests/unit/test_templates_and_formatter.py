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
    assert "[0.00-1.20]" not in prompt


def test_formatter_heuristic_transcript_markdown_has_no_timestamps() -> None:
    formatter = Formatter(command_template="", model_path="")
    markdown, structured, _ = formatter.build_structured_summary(
        transcript=[
            {
                "speaker_name": "Alice",
                "start_s": 0.0,
                "end_s": 1.2,
                "text": "We aligned on scope.",
            },
            {
                "speaker_name": "Bob",
                "start_s": 1.3,
                "end_s": 2.1,
                "text": "I will share the plan.",
            },
        ],
        speakers=["Alice", "Bob"],
        title="Sync",
        template_id="default",
    )

    transcript_md = str(structured["transcript_markdown"])
    assert "- **Alice**: We aligned on scope." in transcript_md
    assert "- **Bob**: I will share the plan." in transcript_md
    assert "[0.00-1.20]" not in transcript_md
    assert "## Transcript (optional)" in markdown


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


def test_formatter_injects_gpu_layers_when_gpu_available(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    monkeypatch.setattr("backend.pipeline.formatter.should_enable_native_gpu", lambda *_: True)

    captured: dict[str, object] = {}

    def fake_run(command, input, capture_output, text, timeout):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout='{"executive_summary":"ok"}', stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path), gpu_layers=99)
    formatter.run_model("hello")

    command = captured["command"]  # type: ignore[index]
    assert "--single-turn" in command
    assert "-ngl" in command


def test_formatter_retries_without_gpu_on_rc0_stderr_error(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    monkeypatch.setattr("backend.pipeline.formatter.should_enable_native_gpu", lambda *_: True)

    calls: list[list[str]] = []

    def fake_run(command, input, capture_output, text, timeout):
        calls.append(command)
        if "-ngl" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="error: unknown argument: -ngl")
        return SimpleNamespace(returncode=0, stdout='{"executive_summary":"ok"}', stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path), gpu_layers=99)
    result = formatter.run_model("hello")

    assert result == {"executive_summary": "ok"}
    assert any("-ngl" in call for call in calls)


def test_formatter_ignores_runtime_warnings_when_output_is_empty(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr=(
                "warning: no usable GPU found, --gpu-layers option will be ignored\n"
                "llama_memory_breakdown_print: | memory breakdown [MiB] |\n"
            ),
        )

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    result = formatter.run_model("hello")

    assert result is None
    assert formatter.last_mode == "heuristic_empty_output"


def test_formatter_preserves_model_markdown_output(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    markdown = (
        "# Meeting Info\n"
        "## Executive Summary\n"
        "- Team aligned.\n"
        "## Decisions\n"
        "- Ship on Friday.\n"
    )

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout=markdown,
            stderr="warning: no usable GPU found, --gpu-layers option will be ignored\n",
        )

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    rendered, structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "We decided to ship on Friday."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert formatter.last_mode == "model_markdown"
    assert rendered == markdown
    assert "executive_summary" in structured


def test_formatter_preserves_model_plaintext_output(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    plaintext = (
        "Meeting Info:\n"
        "Agenda:\n"
        "- Budget review\n"
        "Decisions:\n"
        "- Approve hiring plan\n"
    )

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout=plaintext, stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    rendered, structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "We approve the hiring plan."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert formatter.last_mode == "model_plaintext"
    assert rendered == plaintext
    assert "executive_summary" in structured


def test_formatter_preserves_raw_stdout_even_with_runtime_prefix(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    raw_output = "main: processing\rAgenda:\r- Budget review\rDecisions:\r- Approve hiring plan\r"

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout=raw_output, stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    rendered, _structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "We approve the hiring plan."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert rendered == raw_output


def test_formatter_passthrough_even_for_json_output(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    raw_json = '{"title":"Team Sync","agenda":["Budget review"],"decisions":["Approve hiring plan"]}'

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout=raw_json, stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    rendered, _structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "We approve the hiring plan."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert rendered == raw_json


def test_formatter_uses_markdown_from_stderr_when_stdout_empty(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    stderr_output = (
        "# Minutes of Meeting\n"
        "## Agenda\n"
        "- Remove item 6.3\n"
        "## Decisions\n"
        "- Postpone development agreement\n"
    )

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout="", stderr=stderr_output)

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    rendered, _structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "We postpone item 6.3."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert rendered == stderr_output
