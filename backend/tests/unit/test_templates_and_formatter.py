from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from backend.app.config import SETTINGS
from backend.pipeline.formatter import Formatter, _extract_model_text, _strip_runtime_logs
from backend.pipeline.template_manager import TemplateManager


def test_formatter_uses_pty_fallback_when_shell_wrapper_capture_is_empty(monkeypatch, tmp_path: Path) -> None:
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

    def fake_pty(command, *, prompt, timeout_s):
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                "Minutes of Meeting\n"
                "Agenda item 6.3 removed.\n"
                "warning: no usable GPU found, --gpu-layers option will be ignored\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)
    monkeypatch.setattr("backend.pipeline.formatter._run_with_pty_capture", fake_pty)

    formatter = Formatter(
        command_template='bash -lc "cat >/tmp/prompt; llama-cli -m \\"{model}\\" -n 128 -f /tmp/prompt"',
        model_path=str(model_path),
    )
    result = formatter.run_model("hello")

    assert result == {"_raw_model_text": "Minutes of Meeting\nAgenda item 6.3 removed.\n"}
    assert formatter.last_mode == "model_plaintext"
    assert formatter.last_raw_output == "Minutes of Meeting\nAgenda item 6.3 removed.\n"


def test_formatter_shell_wrapper_uses_single_pty_invocation(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")

    counters = {"pty": 0, "run": 0}

    def fake_run(*args, **kwargs):
        counters["run"] += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_pty(command, *, prompt, timeout_s):
        counters["pty"] += 1
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="## Minutes\n- Item\n## Decisions\n- Keep\n",
            stderr="",
        )

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)
    monkeypatch.setattr("backend.pipeline.formatter._run_with_pty_capture", fake_pty)

    formatter = Formatter(
        command_template='bash -lc "cat >/tmp/prompt; llama-cli -m \\"{model}\\" -n 128 -f /tmp/prompt"',
        model_path=str(model_path),
    )
    rendered, _structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "We aligned."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert rendered.startswith("## Minutes")
    assert counters["pty"] == 1
    assert counters["run"] == 0


def test_formatter_uses_pty_fallback_when_shell_wrapper_stdout_is_only_warnings(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "warning: no usable GPU found, --gpu-layers option will be ignored\n"
                "llama_memory_breakdown_print: | memory breakdown [MiB] |\n"
            ),
            stderr="",
        )

    def fake_pty(command, *, prompt, timeout_s):
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                "Minutes of Meeting\n"
                "Agenda item 6.3 removed.\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)
    monkeypatch.setattr("backend.pipeline.formatter._run_with_pty_capture", fake_pty)

    formatter = Formatter(
        command_template='bash -lc "cat >/tmp/prompt; llama-cli -m \\"{model}\\" -n 128 -f /tmp/prompt"',
        model_path=str(model_path),
    )
    result = formatter.run_model("hello")

    assert result == {"_raw_model_text": "Minutes of Meeting\nAgenda item 6.3 removed.\n"}
    assert formatter.last_mode == "model_plaintext"
    assert formatter.last_raw_output == "Minutes of Meeting\nAgenda item 6.3 removed.\n"


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
    assert markdown == ""


def test_formatter_llama_command_is_forced_non_interactive(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_pty(command, *, prompt, timeout_s):
        captured["command"] = command
        captured["timeout"] = timeout_s
        return subprocess.CompletedProcess(args=command, returncode=0, stdout='{"decisions":["ok"]}', stderr="")

    monkeypatch.setattr("backend.pipeline.formatter._run_with_pty_capture", fake_pty)

    formatter = Formatter(
        command_template='bash -lc "cat >/tmp/prompt; llama-cli -m \\"{model}\\" -n 128 -f /tmp/prompt"',
        model_path=str(model_path),
    )

    result = formatter.run_model("hello")

    assert result == {"decisions": ["ok"]}
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
        return SimpleNamespace(returncode=0, stdout='{"decisions":["ok"]}', stderr="")

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
        return SimpleNamespace(returncode=0, stdout='{"decisions":["ok"]}', stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path), gpu_layers=99)
    result = formatter.run_model("hello")

    assert result == {"decisions": ["ok"]}
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
        "## Agenda\n"
        "- Budget review.\n"
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
    assert "decisions" in structured


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
    assert "decisions" in structured


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

    assert rendered == "Agenda:\n- Budget review\nDecisions:\n- Approve hiring plan\n"


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


def test_formatter_treats_prefixed_stderr_markdown_as_valid_output(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    stderr_output = (
        "main: ## Minutes of Meeting\n"
        "main: ## Agenda\n"
        "main: - Remove item 6.3\n"
        "main: ## Decisions\n"
        "main: - Postpone development agreement\n"
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

    assert formatter.last_mode == "model_plaintext"
    assert rendered == stderr_output
    assert formatter.last_raw_output == stderr_output
    assert formatter.last_stdout == ""
    assert formatter.last_stderr == stderr_output


def test_formatter_treats_prefixed_stderr_plaintext_as_valid_output(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    stderr_output = (
        "main: Minutes of Meeting\n"
        "main: Agenda item 6.3 removed.\n"
        "main: Decision: postpone development agreement.\n"
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
    assert formatter.last_mode == "model_plaintext"
    assert formatter.last_raw_output == stderr_output


def test_strip_runtime_logs_removes_main_runtime_lines_but_keeps_model_content() -> None:
    stderr_output = (
        "main: build = 8006\n"
        "main: available commands:\n"
        "main: /exit or Ctrl+C\n"
        "## Minutes of Meeting\n"
        "- Agenda item 6.3 removed.\n"
    )

    cleaned = _strip_runtime_logs(stderr_output)

    assert "build = 8006" not in cleaned
    assert "available commands" not in cleaned
    assert "/exit or Ctrl+C" not in cleaned
    assert "Minutes of Meeting" in cleaned
    assert "Agenda item 6.3 removed." in cleaned


def test_extract_model_text_from_llama_cli_stdout_keeps_only_generated_text() -> None:
    stdout = (
        "Loading model... \n\n"
        "available commands:\n"
        "  /exit or Ctrl+C\n\n"
        "> Write a markdown summary.\n\n"
        "## Agenda\n"
        "- Item 1\n\n"
        "## Decisions\n"
        "- Decision 1\n\n"
        "[ Prompt: 60.0 t/s | Generation: 10.0 t/s ]\n\n"
        "Exiting...\n"
    )

    extracted = _extract_model_text(stdout, "")

    assert "Loading model" not in extracted
    assert "available commands" not in extracted
    assert "[ Prompt:" not in extracted
    assert "Exiting..." not in extracted
    assert "## Agenda" in extracted
    assert "## Decisions" in extracted


def test_extract_model_text_removes_prompt_echo_when_prompt_provided() -> None:
    prompt = "Title: Sync\nSpeakers: Alice\nTranscript:\nAlice: Hello"
    stdout = (
        "Loading model...\n"
        "> " + prompt + "\n\n"
        "|/-\\\n"
        "## Agenda\n- Budget review\n\n## Decisions\n- Approve rollout\n\n"
        "[ Prompt: 60.0 t/s | Generation: 10.0 t/s ]\n"
        "Exiting...\n"
    )

    extracted = _extract_model_text(stdout, "", prompt=prompt)

    assert prompt not in extracted
    assert "Loading model" not in extracted
    assert "## Agenda" in extracted
    assert "## Decisions" in extracted
    assert "[ Prompt:" not in extracted
    assert "Exiting..." not in extracted


def test_extract_model_text_handles_truncated_prompt_echo() -> None:
    prompt = "Title: Sync\nSpeakers: Alice\nTranscript:\nAlice: " + ("hello " * 100)
    stdout = (
        "Loading model...\n"
        "available commands:\n"
        "> " + prompt[:80] + " ... (truncated)\n\n"
        "|/-\\\n"
        "## Agenda\n- Budget review\n\n## Decisions\n- Approve rollout\n\n"
        "[ Prompt: 60.0 t/s | Generation: 10.0 t/s ]\n"
        "Exiting...\n"
    )

    extracted = _extract_model_text(stdout, "", prompt=prompt)

    assert "Loading model" not in extracted
    assert "available commands" not in extracted
    assert "(truncated)" not in extracted
    assert "## Agenda" in extracted
    assert "## Decisions" in extracted


def test_extract_model_text_preserves_markdown_tables() -> None:
    stdout = (
        "\n> Write markdown minutes.\n\n"
        "| Agenda | Decisions |\n"
        "| --- | --- |\n"
        "| Remove item 6.3 | Postpone agreement |\n"
        "[ Prompt: 60.0 t/s | Generation: 10.0 t/s ]\n"
        "Exiting...\n"
    )

    extracted = _extract_model_text(stdout, "")

    assert "| Agenda | Decisions |" in extracted
    assert "| --- | --- |" in extracted
    assert "| Remove item 6.3 | Postpone agreement |" in extracted


def test_formatter_preserves_output_even_when_process_returncode_nonzero(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    stdout_output = "# Minutes\n## Decisions\n- Keep shipping\n"

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(returncode=1, stdout=stdout_output, stderr="error: partial failure")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    rendered, _structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "Keep shipping."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert rendered == stdout_output
    assert formatter.last_mode == "model_markdown"
    assert formatter.last_raw_output == stdout_output


def test_formatter_fallback_markdown_contains_no_executive_summary() -> None:
    formatter = Formatter(command_template="", model_path="")
    markdown, _structured, _ = formatter.build_structured_summary(
        transcript=[{"speaker_name": "Alice", "text": "We aligned on scope."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
    )

    assert markdown == ""


def test_formatter_writes_model_output_directly_to_mom_file(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")
    mom_path = tmp_path / "mom.md"
    model_output = "# Minutes\n## Decisions\n- Keep shipping\n"

    def fake_run(command, input, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout=model_output, stderr="")

    monkeypatch.setattr("backend.pipeline.formatter.subprocess.run", fake_run)

    formatter = Formatter(command_template="llama-cli -m {model}", model_path=str(model_path))
    structured, _prompt = formatter.write_model_output_to_mom(
        transcript=[{"speaker_name": "Alice", "text": "Keep shipping."}],
        speakers=["Alice"],
        title="Sync",
        template_id="default",
        output_path=mom_path,
    )

    assert mom_path.read_text(encoding="utf-8") == model_output
    assert formatter.last_mode == "model_markdown"
    assert "decisions" in structured
