from __future__ import annotations

import json
import urllib.error
from pathlib import Path

from backend.app.config import SETTINGS
from backend.pipeline.formatter import Formatter, _extract_model_text, _strip_runtime_logs, validate_markdown_output
from backend.pipeline.template_manager import TemplateManager


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        """! @brief Initialize the _FakeHTTPResponse instance.
        @param payload Payload consumed or produced by the operation.
        """
        self._payload = payload

    def read(self) -> bytes:
        """! @brief Read operation.
        @return Result produced by the operation.
        """
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        """! @brief Enter operation.
        @return Result produced by the operation.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """! @brief Exit operation.
        @param exc_type Value for exc type.
        @param exc Value for exc.
        @param tb Value for tb.
        @return Result produced by the operation.
        """
        return False


def test_formatter_prompt_assembly(isolated_settings) -> None:
    """! @brief Test formatter prompt assembly.
    @param isolated_settings Value for isolated settings.
    """
    manager = TemplateManager()
    bundle = manager.build_formatter_request(
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

    assert "Final output must be in English" in bundle.system_prompt
    assert "Required section order" in bundle.system_prompt
    assert "Sprint Review" in bundle.user_prompt
    assert "Alice" in bundle.user_prompt


def test_formatter_heuristic_transcript_markdown_has_no_timestamps() -> None:
    """! @brief Test formatter heuristic transcript markdown has no timestamps.
    """
    formatter = Formatter(command_template="", model_path="")
    structured = formatter._heuristic_structuring(
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
    )

    transcript_md = str(structured["transcript_markdown"])
    assert "- **Alice**: We aligned on scope." in transcript_md
    assert "- **Bob**: I will share the plan." in transcript_md


def test_formatter_uses_ollama_response(monkeypatch) -> None:
    """! @brief Test formatter uses ollama response.
    @param monkeypatch Value for monkeypatch.
    """
    def fake_urlopen(request, timeout=0):
        """! @brief Fake urlopen.
        @param request Request payload for the operation.
        @param timeout Optional timeout in seconds.
        @return Result produced by the operation.
        """
        assert request.full_url.endswith("/api/generate")
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["system"]
        return _FakeHTTPResponse({"response": "## Minutes\n- Item\n## Decisions\n- Keep\n"})

    monkeypatch.setattr("backend.pipeline.formatter.urllib.request.urlopen", fake_urlopen)
    formatter = Formatter(ollama_host="http://127.0.0.1:11434", ollama_model="llama3.1:8b")
    result = formatter.run_model("hello")

    assert result == {"_raw_markdown_text": "## Minutes\n- Item\n## Decisions\n- Keep\n"}
    assert formatter.last_mode == "model_markdown"


def test_formatter_handles_ollama_unavailable(monkeypatch) -> None:
    """! @brief Test formatter handles ollama unavailable.
    @param monkeypatch Value for monkeypatch.
    """
    def fake_urlopen(request, timeout=0):
        """! @brief Fake urlopen.
        @param request Request payload for the operation.
        @param timeout Optional timeout in seconds.
        @return Result produced by the operation.
        """
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("backend.pipeline.formatter.urllib.request.urlopen", fake_urlopen)
    formatter = Formatter(ollama_host="http://127.0.0.1:11434", ollama_model="llama3.1:8b")
    result = formatter.run_model("hello")

    assert result is None
    assert formatter.last_mode == "heuristic_ollama_unavailable"


def test_formatter_handles_ollama_timeout(monkeypatch) -> None:
    """! @brief Test formatter handles ollama timeout.
    @param monkeypatch Value for monkeypatch.
    """
    def fake_urlopen(request, timeout=0):
        """! @brief Fake urlopen.
        @param request Request payload for the operation.
        @param timeout Optional timeout in seconds.
        @return Result produced by the operation.
        """
        raise TimeoutError("timed out")

    monkeypatch.setattr("backend.pipeline.formatter.urllib.request.urlopen", fake_urlopen)
    formatter = Formatter(ollama_host="http://127.0.0.1:11434", ollama_model="llama3.1:8b")
    result = formatter.run_model("hello")

    assert result is None
    assert formatter.last_mode == "heuristic_ollama_timeout"


def test_formatter_uses_openai_response(monkeypatch) -> None:
    """! @brief Test formatter uses openai response.
    @param monkeypatch Value for monkeypatch.
    """
    monkeypatch.setattr(
        "backend.pipeline.formatter.OpenAIClient.generate_text",
        lambda self, prompt, model, timeout_s, instructions="": "## Minutes\n- Strong summary\n## Decisions\n- Ship it\n",
    )
    formatter = Formatter(openai_api_key="sk-test", openai_model="gpt-5-mini")

    result = formatter.run_model("hello")

    assert result == {"_raw_markdown_text": "## Minutes\n- Strong summary\n## Decisions\n- Ship it\n"}
    assert formatter.last_mode == "model_markdown"


def test_formatter_legacy_command_still_supported(tmp_path: Path) -> None:
    """! @brief Test formatter legacy command still supported.
    @param tmp_path Value for tmp path.
    """
    model_path = tmp_path / "formatter.gguf"
    model_path.write_text("model", encoding="utf-8")

    script = "import sys; sys.stdin.read(); sys.stdout.write('## Minutes\\n- Item\\n## Decisions\\n- Keep\\n')"
    formatter = Formatter(
        command_template=f"python -c \"{script}\"",
        model_path=str(model_path),
        timeout_s=SETTINGS.formatter_timeout_s,
    )
    result = formatter.run_model("hello")

    assert result == {"_raw_markdown_text": "## Minutes\n- Item\n## Decisions\n- Keep\n"}
    assert formatter.last_mode == "model_markdown"


def test_strip_runtime_logs_basic() -> None:
    """! @brief Test strip runtime logs basic.
    """
    text = "time=1.23s\n## Minutes\n- Keep\n"
    assert _strip_runtime_logs(text) == "## Minutes\n- Keep\n"


def test_extract_model_text_prefers_stdout() -> None:
    """! @brief Test extract model text prefers stdout.
    """
    stdout = "## Minutes\n- Keep\n"
    stderr = "error\n"
    assert _extract_model_text(stdout, stderr).startswith("## Minutes")


def test_validate_markdown_output_requires_template_headings() -> None:
    """! @brief Test validate markdown output reqUIres template headings.
    """
    manager = TemplateManager()
    sections = manager.load("default").sections

    result = validate_markdown_output(
        "### Title: Minutes of Meeting - Test\n#### Participants:\nNone\n#### RISKS:\nNone\n",
        sections,
    )

    assert not result["valid"]
    assert any("Missing required heading '#### Concise Overview:'" in item for item in result["errors"])
