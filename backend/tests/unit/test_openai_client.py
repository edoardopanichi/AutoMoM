from __future__ import annotations

from backend.pipeline.openai_client import (
    _build_multipart_body,
    _extract_response_output_text,
)


def test_extract_response_output_text_reads_top_level_field() -> None:
    """! @brief Test extract response output text reads top level field.
    """
    assert _extract_response_output_text({"output_text": "Hello"}) == "Hello"


def test_extract_response_output_text_reads_output_blocks() -> None:
    """! @brief Test extract response output text reads output blocks.
    """
    payload = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "Line one"},
                    {"type": "output_text", "text": "Line two"},
                ]
            }
        ]
    }

    assert _extract_response_output_text(payload) == "Line one\nLine two"


def test_build_multipart_body_contains_fields_and_file(tmp_path) -> None:
    """! @brief Test bUIld multipart body contains fields and file.
    @param tmp_path Value for tmp path.
    """
    file_path = tmp_path / "clip.wav"
    file_path.write_bytes(b"abc123")

    body = _build_multipart_body(
        boundary="test-boundary",
        fields={"model": "gpt-4o-transcribe", "response_format": "json"},
        file_field="file",
        file_path=file_path,
        file_bytes=file_path.read_bytes(),
    )
    decoded = body.decode("utf-8", errors="replace")

    assert 'name="model"' in decoded
    assert "gpt-4o-transcribe" in decoded
    assert 'name="file"; filename="clip.wav"' in decoded
    assert decoded.endswith("--test-boundary--\r\n")
