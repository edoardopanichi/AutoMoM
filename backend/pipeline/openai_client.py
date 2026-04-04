from __future__ import annotations

import json
import mimetypes
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


OPENAI_API_BASE_URL = "https://api.openai.com/v1"
OPENAI_MAX_FILE_BYTES = 25 * 1024 * 1024


class OpenAIAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIDiarizedSegment:
    speaker_id: str
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class OpenAIDiarizationResult:
    text: str
    segments: list[OpenAIDiarizedSegment]


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str = OPENAI_API_BASE_URL) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("OpenAI API key cannot be empty")
        self.api_key = normalized_key
        self.base_url = base_url.rstrip("/")

    def diarize_audio(self, audio_path: Path, model: str) -> OpenAIDiarizationResult:
        payload = self._post_multipart(
            "/audio/transcriptions",
            fields={
                "model": model,
                "response_format": "diarized_json",
                "chunking_strategy": "auto",
            },
            file_field="file",
            file_path=audio_path,
        )
        segments_payload = payload.get("segments")
        if not isinstance(segments_payload, list) or not segments_payload:
            raise OpenAIAPIError("OpenAI diarization returned no speaker segments.")
        speaker_map: dict[str, str] = {}
        normalized_segments: list[OpenAIDiarizedSegment] = []
        for item in segments_payload:
            if not isinstance(item, dict):
                continue
            raw_label = str(item.get("speaker") or "").strip() or "speaker"
            speaker_id = speaker_map.setdefault(raw_label, f"SPEAKER_{len(speaker_map)}")
            normalized_segments.append(
                OpenAIDiarizedSegment(
                    speaker_id=speaker_id,
                    start_s=float(item.get("start") or 0.0),
                    end_s=float(item.get("end") or 0.0),
                    text=str(item.get("text") or "").strip(),
                )
            )
        if not normalized_segments:
            raise OpenAIAPIError("OpenAI diarization returned no usable segments.")
        return OpenAIDiarizationResult(
            text=str(payload.get("text") or "").strip(),
            segments=normalized_segments,
        )

    def transcribe_audio(self, audio_path: Path, model: str) -> str:
        payload = self._post_multipart(
            "/audio/transcriptions",
            fields={
                "model": model,
                "response_format": "json",
            },
            file_field="file",
            file_path=audio_path,
        )
        text = str(payload.get("text") or "").strip()
        if not text:
            raise OpenAIAPIError("OpenAI transcription returned empty text.")
        return text

    def generate_text(self, prompt: str, model: str, timeout_s: int) -> str:
        payload = self._post_json(
            "/responses",
            body={
                "model": model,
                "instructions": (
                    "Write concise, professional markdown minutes of meeting in English. "
                    "Return only the final markdown document."
                ),
                "input": prompt,
            },
            timeout_s=timeout_s,
        )
        text = _extract_response_output_text(payload)
        if not text:
            raise OpenAIAPIError("OpenAI formatter response was empty.")
        return text

    def _post_json(self, path: str, body: dict[str, object], timeout_s: int) -> dict[str, object]:
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self._read_json_response(request, timeout_s=timeout_s)

    def _post_multipart(
        self,
        path: str,
        *,
        fields: dict[str, str],
        file_field: str,
        file_path: Path,
        timeout_s: int = 300,
    ) -> dict[str, object]:
        file_bytes = file_path.read_bytes()
        if len(file_bytes) > OPENAI_MAX_FILE_BYTES:
            raise OpenAIAPIError(
                f"Audio file '{file_path.name}' exceeds the 25 MB OpenAI upload limit."
            )

        boundary = f"automom-{uuid.uuid4().hex}"
        body = _build_multipart_body(
            boundary=boundary,
            fields=fields,
            file_field=file_field,
            file_path=file_path,
            file_bytes=file_bytes,
        )
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        return self._read_json_response(request, timeout_s=timeout_s)

    def _read_json_response(self, request: urllib.request.Request, timeout_s: int) -> dict[str, object]:
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                raw_error = exc.read().decode("utf-8", errors="replace")
            except Exception:
                raw_error = ""
            if raw_error:
                try:
                    payload = json.loads(raw_error)
                    error_payload = payload.get("error")
                    if isinstance(error_payload, dict):
                        detail = str(error_payload.get("message") or "").strip()
                    if not detail:
                        detail = str(payload.get("message") or raw_error).strip()
                except json.JSONDecodeError:
                    detail = raw_error.strip()
            raise OpenAIAPIError(detail or f"OpenAI request failed with HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            raise OpenAIAPIError(f"OpenAI request failed: {exc.reason}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenAIAPIError("OpenAI returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise OpenAIAPIError("OpenAI returned an unexpected response payload.")
        return payload


def _build_multipart_body(
    *,
    boundary: str,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    file_bytes: bytes,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks)


def _extract_response_output_text(payload: dict[str, object]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "output_text":
                continue
            text = str(block.get("text") or "").strip()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts).strip()
