from __future__ import annotations

import json
import mimetypes
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class RemoteWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteEmbeddingResult:
    vector: list[float]
    threshold: float
    library_version: str
    profile_model_ref: str
    embedding_model_ref: str
    engine_kind: str = "remote_pyannote"


class RemoteWorkerClient:
    def __init__(self, *, base_url: str, auth_token: str = "", timeout_s: int = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token.strip()
        self.timeout_s = max(1, int(timeout_s))

    def health(self) -> dict[str, object]:
        return self._request_json("GET", "/health")

    def diarize(
        self,
        *,
        audio_path: Path,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> dict[str, object]:
        fields: dict[str, str] = {}
        if min_speakers is not None:
            fields["min_speakers"] = str(int(min_speakers))
        if max_speakers is not None:
            fields["max_speakers"] = str(int(max_speakers))
        return self._request_json("POST", "/diarize", files={"audio_file": audio_path}, fields=fields)

    def transcribe(self, *, audio_path: Path) -> dict[str, object]:
        return self._request_json("POST", "/transcribe", files={"audio_file": audio_path})

    def embed(self, *, audio_path: Path, clip_ranges: list[tuple[float, float]]) -> RemoteEmbeddingResult:
        payload = self._request_json(
            "POST",
            "/embed",
            files={"audio_file": audio_path},
            fields={"clip_ranges": json.dumps([{"start_s": start_s, "end_s": end_s} for start_s, end_s in clip_ranges])},
        )
        return RemoteEmbeddingResult(
            vector=[float(item) for item in payload.get("vector", [])],
            threshold=float(payload.get("threshold", 0.82)),
            library_version=str(payload.get("library_version", "")),
            profile_model_ref=str(payload.get("profile_model_ref", "")),
            embedding_model_ref=str(payload.get("embedding_model_ref", "")),
            engine_kind=str(payload.get("engine_kind", "remote_pyannote")),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        files: dict[str, Path] | None = None,
        fields: dict[str, str] | None = None,
    ) -> dict[str, object]:
        headers = self._headers()
        data = None
        if files:
            data, content_type = _encode_multipart(fields or {}, files)
            headers["Content-Type"] = content_type
        elif fields:
            data = json.dumps(fields).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in {401, 403}:
                raise RemoteWorkerError(f"Remote worker auth failed: {self.base_url}") from exc
            raise RemoteWorkerError(f"Remote worker request failed ({exc.code}): {detail or self.base_url}") from exc
        except urllib.error.URLError as exc:
            raise RemoteWorkerError(f"Remote worker unreachable: {self.base_url}") from exc
        except json.JSONDecodeError as exc:
            raise RemoteWorkerError(f"Remote worker returned invalid JSON: {self.base_url}{path}") from exc

    def _headers(self) -> dict[str, str]:
        if not self.auth_token:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}


def _encode_multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"automom-{uuid.uuid4().hex}"
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
    for name, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
