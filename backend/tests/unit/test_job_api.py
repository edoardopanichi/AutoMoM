from __future__ import annotations

import asyncio

from backend.app.main import create_job


class FakeUploadFile:
    filename = "meeting.wav"

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    async def close(self) -> None:
        pass


def test_api_backed_job_creation_does_not_resolve_local_models(isolated_settings, monkeypatch) -> None:
    """! @brief Test api backed job creation does not resolve local models.
    @param isolated_settings Value for isolated settings.
    @param monkeypatch Value for monkeypatch.
    """
    monkeypatch.setattr(
        "backend.app.main.LOCAL_MODEL_CATALOG.validate_selection",
        lambda selections: (True, None),
    )
    monkeypatch.setattr(
        "backend.app.main.resolve_local_diarization_model",
        lambda model_id: (_ for _ in ()).throw(AssertionError("resolved diarization")),
    )
    monkeypatch.setattr(
        "backend.app.main.LOCAL_MODEL_CATALOG.resolve_model",
        lambda stage, model_id: (_ for _ in ()).throw(AssertionError(f"resolved {stage}")),
    )
    monkeypatch.setattr("backend.app.main.TEMPLATE_MANAGER.load", lambda template_id: object())
    monkeypatch.setattr("backend.app.main.ORCHESTRATOR.submit", lambda job_id: None)

    response = asyncio.run(
        create_job(
            audio_file=FakeUploadFile(b"abc"),
            template_id="default",
            language_mode="auto",
            title="Cloud",
            diarization_execution="api",
            transcription_execution="api",
            formatter_execution="api",
            openai_api_key="sk-test",
            local_diarization_model_id="",
            local_transcription_model_id="",
            local_formatter_model_id="",
            openai_diarization_model="gpt-diarize",
            openai_transcription_model="gpt-transcribe",
            openai_formatter_model="gpt-format",
        )
    )

    assert response["job_id"]
