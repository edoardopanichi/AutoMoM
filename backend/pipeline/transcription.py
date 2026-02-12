from __future__ import annotations

import subprocess
from pathlib import Path
import shutil
from typing import Callable

from backend.pipeline.diarization import merge_transcript_segments


class VoxtralTranscriber:
    def __init__(self, binary_path: str | None, model_path: str | None) -> None:
        self.binary_path = binary_path or ""
        self.model_path = model_path or ""
        self._resolved_binary_path = self._resolve_binary_path(self.binary_path)
        self._runtime_available = bool(
            self._resolved_binary_path and self.model_path and Path(self.model_path).exists()
        )

    def available(self) -> bool:
        return self._runtime_available

    def transcribe(self, segment_path: Path) -> str:
        if not self._runtime_available or not self._resolved_binary_path:
            return self._fallback_transcription(segment_path)

        command = [self._resolved_binary_path, "-m", self.model_path, "-f", str(segment_path)]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            return self._fallback_transcription(segment_path)
        text = process.stdout.strip()
        return text or self._fallback_transcription(segment_path)

    @staticmethod
    def _resolve_binary_path(binary_path: str) -> str | None:
        if not binary_path:
            return None
        as_path = Path(binary_path)
        if as_path.exists():
            return str(as_path)
        which_path = shutil.which(binary_path)
        if which_path:
            return which_path
        return None

    @staticmethod
    def _fallback_transcription(segment_path: Path) -> str:
        name = segment_path.stem.replace("_", " ")
        return f"[Offline fallback transcript for {name}]"



def transcribe_segments(
    transcriber: VoxtralTranscriber,
    segment_jobs: list[dict[str, object]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, object]]:
    transcripts: list[dict[str, object]] = []
    total = len(segment_jobs)

    for idx, segment in enumerate(segment_jobs, start=1):
        audio_path = Path(str(segment["segment_path"]))
        text = transcriber.transcribe(audio_path)
        transcripts.append(
            {
                "speaker_id": segment["speaker_id"],
                "speaker_name": segment["speaker_name"],
                "start_s": float(segment["start_s"]),
                "end_s": float(segment["end_s"]),
                "text": text,
            }
        )
        if progress_callback:
            progress_callback(idx, total)

    ordered = sorted(transcripts, key=lambda item: (item["start_s"], item["end_s"]))
    return merge_transcript_segments(ordered)
