from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.pipeline.diarization import _diarize_with_pyannote_impl
from backend.pipeline.vad import SpeechRegion


def _emit(message: dict[str, object]) -> None:
    print(json.dumps(message), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        _emit({"type": "error", "error": "pyannote_worker_invalid_args"})
        return 2

    request_path = Path(args[0])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _emit({"type": "error", "error": f"pyannote_worker_request_error:{exc.__class__.__name__}"})
        return 2

    try:
        speech_regions = [
            SpeechRegion(start_s=float(item.get("start_s", 0.0)), end_s=float(item.get("end_s", 0.0)))
            for item in request.get("speech_regions", [])
            if isinstance(item, dict)
        ]
        result, error = _diarize_with_pyannote_impl(
            audio_path=Path(str(request["audio_path"])),
            speech_regions=speech_regions,
            min_speakers=request.get("min_speakers"),
            max_speakers=request.get("max_speakers"),
            model_path=Path(str(request["model_path"])) if request.get("model_path") else None,
            compute_device=str(request.get("compute_device") or "auto"),
            cuda_device_id=int(request.get("cuda_device_id") or 0),
            pipeline_path=str(request.get("pipeline_path") or "") or None,
            embedding_model=str(request.get("embedding_model") or "") or None,
            progress_callback=lambda event: _emit({"type": "progress", "event": event}),
        )
    except Exception as exc:  # pragma: no cover
        _emit({"type": "error", "error": f"pyannote_worker_runtime_error:{exc.__class__.__name__}"})
        return 1

    if error is not None:
        _emit({"type": "error", "error": error})
        return 1

    assert result is not None
    _emit(
        {
            "type": "result",
            "result": {
                "segments": result.to_json(),
                "speaker_count": result.speaker_count,
                "mode": result.mode,
                "details": result.details,
                "chunk_plan": result.chunk_plan,
                "stitching_debug": result.stitching_debug,
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
