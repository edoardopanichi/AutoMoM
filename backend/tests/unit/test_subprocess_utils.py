from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from backend.app.job_store import JOB_STORE
from backend.pipeline.subprocess_utils import SubprocessCancelledError, run_cancellable_subprocess


def test_run_cancellable_subprocess_stops_registered_job(isolated_settings, tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    runtime = JOB_STORE.create_job(
        audio_path=audio,
        original_filename=audio.name,
        template_id="default",
        language_mode="auto",
        title="Cancel Test",
        local_diarization_model_id="pyannote-community-1",
    )

    result: dict[str, object] = {}

    def run() -> None:
        try:
            run_cancellable_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                job_id=runtime.state.job_id,
            )
        except Exception as exc:  # pragma: no branch
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()

    deadline = time.time() + 5.0
    while not JOB_STORE.get_runtime(runtime.state.job_id).active_processes and time.time() < deadline:
        time.sleep(0.05)

    JOB_STORE.cancel(runtime.state.job_id)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert isinstance(result.get("error"), SubprocessCancelledError)
    assert not JOB_STORE.get_runtime(runtime.state.job_id).active_processes
