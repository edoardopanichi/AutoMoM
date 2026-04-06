from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

from backend.app.job_store import JOB_STORE


class SubprocessCancelledError(RuntimeError):
    pass


def run_cancellable_subprocess(
    command: list[str],
    *,
    job_id: str | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
    capture_output: bool = True,
    text: bool = True,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    if job_id and JOB_STORE.is_cancelled(job_id):
        raise SubprocessCancelledError("Job cancelled")

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        start_new_session=True,
        **kwargs,
    )
    if job_id:
        JOB_STORE.register_process(job_id, process)

    communicate_input = input_text
    start = time.monotonic()
    try:
        while True:
            if job_id and JOB_STORE.is_cancelled(job_id):
                terminate_process_tree(process)
                raise SubprocessCancelledError("Job cancelled")

            remaining_timeout: float | None = None
            if timeout is not None:
                elapsed = time.monotonic() - start
                remaining_timeout = max(0.0, timeout - elapsed)
                if remaining_timeout == 0.0:
                    terminate_process_tree(process)
                    raise subprocess.TimeoutExpired(command, timeout)

            try:
                stdout, stderr = process.communicate(
                    input=communicate_input,
                    timeout=min(0.2, remaining_timeout) if remaining_timeout is not None else 0.2,
                )
                if job_id and JOB_STORE.is_cancelled(job_id):
                    raise SubprocessCancelledError("Job cancelled")
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                communicate_input = None
                continue
    finally:
        if job_id:
            JOB_STORE.unregister_process(job_id, process)


def terminate_process_tree(process: subprocess.Popen[str], *, grace_timeout_s: float = 1.5) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except Exception:
        return

    try:
        process.wait(timeout=grace_timeout_s)
        return
    except Exception:
        pass

    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=1.0)
    except Exception:
        pass
