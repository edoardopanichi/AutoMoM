#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import SETTINGS, ensure_directories
from backend.app.job_store import JOB_STORE
from backend.app.schemas import SpeakerMappingItem
from backend.pipeline.orchestrator import ORCHESTRATOR
from backend.pipeline.template_manager import TemplateManager


def prepare_mock_models() -> None:
    targets = [
        (
            Path(SETTINGS.diarization_model_path),
            b"version: 3.1\npipeline:\n  name: mock\n",
        ),
        (SETTINGS.models_dir / "voxtral" / "model.gguf", b"mock"),
        (SETTINGS.models_dir / "formatter" / "selected_model.txt", b"llama3.1:8b-instruct-q4_K_M"),
    ]
    for path, payload in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--template-id", default="default")
    parser.add_argument("--title", default="Long Audio Validation")
    args = parser.parse_args()

    ensure_directories()
    prepare_mock_models()
    TemplateManager()

    runtime = JOB_STORE.create_job(
        audio_path=args.audio_path,
        original_filename=args.audio_path.name,
        template_id=args.template_id,
        language_mode="auto",
        title=args.title,
    )
    job_id = runtime.state.job_id
    print(f"Job created: {job_id}")

    ORCHESTRATOR.submit(job_id)

    terminal = {"completed", "failed", "cancelled"}
    last_status = None
    while True:
        state = JOB_STORE.get_state(job_id)
        if state.status != last_status:
            print(
                f"status={state.status} stage={state.current_stage} "
                f"overall={state.overall_percent:.1f}% stage_pct={state.stage_percent:.1f}%"
            )
            last_status = state.status

        if state.status == "waiting_speaker_input" and state.speaker_info:
            mappings = [
                SpeakerMappingItem(
                    speaker_id=speaker.speaker_id,
                    name=speaker.suggested_name or speaker.speaker_id,
                    save_voice_profile=False,
                )
                for speaker in state.speaker_info.speakers
            ]
            JOB_STORE.submit_speaker_mapping(job_id, mappings)
            print(f"submitted speaker mapping for {len(mappings)} speakers")

        if state.status in terminal:
            print(f"final_status={state.status}")
            if state.error:
                print(f"error={state.error}")
            print(f"artifacts={state.artifact_paths}")
            return 0 if state.status == "completed" else 1

        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
