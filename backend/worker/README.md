# AutoMoM Remote Workers

This folder contains small HTTP workers that let the main AutoMoM app run heavy model stages on another machine over the LAN.

They are examples and launch helpers around the same FastAPI app:

```bash
uvicorn backend.worker.main:app
```

The selected worker behavior is controlled by environment variables.

## What They Are For

- `run_diarization_service.sh` starts a worker that exposes pyannote diarization and profile embedding endpoints.
- `run_transcription_service.sh` starts a worker that exposes a whisper.cpp transcription endpoint.
- There is no formatter worker script here. Remote formatter execution uses an Ollama HTTP server directly, for example `http://gpu-box:11434`.

Both scripts run `backend.worker.main:app` because there is one worker server implementation. The script only changes which stage is enabled and which model paths/names are advertised.

## Endpoints

- `GET /health` reports enabled stages, model names, hostname, and whether auth is required.
- `POST /diarize` runs diarization. Enabled only when `AUTOMOM_REMOTE_WORKER_ENABLED_STAGES` contains `diarization`.
- `POST /embed` computes voice-profile embeddings. Enabled only with `diarization`.
- `POST /transcribe` runs whisper.cpp transcription. Enabled only when `AUTOMOM_REMOTE_WORKER_ENABLED_STAGES` contains `transcription`.

## Variables You Usually Adjust

For both workers:

- `AUTOMOM_REMOTE_WORKER_HOST`: bind address. Use `0.0.0.0` for a real LAN worker, `127.0.0.1` for same-laptop testing.
- `AUTOMOM_REMOTE_WORKER_PORT`: HTTP port. Defaults are `8010` for diarization and `8011` for transcription.
- `AUTOMOM_REMOTE_AUTH_TOKEN`: optional bearer token. If set, the main app must register the same token in the remote model config.
- `AUTOMOM_COMPUTE_DEVICE`: inherited from the normal app config. Use `auto`, `cpu`, or `cuda`.
- `AUTOMOM_CUDA_DEVICE_ID`: GPU index when CUDA is used.
- `AUTOMOM_FFMPEG_BIN`: ffmpeg executable path if it is not just `ffmpeg`.

For diarization:

- `AUTOMOM_REMOTE_DIARIZATION_PIPELINE`: path to the pyannote pipeline `config.yaml` on the worker machine.
- `AUTOMOM_REMOTE_DIARIZATION_MODEL_NAME`: display/compatibility name reported by `/health`.
- `AUTOMOM_REMOTE_PROFILE_MODEL_REF`: profile compatibility id. Keep this stable across machines if you want saved voice profiles to match.
- `AUTOMOM_REMOTE_DIARIZATION_EMBEDDING_MODEL`: embedding model reference used for speaker/profile matching.

Important: set `AUTOMOM_REMOTE_DIARIZATION_PIPELINE` as a single-line value. Do not split the path across lines. A wrapped value can inject a newline and lead to pyannote/Hugging Face validation failures (for example `HFValidationError`).

For transcription:

- `AUTOMOM_REMOTE_TRANSCRIPTION_BIN`: path to `whisper-cli` on the worker machine.
- `AUTOMOM_REMOTE_TRANSCRIPTION_MODEL`: path to the `.gguf` transcription model on the worker machine.
- `AUTOMOM_REMOTE_TRANSCRIPTION_MODEL_NAME`: display/compatibility name reported by `/health`.
- `AUTOMOM_TRANSCRIPTION_THREADS`, `AUTOMOM_TRANSCRIPTION_PROCESSORS`, and `AUTOMOM_TRANSCRIPTION_GPU_LAYERS`: normal whisper.cpp tuning values, read by the worker through shared app settings.

## Same-Laptop Test

Use this when you only have one laptop and want to test the LAN code path through `127.0.0.1`.

Terminal 1:

```bash
./scripts/run_automom.sh
```

Terminal 2:

```bash
source .venv/bin/activate
AUTOMOM_REMOTE_WORKER_HOST=127.0.0.1 \
AUTOMOM_REMOTE_DIARIZATION_PIPELINE="data/models/diarization/pyannote-speaker-diarization-community-1/config.yaml" \
backend/worker/run_diarization_service.sh
```

Keep the `AUTOMOM_REMOTE_DIARIZATION_PIPELINE="..."` assignment on one line exactly as above.

Terminal 3:

```bash
source .venv/bin/activate
AUTOMOM_REMOTE_WORKER_HOST=127.0.0.1 \
AUTOMOM_REMOTE_TRANSCRIPTION_BIN="<path-to-whisper-cli>" \
AUTOMOM_REMOTE_TRANSCRIPTION_MODEL="data/models/transcription/model.gguf" \
AUTOMOM_REMOTE_TRANSCRIPTION_MODEL_NAME="large-v3" \
backend/worker/run_transcription_service.sh
```

Check the workers:

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8011/health
```

Then register remote models in the main AutoMoM UI:

- Diarization remote base URL: `http://127.0.0.1:8010`
- Transcription remote base URL: `http://127.0.0.1:8011`
- Formatter remote base URL, if testing remote Ollama: `http://127.0.0.1:11434`

## Real LAN Worker

On the worker machine, bind to all interfaces:

```bash
AUTOMOM_REMOTE_WORKER_HOST=0.0.0.0 backend/worker/run_diarization_service.sh
```

In the main AutoMoM UI, register the model with the worker machine address, for example:

```text
http://office-gpu:8010
```

The model paths in the worker variables are paths on the worker machine, not paths on the laptop running the main AutoMoM app.

## Notes

- The scripts provide practical defaults for container/server deployments, such as `/models/...`; override those paths for laptop testing.
- The values registered in the main app must match what `/health` reports. For example, if `/health` reports transcription model name `large-v3`, register `large-v3`.
- Use `AUTOMOM_REMOTE_AUTH_TOKEN` on real LANs where other users or machines can reach the worker port.
