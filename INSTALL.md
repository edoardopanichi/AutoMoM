# AutoMoM Installation Guide (Linux)

This guide is for Ubuntu/Debian-like Linux systems and covers both CPU-only and CUDA-enabled setups.

## 1. Install system dependencies

### 1.1 Common dependencies (CPU and CUDA)

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg curl git
```

Install Ollama (formatter backend default):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 1.2 CUDA path only (optional)

If you want GPU acceleration for local diarization/transcription:

1. Install a compatible NVIDIA driver.
2. Install CUDA toolkit/runtime supported by your PyTorch stack.
3. Verify GPU visibility:

```bash
nvidia-smi
```

If you do not have a compatible NVIDIA GPU, stay on the CPU path.

## 2. Clone and enter the repository

```bash
git clone <your-repo-url> AutoMoM
cd AutoMoM
```

## 3. Create environment configuration

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- `AUTOMOM_DIARIZATION_MODEL` (or `AUTOMOM_DIARIZATION_PIPELINE`) to your local pyannote `config.yaml` path.
- `AUTOMOM_TRANSCRIPTION_MODEL` if you do not use the default `data/models/transcription/model.gguf` path.
- `AUTOMOM_COMPUTE_DEVICE`:
  - `cpu` for CPU-only deployments.
  - `cuda` for CUDA deployments.
  - `auto` to let runtime decide.

## 4. Prepare local models

### 4.1 Diarization model (pyannote)

AutoMoM local diarization expects a local pyannote pipeline directory that contains `config.yaml`.

Set one of these in `.env`:

- `AUTOMOM_DIARIZATION_PIPELINE=/abs/path/to/.../config.yaml`
- `AUTOMOM_DIARIZATION_MODEL=/abs/path/to/.../config.yaml`

### 4.2 Transcription model (whisper.cpp)

You need:

- A working `whisper-cli` binary available in `PATH`, or set `AUTOMOM_TRANSCRIPTION_BIN`.
- A local transcription model file (for example `.gguf`) at `AUTOMOM_TRANSCRIPTION_MODEL`.

### 4.3 Formatter model (Ollama)

Pull your formatter model tag (default from `.env`):

```bash
ollama pull qwen2.5:3b-instruct-q5_K_M
```

If you change the tag, also update `AUTOMOM_FORMATTER_OLLAMA_MODEL`.

## 5. Start AutoMoM

```bash
./scripts/run_automom.sh
```

Notes:

- The script creates `.venv`, installs Python dependencies, and starts the API/UI.
- Ollama startup checks run only when `AUTOMOM_FORMATTER_BACKEND=ollama`.

## 6. Verify startup

In another terminal:

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/system/startup-check
```

Open UI:

- `http://127.0.0.1:8000`

In Settings, verify local model entries are marked installed, then run a short test audio job.

## 7. Run tests (recommended)

```bash
source .venv/bin/activate
pytest backend/tests -q
```

## 8. Optional: OpenAI execution path

To run diarization/transcription/formatter stages via OpenAI, provide an API key in the job form and set the relevant stage execution mode to `api`.

## 9. Optional: Remote worker setup

For LAN workers (remote diarization/transcription), see:

- `backend/worker/README.md`

## Troubleshooting

- `Pyannote pipeline path does not exist`: verify `AUTOMOM_DIARIZATION_PIPELINE` points to a real local `config.yaml`.
- `ASR binary is not configured or not found`: set `AUTOMOM_TRANSCRIPTION_BIN` to a valid `whisper-cli` executable.
- `ASR model file is missing`: set `AUTOMOM_TRANSCRIPTION_MODEL` to an existing local model file.
- `Ollama API unreachable`: start `ollama serve` manually, or keep `AUTOMOM_OLLAMA_AUTOSTART=1`.
- CUDA requested but not active: verify driver/CUDA installation and your `whisper-cli` GPU-capable build.
