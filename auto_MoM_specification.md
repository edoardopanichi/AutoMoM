# AutoMoM (Automatic Minutes of Meeting) – Specification

## 0. Summary
AutoMoM is a fully local (offline) tool that converts a phone-recorded meeting audio file into an English Minutes of Meeting (MoM) document.

Pipeline:
1. Ingest audio (upload via web UI) and normalize it.
2. Detect speech regions (optional but recommended) and run speaker diarization.
3. Show how many speakers are detected.
4. Let the user listen to short snippets per speaker and assign names.
5. Support **local voice profiles** so known speakers can be auto-identified in future meetings.
6. Transcribe diarized segments locally using **whisper.cpp**.
7. Reorganize and summarize the transcript into an English MoM using a local text model and a user-selected template.
8. Show a stage-based progress bar throughout the run.
9. Export **Markdown** (only) with the full transcript available behind a toggle/section.

Non-realtime by design. Single-user initially. Runs on a normal laptop even without GPU (slower is acceptable).

---

## 1. Goals
### 1.1 Functional goals
- Offline processing after models are available locally.
- Speaker-separated transcript and MoM output.
- Speaker count display.
- Interactive speaker naming with audio snippet playback.
- Local voice profiles to auto-name recurring speakers.
- Multilingual input audio supported to the extent that the local ASR model supports it.
- **Final MoM always in English**, regardless of input language.
- Template-driven MoM formatting with default template included.
- Multiple templates selectable from the web interface.
- Clear progress reporting.

### 1.2 Non-functional goals
- Works on Windows, macOS, and Linux (as much as practical).
- Robust error handling and resumable jobs (within single-user scope).
- Comprehensive unit tests and an end-to-end test.
- Model management: check availability, prompt for each download, download only after explicit permission.

---

## 2. Definitions
- **Diarization**: split audio into time segments and assign anonymous speaker labels (SPEAKER_0, SPEAKER_1, …).
- **Speaker identification**: map a diarization cluster to a known person name using voice embeddings and stored profiles.
- **ASR**: automatic speech recognition (transcription).
- **Formatter model**: local LLM used to produce structured MoM from transcript + template.

---

## 3. Supported input and output
### 3.1 Input
- Audio file uploaded via web UI.
- Supported formats (minimum): WAV, MP3, M4A.
- Duration:
  - Average: 1 hour
  - Maximum: 3 hours
- Speakers:
  - Typical: 5
  - Range: 2 to 20

### 3.2 Output
- Markdown document containing:
  - MoM in the template structure
  - A transcript section included but **hidden behind a toggle** (implemented as a collapsible HTML block inside Markdown, or a clearly marked optional section).
- Optional intermediate artifacts saved locally (JSON/RTTM, segment audio files, logs).

---

## 4. High-level architecture
### 4.1 Components
1. **Web UI** (local web app)
   - Upload audio
   - Show diarization results and speaker count
   - Play speaker snippets and assign names
   - Manage and select templates
   - Show stage-based progress bar
   - Preview and download Markdown
   - Manage voice profiles (create/edit/delete)

2. **Backend API + Orchestrator**
   - Runs the pipeline as a multi-stage job
   - Exposes job status and progress via WebSocket/SSE
   - Stores artifacts locally

3. **Model Manager**
   - Verifies local models are present
   - Prompts user per download and records consent
   - Downloads and verifies checksums

4. **Local Storage**
   - Projects/jobs directory
   - Model cache directory
   - Templates directory
   - Voice profiles directory

### 4.2 Data flow
Upload → Normalize → (VAD) → Diarize → Speaker naming/ID → Segment extraction → Transcribe with whisper.cpp → Assemble transcript → Summarize/structure with formatter model → Render Markdown → Export.

---

## 5. Technology choices
### 5.1 Recommended languages
- **Python**: orchestration, diarization integration, audio processing wrappers, templating, tests.
- **C/C++**: compilation and execution of **whisper.cpp**.
- **TypeScript**: web UI (React recommended).

Rationale: Python ecosystem for diarization/audio; whisper.cpp is C++-based; TypeScript for a maintainable web UI.

### 5.2 Backend framework
- **FastAPI** for HTTP API.
- WebSocket or SSE for live progress updates.
- Local job execution options:
  - Baseline: single-process background tasks (single user).
  - Upgrade path: local queue (SQLite-backed or Redis) if reliability across restarts is needed.

### 5.3 Audio utilities
- `ffmpeg` for transcoding.
- `soundfile`/`librosa`/`pydub` or equivalent to handle segmentation and snippet creation.

### 5.4 Formatter runtime
- Local LLM inference via **llama.cpp** (quantized model, CPU-friendly).

---

## 6. Local models (three required)
### 6.1 Speaker detection (diarization)
Baseline requirement: offline diarization that can run on CPU.

Recommended default:
- **pyannote.audio diarization pipeline** (accuracy-first baseline; CPU supported but slower).

Recommended optional acceleration:
- **Silero VAD** pre-pass to remove long non-speech regions and speed up diarization and ASR.

Outputs:
- `diarization.json` containing speaker segments:
  - `speaker_id` (anonymous label)
  - `start_s`, `end_s`
  - optional confidence scores

### 6.2 Transcription model
- **whisper.cpp** for local ASR using compatible local ASR weights.

Outputs:
- `segments_transcript.json`:
  - `speaker_id` / `speaker_name`
  - `start_s`, `end_s`
  - `text`
  - model/version metadata

### 6.3 Formatter model (reorganization to MoM)
- A local, quantized instruction model runnable on CPU (3B to 8B class).

Requirements:
- Must accept:
  - full transcript (chronological)
  - speaker list (names)
  - user-selected template
  - settings: “final output must be English”
- Must generate:
  - structured MoM in the template sections
  - action items, decisions, open questions (best-effort)

---

## 7. Voice profiles (local speaker identification)
### 7.1 Purpose
Allow AutoMoM to auto-assign speaker names in future meetings by matching diarization clusters to stored voice profiles.

### 7.2 Profile creation
Two supported flows:
1. **From the current meeting**
   - After diarization, user labels a speaker.
   - System offers “Save as voice profile” for that speaker.
   - System aggregates several segments from that cluster to compute a stable embedding.

2. **From a dedicated enrollment audio**
   - User uploads a short voice sample for a person.
   - System extracts embedding(s) and stores the profile.

### 7.3 Embedding model
- Use a CPU-capable speaker embedding model (e.g., ECAPA-TDNN class) via a local inference library.

### 7.4 Matching logic
- Compute embedding per diarization cluster.
- Compare to known profile embeddings (cosine similarity).
- If similarity >= threshold → auto-assign that name.
- If ambiguous (multiple above threshold or too close) → keep anonymous and prompt user.

### 7.5 Storage format
- `profiles/<profile_id>.json`:
  - name
  - created_at
  - embedding vectors (or references)
  - model version and threshold used

---

## 8. Templates
### 8.1 Default template (must ship)
AutoMoM ships with a default MoM template with these sections:
- **Meeting Info**
  - Title
  - Date/time
  - Participants
- **Executive Summary**
- **Agenda** (inferred or “Not detected”)
- **Discussion Summary** (bullets)
- **Decisions** (bullets)
- **Action Items**
  - Owner, task, due date (best-effort extraction)
- **Open Questions / Risks**
- **Transcript (optional)**
  - Hidden behind a toggle

### 8.2 Multiple templates
- Templates stored locally in `templates/`.
- UI allows:
  - pick a template for the run
  - set a default template
  - create/edit templates (recommended)
  - import/export templates (recommended)

### 8.3 Template format
- Markdown + placeholders (Jinja2-style) plus a “prompt block” for the formatter model.
- Templates must be versioned and testable.

---

## 9. User interface requirements
### 9.1 Pages
1. **Home / New Job**
   - Upload audio
   - Select template
   - Configure language handling (default: auto-detect)
   - Start job

2. **Progress page**
   - Stage list + progress bar
   - Logs panel
   - Ability to cancel

3. **Speaker naming** (blocking step)
   - Shows detected number of speakers
   - For each speaker:
     - Play 1–3 snippets
     - Name input
     - “Save voice profile” toggle

4. **Result**
   - MoM preview
   - Download Markdown
   - Transcript toggle

5. **Settings**
   - Model manager (installed models, disk usage)
   - Templates manager
   - Voice profiles manager

### 9.2 Progress reporting
Progress must be visible as:
- overall percent
- current stage
- stage percent
- segment-level progress for transcription (N of M segments)

Progress stages:
1. Validate/Normalize
2. VAD (optional)
3. Diarization
4. Snippet extraction
5. Speaker naming (user action)
6. Transcription
7. Transcript assembly
8. MoM formatting
9. Export

---

## 10. Model availability checks and permission-gated downloads
### 10.1 Checks
On program start and before running a job:
- Verify diarization model present
- Verify whisper.cpp weights present
- Verify formatter model present

### 10.2 Permission-gated downloads
If any are missing:
- Show a UI prompt listing each model separately:
  - model name
  - size
  - source
  - disk space required
- User must approve each download individually.
- Downloads must support:
  - resume/retry
  - checksum verification
- If user declines:
  - job cannot start; show a clear message.

---

## 11. Pipeline specification (detailed)
### 11.1 Stage A: Audio normalization
Inputs: uploaded file
Outputs:
- `audio_normalized.wav` (16 kHz, mono)
- metadata: duration, sample rate, format
Errors:
- unsupported file format
- decode errors

### 11.2 Stage B: Voice activity detection (recommended)
Inputs: normalized audio
Outputs:
- speech-only segments or a VAD mask
- optional “trimmed” audio for faster downstream

### 11.3 Stage C: Diarization
Inputs: normalized (or VAD-trimmed) audio
Outputs:
- diarization segments
- detected speaker count
- quality metrics (optional)

### 11.4 Stage D: Snippet extraction for labeling
For each speaker cluster:
- pick representative snippets (3–8 s) from high-energy, clean segments
- generate `snippets/<speaker_id>_k.wav`

### 11.5 Stage E: Speaker naming and profile linking
- UI collects `speaker_id → name` mapping.
- If voice profile creation enabled:
  - compute speaker embedding from multiple segments
  - store/update profile

### 11.6 Stage F: Segment extraction for transcription
- Cut diarized segments to WAV files for ASR.
- Apply padding rules (e.g., ±0.2 s) to reduce word truncation.

### 11.7 Stage G: Transcription via whisper.cpp
- Run whisper.cpp on each segment.
- Collect text and timestamps.
- Merge contiguous segments of same speaker where gaps are small.

### 11.8 Stage H: Transcript assembly
Build chronological transcript structure:
- speaker name
- timestamps
- text

### 11.9 Stage I: MoM formatting (always English)
Inputs:
- transcript
- speaker list
- selected template
- rule: final output language = English

Outputs:
- `mom.md`
- `mom_structured.json` (optional)

### 11.10 Stage J: Export
- Provide Markdown download.
- Store job artifacts for later retrieval.

---

## 12. Performance expectations
- CPU-only operation acceptable.
- For 1 hour audio, expect minutes to tens of minutes depending on laptop.
- For 3 hours and up to 20 speakers, allow longer; progress reporting must remain responsive.
- The full code runs by using one command only! So make a script that activates all the necessary background processes
---

## 13. Security and privacy
- Local-only by default.
- No external network calls except optional model downloads (permission gated).
- All uploaded audio and outputs stored locally; user can delete jobs.

---

## 14. Testing requirements
### 14.1 Unit tests (mandatory)
- Audio normalization and format conversion
- VAD wrapper (mask correctness)
- Diarization wrapper and output parsing
- Snippet selection and extraction
- Voice profile embedding generation and matching
- Segment stitching/merging logic
- whisper.cpp invocation wrapper (mocked in unit tests)
- Template rendering
- Formatter prompt assembly
- Permission-gated model download flows

### 14.2 Integration tests (mandatory)
- End-to-end job run on a short sample audio (2 speakers, 1–3 minutes)
- Snapshot/golden-file testing for JSON outputs and Markdown structure
- Before finishing, do a long test on the provided audio file stored in the folder audio_trace_for_testing. This a audio trace of almost 3 hours. 

### 14.3 UI tests (recommended)
- Upload and start job
- Speaker naming flow
- Template switching
- Progress updates
- Download Markdown

---

## 15. Repository structure (recommended)
- `backend/`
  - `app/` (FastAPI)
  - `pipeline/` (stages)
  - `models/` (model manager)
  - `profiles/` (speaker ID)
  - `templates/`
  - `tests/`
- `frontend/`
  - React app
  - Playwright tests
- `native/`
  - whisper.cpp build scripts and binaries
- `scripts/`
  - installer scripts

---

## 16. Setup and packaging
### 16.1 Python environment
- Create venv:
  - `python -m venv .venv`
- Provide:
  - `requirements.txt` pinned
  - optional `requirements-dev.txt` for test tooling

### 16.2 System dependencies
- `ffmpeg`
- build tools for whisper.cpp (platform-specific)

### 16.3 First-run experience
- Launch AutoMoM (with one command line command only!)
- “Model check” screen
- Prompt user per missing model download
- Once models are available, allow job creation

---

## 17. Open decisions to confirm (optional)
- Whether to ship an embedded database (SQLite) to store jobs, templates, profiles, and settings.
- Whether to allow background processing of multiple jobs (queue) or strictly one at a time.
- Whether to implement transcript toggle using Markdown + HTML details/summary.

## 18. Documentation
- Write a clear README to explain how to use the program and the main features and dependencies
- Write an AGENT.md for agents to understand the code and work on it
