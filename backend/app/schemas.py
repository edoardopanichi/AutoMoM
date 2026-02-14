from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


PipelineStage = Literal[
    "Validate/Normalize",
    "VAD",
    "Diarization",
    "Snippet extraction",
    "Speaker naming",
    "Transcription",
    "Transcript assembly",
    "MoM formatting",
    "Export",
]


class ModelStatus(BaseModel):
    model_id: str
    name: str
    size_mb: int
    source: str
    required_disk_mb: int
    installed: bool
    consent_granted: bool
    file_path: str
    checksum_sha256: str | None = None
    download_url: str | None = None


class ModelConsentRequest(BaseModel):
    model_id: str
    approved: bool


class ModelDownloadRequest(BaseModel):
    model_id: str


class TemplateDefinition(BaseModel):
    template_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    content: str
    prompt_block: str


class TemplateSummary(BaseModel):
    template_id: str
    name: str
    version: str
    description: str


class SpeakerSnippet(BaseModel):
    speaker_id: str
    snippet_path: str
    start_s: float
    end_s: float


class SpeakerState(BaseModel):
    speaker_id: str
    suggested_name: str | None = None
    snippets: list[SpeakerSnippet] = Field(default_factory=list)


class JobSpeakerInfo(BaseModel):
    detected_speakers: int
    speakers: list[SpeakerState]


class SpeakerMappingItem(BaseModel):
    speaker_id: str
    name: str
    save_voice_profile: bool = False


class SubmitSpeakerMappingRequest(BaseModel):
    mappings: list[SpeakerMappingItem]


class JobState(BaseModel):
    job_id: str
    status: Literal[
        "created",
        "running",
        "waiting_speaker_input",
        "completed",
        "failed",
        "cancelled",
    ]
    created_at: datetime
    updated_at: datetime
    current_stage: PipelineStage | None = None
    stage_percent: float = 0.0
    overall_percent: float = 0.0
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    speaker_info: JobSpeakerInfo | None = None
    transcript_segment_progress: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobState]


class VoiceProfile(BaseModel):
    profile_id: str
    name: str
    created_at: datetime
    embedding: list[float]
    model_version: str
    threshold: float


class CreateVoiceProfileRequest(BaseModel):
    name: str
    audio_path: str | None = None


class MatchResult(BaseModel):
    profile_id: str
    name: str
    score: float


class MatchResponse(BaseModel):
    best_match: MatchResult | None
    ambiguous_matches: list[MatchResult] = Field(default_factory=list)
