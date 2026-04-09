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

LocalModelStage = Literal["diarization", "transcription", "formatter"]
LocalModelRuntime = Literal["pyannote", "whisper.cpp", "faster-whisper", "ollama", "command"]


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


class FormatterModelRequest(BaseModel):
    model_tag: str


class FormatterModelResponse(BaseModel):
    model_tag: str


class LocalModelRecord(BaseModel):
    model_id: str
    stage: LocalModelStage
    runtime: LocalModelRuntime
    name: str
    installed: bool
    languages: list[str] = Field(default_factory=list)
    notes: str = ""
    config: dict[str, str] = Field(default_factory=dict)
    validation_error: str | None = None


class LocalModelCatalogResponse(BaseModel):
    defaults: dict[LocalModelStage, str]
    models: list[LocalModelRecord]


class LocalStageModelResponse(BaseModel):
    stage: LocalModelStage
    selected_model_id: str
    models: list[LocalModelRecord]


class LocalModelRegistrationRequest(BaseModel):
    stage: LocalModelStage
    runtime: LocalModelRuntime
    model_id: str | None = None
    name: str
    languages: list[str] = Field(default_factory=list)
    notes: str = ""
    config: dict[str, str] = Field(default_factory=dict)
    set_as_default: bool = False


class LocalModelDefaultRequest(BaseModel):
    stage: LocalModelStage
    model_id: str


class TemplateSection(BaseModel):
    heading: str
    required: bool = True
    allow_prefix: bool = False
    empty_value: str = "None"


class TemplateDefinition(BaseModel):
    template_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    prompt_block: str
    sections: list[TemplateSection] = Field(default_factory=list)


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


class SpeakerProfileMatch(BaseModel):
    profile_id: str
    sample_id: str | None = None
    name: str
    score: float
    model_key: str | None = None
    status: Literal["matched", "ambiguous"]
    ambiguous_names: list[str] = Field(default_factory=list)


class SpeakerState(BaseModel):
    speaker_id: str
    suggested_name: str | None = None
    matched_profile: SpeakerProfileMatch | None = None
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
    stage_detail: str | None = None
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
    updated_at: datetime | None = None
    sample_count: int = 0
    samples: list["VoiceProfileSample"] = Field(default_factory=list)


class VoiceProfileClipRange(BaseModel):
    start_s: float
    end_s: float


class VoiceProfileEmbedding(BaseModel):
    embedding_id: str
    engine_kind: Literal["local_pyannote"]
    diarization_model_id: str
    embedding_model_ref: str
    library_version: str
    threshold: float
    vector: list[float]
    created_at: datetime
    model_key: str


class VoiceProfileSample(BaseModel):
    sample_id: str
    created_at: datetime
    source_job_id: str | None = None
    source_speaker_id: str | None = None
    reference_audio_path: str
    clip_ranges: list[VoiceProfileClipRange] = Field(default_factory=list)
    embeddings: list[VoiceProfileEmbedding] = Field(default_factory=list)


class CreateVoiceProfileRequest(BaseModel):
    name: str
    audio_path: str | None = None
    diarization_model_id: str | None = None


class MatchResult(BaseModel):
    profile_id: str
    sample_id: str | None = None
    name: str
    score: float
    model_key: str | None = None


class MatchResponse(BaseModel):
    best_match: MatchResult | None
    ambiguous_matches: list[MatchResult] = Field(default_factory=list)


class DiarizationLocalModel(BaseModel):
    model_id: str
    name: str
    pipeline_path: str
    embedding_model_ref: str


class DiarizationLocalModelResponse(BaseModel):
    selected_model_id: str
    models: list[DiarizationLocalModel]


class ProfileRefreshRequest(BaseModel):
    diarization_execution: Literal["local", "api"] = "local"
    local_diarization_model_id: str | None = None
    openai_diarization_model: str | None = None


class ProfileRefreshTask(BaseModel):
    task_id: str
    status: Literal["queued", "running", "completed", "failed"]
    diarization_execution: Literal["local", "api"]
    local_diarization_model_id: str | None = None
    openai_diarization_model: str | None = None
    total_samples: int = 0
    processed_samples: int = 0
    created_at: datetime
    updated_at: datetime
    message: str | None = None
