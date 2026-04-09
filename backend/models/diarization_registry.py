from __future__ import annotations

from pathlib import Path

from backend.app.config import SETTINGS
from backend.app.schemas import DiarizationLocalModel


DEFAULT_LOCAL_DIARIZATION_MODEL_ID = "pyannote-community-1"


def _default_pipeline_path() -> Path:
    """! @brief Default pipeline path.
    @return Path result produced by the operation.
    """
    return Path(SETTINGS.diarization_pipeline_path or SETTINGS.diarization_model_path).expanduser()


def _default_embedding_ref(pipeline_path: Path) -> str:
    """! @brief Default embedding ref.
    @param pipeline_path Value for pipeline path.
    @return str result produced by the operation.
    """
    embedding_dir = pipeline_path.resolve().parent / "embedding"
    if embedding_dir.exists():
        return str(embedding_dir)
    return SETTINGS.diarization_embedding_model


def list_local_diarization_models() -> list[DiarizationLocalModel]:
    """! @brief List local diarization models.
    @return List produced by the operation.
    """
    pipeline_path = _default_pipeline_path()
    return [
        DiarizationLocalModel(
            model_id=DEFAULT_LOCAL_DIARIZATION_MODEL_ID,
            name="Pyannote Community-1",
            pipeline_path=str(pipeline_path),
            embedding_model_ref=_default_embedding_ref(pipeline_path),
        )
    ]


def resolve_local_diarization_model(model_id: str | None) -> DiarizationLocalModel:
    """! @brief Resolve local diarization model.
    @param model_id Identifier of the target model.
    @return Result produced by the operation.
    """
    normalized = (model_id or DEFAULT_LOCAL_DIARIZATION_MODEL_ID).strip() or DEFAULT_LOCAL_DIARIZATION_MODEL_ID
    for item in list_local_diarization_models():
        if item.model_id == normalized:
            return item
    raise ValueError(f"Unknown local diarization model: {normalized}")
