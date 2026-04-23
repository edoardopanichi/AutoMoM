from __future__ import annotations

from pathlib import Path

from backend.app.schemas import DiarizationLocalModel
from backend.models.local_catalog import LOCAL_MODEL_CATALOG


DEFAULT_LOCAL_DIARIZATION_MODEL_ID = "pyannote-community-1"


def list_local_diarization_models() -> list[DiarizationLocalModel]:
    """! @brief List local diarization models.
    @return List produced by the operation.
    """
    payload = LOCAL_MODEL_CATALOG.list_stage("diarization")
    models: list[DiarizationLocalModel] = []
    for item in payload.models:
        pipeline_path = item.config.get("pipeline_path", "")
        base_url = item.config.get("base_url", "")
        embedding_ref = item.config.get("embedding_model_ref", "")
        profile_model_ref = item.config.get("profile_model_ref", "") or item.model_id
        if item.location == "remote":
            if not base_url:
                continue
            models.append(
                DiarizationLocalModel(
                    model_id=item.model_id,
                    name=item.name,
                    location=item.location,
                    runtime=item.runtime,
                    base_url=base_url,
                    profile_model_ref=profile_model_ref,
                    embedding_model_ref=embedding_ref,
                )
            )
            continue
        if not pipeline_path:
            continue
        path = Path(pipeline_path).expanduser()
        if path.exists():
            embedding_dir = path.resolve().parent / "embedding"
            if embedding_dir.exists() and not embedding_ref:
                embedding_ref = str(embedding_dir)
        models.append(
            DiarizationLocalModel(
                model_id=item.model_id,
                name=item.name,
                location=item.location,
                runtime=item.runtime,
                pipeline_path=pipeline_path,
                profile_model_ref=profile_model_ref,
                embedding_model_ref=embedding_ref,
            )
        )
    return models


def resolve_local_diarization_model(model_id: str | None) -> DiarizationLocalModel:
    """! @brief Resolve local diarization model.
    @param model_id Identifier of the target model.
    @return Result produced by the operation.
    """
    normalized = (model_id or "").strip() or LOCAL_MODEL_CATALOG.list_stage("diarization").selected_model_id
    for item in list_local_diarization_models():
        if item.model_id == normalized:
            return item
    raise ValueError(f"Unknown local diarization model: {normalized}")
