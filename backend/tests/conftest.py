from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import SETTINGS, ensure_directories
from backend.app.job_store import JOB_STORE


@pytest.fixture
def isolated_settings(tmp_path: Path):
    """! @brief Isolated settings.
    @param tmp_path Value for tmp path.
    @return Result produced by the operation.
    """
    original_values = {
        "data_dir": SETTINGS.data_dir,
        "jobs_dir": SETTINGS.jobs_dir,
        "models_dir": SETTINGS.models_dir,
        "templates_dir": SETTINGS.templates_dir,
        "profiles_dir": SETTINGS.profiles_dir,
        "uploads_dir": SETTINGS.uploads_dir,
    }

    object.__setattr__(SETTINGS, "data_dir", tmp_path / "data")
    object.__setattr__(SETTINGS, "jobs_dir", tmp_path / "data" / "jobs")
    object.__setattr__(SETTINGS, "models_dir", tmp_path / "data" / "models")
    object.__setattr__(SETTINGS, "templates_dir", tmp_path / "data" / "templates")
    object.__setattr__(SETTINGS, "profiles_dir", tmp_path / "data" / "profiles")
    object.__setattr__(SETTINGS, "uploads_dir", tmp_path / "data" / "uploads")

    ensure_directories()
    JOB_STORE._jobs.clear()  # type: ignore[attr-defined]

    yield

    JOB_STORE._jobs.clear()  # type: ignore[attr-defined]
    for key, value in original_values.items():
        object.__setattr__(SETTINGS, key, value)
    ensure_directories()
