from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Thread
from typing import Iterable, Sequence
from uuid import uuid4

import numpy as np
import soundfile as sf

from backend.app.config import SETTINGS
from backend.app.schemas import (
    MatchResponse,
    MatchResult,
    ProfileRefreshTask,
    VoiceProfile,
    VoiceProfileClipRange,
    VoiceProfileEmbedding,
    VoiceProfileSample,
)
from backend.pipeline.diarization import compute_profile_embedding, pyannote_audio_version
from backend.pipeline.remote_worker_client import RemoteWorkerClient


DEFAULT_THRESHOLD = 0.82
AMBIGUITY_MARGIN = 0.03


class VoiceProfileManager:
    def __init__(self) -> None:
        """! @brief Initialize the VoiceProfileManager instance.
        """
        SETTINGS.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_tasks: dict[str, ProfileRefreshTask] = {}
        self._refresh_lock = RLock()

    def load_mono_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """! @brief Load mono audio.
        @param audio_path Path to the audio file.
        @return Tuple produced by the operation.
        """
        audio, sample_rate = sf.read(str(audio_path), always_2d=False)
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        return mono, int(sample_rate)

    def purge_all(self) -> None:
        """! @brief Purge all.
        """
        for path in SETTINGS.profiles_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def list_profiles(self) -> list[VoiceProfile]:
        """! @brief List profiles.
        @return List produced by the operation.
        """
        profiles: list[VoiceProfile] = []
        for path in sorted(SETTINGS.profiles_dir.iterdir()):
            if not path.is_dir():
                continue
            manifest = path / "profile.json"
            if not manifest.exists():
                continue
            profiles.append(VoiceProfile(**json.loads(manifest.read_text(encoding="utf-8"))))
        return profiles

    def get_profile(self, profile_id: str) -> VoiceProfile:
        """! @brief Get profile.
        @param profile_id Identifier of the voice profile.
        @return Result produced by the operation.
        """
        path = self._profile_manifest(profile_id)
        if not path.exists():
            raise FileNotFoundError(profile_id)
        return VoiceProfile(**json.loads(path.read_text(encoding="utf-8")))

    def delete(self, profile_id: str) -> None:
        """! @brief Delete operation.
        @param profile_id Identifier of the voice profile.
        """
        shutil.rmtree(self._profile_dir(profile_id), ignore_errors=True)

    def save_profile_sample(
        self,
        *,
        name: str,
        source_audio_path: Path,
        clip_ranges: Sequence[tuple[float, float]],
        diarization_model_id: str,
        embedding_model_ref: str,
        source_job_id: str | None = None,
        source_speaker_id: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        audio_data: np.ndarray | None = None,
        sample_rate: int | None = None,
        compute_device: str = "auto",
        cuda_device_id: int = 0,
        embedding_vector: Sequence[float] | np.ndarray | None = None,
        library_version: str | None = None,
        engine_kind: str = "local_pyannote",
    ) -> VoiceProfile:
        """! @brief Save profile sample.
        @param name Value for name.
        @param source_audio_path Path to the source audio file.
        @param clip_ranges Value for clip ranges.
        @param diarization_model_id Value for diarization model id.
        @param embedding_model_ref Value for embedding model ref.
        @param source_job_id Value for source job id.
        @param source_speaker_id Value for source speaker id.
        @param threshold Value for threshold.
        @param audio_data Value for audio data.
        @param sample_rate Value for sample rate.
        @param compute_device Requested compute device preference.
        @param cuda_device_id CUDA device index to prefer when GPU execution is enabled.
        """
        if not clip_ranges:
            raise ValueError("At least one clip range is required to save a voice profile.")

        profile = self._find_by_name(name) or self._new_profile(name)
        profile_dir = self._profile_dir(profile.profile_id, profile.name)
        samples_dir = profile_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        if audio_data is None or sample_rate is None:
            audio_data, sample_rate = self.load_mono_audio(source_audio_path)

        sample_id = str(uuid4())
        sample_path = samples_dir / f"{sample_id}.wav"
        normalized_ranges = [(float(start_s), float(end_s)) for start_s, end_s in clip_ranges[:2]]
        self._write_reference_audio(sample_path, audio_data, sample_rate, normalized_ranges)
        embedding = (
            np.asarray(embedding_vector, dtype=np.float32)
            if embedding_vector is not None
            else self.compute_embedding(
                sample_path,
                diarization_model_id=diarization_model_id,
                embedding_model_ref=embedding_model_ref,
                compute_device=compute_device,
                cuda_device_id=cuda_device_id,
            )
        )
        embedding_entry = self._build_embedding_entry(
            diarization_model_id=diarization_model_id,
            embedding_model_ref=embedding_model_ref,
            vector=embedding,
            threshold=threshold,
            library_version=library_version,
            engine_kind=engine_kind,
        )
        sample = VoiceProfileSample(
            sample_id=sample_id,
            created_at=datetime.now(timezone.utc),
            source_job_id=source_job_id,
            source_speaker_id=source_speaker_id,
            reference_audio_path=str(sample_path),
            clip_ranges=[VoiceProfileClipRange(start_s=start_s, end_s=end_s) for start_s, end_s in normalized_ranges],
            embeddings=[embedding_entry],
        )
        updated_samples = [*profile.samples, sample]
        updated_profile = VoiceProfile(
            profile_id=profile.profile_id,
            name=profile.name,
            created_at=profile.created_at,
            updated_at=datetime.now(timezone.utc),
            sample_count=len(updated_samples),
            samples=updated_samples,
        )
        self._persist_profile(updated_profile)
        return updated_profile

    def compute_embedding(
        self,
        audio_path: Path,
        *,
        diarization_model_id: str,
        embedding_model_ref: str,
        compute_device: str = "auto",
        cuda_device_id: int = 0,
        segments: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """! @brief Compute embedding.
        @param audio_path Path to the audio file.
        @param diarization_model_id Value for diarization model id.
        @param embedding_model_ref Value for embedding model ref.
        @param compute_device Requested compute device preference.
        @param cuda_device_id CUDA device index to prefer when GPU execution is enabled.
        @param segments Segment collection processed by the operation.
        @return Result produced by the operation.
        """
        return compute_profile_embedding(
            audio_path,
            model_ref=embedding_model_ref,
            compute_device=compute_device,
            cuda_device_id=cuda_device_id,
            segments=segments,
        )

    def match(
        self,
        embedding: np.ndarray,
        *,
        diarization_model_id: str,
        embedding_model_ref: str,
        threshold: float = DEFAULT_THRESHOLD,
        profiles: Sequence[VoiceProfile] | None = None,
    ) -> MatchResponse:
        """! @brief Match operation.
        @param embedding Value for embedding.
        @param diarization_model_id Value for diarization model id.
        @param embedding_model_ref Value for embedding model ref.
        @param threshold Value for threshold.
        @param profiles Value for profiles.
        @return Result produced by the operation.
        """
        target_key = self._model_key(diarization_model_id, embedding_model_ref)
        normalized = self._normalize(embedding)
        per_profile: dict[str, MatchResult] = {}

        candidates = profiles if profiles is not None else self.list_profiles()
        for profile in candidates:
            best_for_profile: MatchResult | None = None
            for sample in profile.samples:
                for item in sample.embeddings:
                    if item.model_key != target_key:
                        continue
                    profile_embedding = np.asarray(item.vector, dtype=np.float32)
                    score = float(np.dot(normalized, self._normalize(profile_embedding)))
                    effective_threshold = max(threshold, float(item.threshold))
                    if score < effective_threshold:
                        continue
                    candidate = MatchResult(
                        profile_id=profile.profile_id,
                        sample_id=sample.sample_id,
                        name=profile.name,
                        score=score,
                        model_key=item.model_key,
                    )
                    if best_for_profile is None or candidate.score > best_for_profile.score:
                        best_for_profile = candidate
            if best_for_profile is not None:
                per_profile[profile.profile_id] = best_for_profile

        scored = sorted(per_profile.values(), key=lambda item: item.score, reverse=True)
        if not scored:
            return MatchResponse(best_match=None, ambiguous_matches=[])

        best = scored[0]
        ambiguous = [item for item in scored[1:] if abs(best.score - item.score) <= AMBIGUITY_MARGIN]
        return MatchResponse(best_match=best, ambiguous_matches=ambiguous)

    def rank_matches(
        self,
        embedding: np.ndarray,
        *,
        diarization_model_id: str,
        embedding_model_ref: str,
        profiles: Sequence[VoiceProfile] | None = None,
    ) -> list[MatchResult]:
        """! @brief Rank saved voice profiles without applying profile thresholds.
        @param embedding Probe embedding.
        @param diarization_model_id Value for diarization model id.
        @param embedding_model_ref Value for embedding model ref.
        @param profiles Optional profile collection.
        @return Ranked profile matches.
        """
        target_key = self._model_key(diarization_model_id, embedding_model_ref)
        normalized = self._normalize(embedding)
        per_profile: dict[str, MatchResult] = {}

        candidates = profiles if profiles is not None else self.list_profiles()
        for profile in candidates:
            best_for_profile: MatchResult | None = None
            for sample in profile.samples:
                for item in sample.embeddings:
                    if item.model_key != target_key:
                        continue
                    profile_embedding = np.asarray(item.vector, dtype=np.float32)
                    score = float(np.dot(normalized, self._normalize(profile_embedding)))
                    candidate = MatchResult(
                        profile_id=profile.profile_id,
                        sample_id=sample.sample_id,
                        name=profile.name,
                        score=score,
                        model_key=item.model_key,
                    )
                    if best_for_profile is None or candidate.score > best_for_profile.score:
                        best_for_profile = candidate
            if best_for_profile is not None:
                per_profile[profile.profile_id] = best_for_profile

        return sorted(per_profile.values(), key=lambda item: item.score, reverse=True)

    def start_refresh_task(
        self,
        *,
        diarization_execution: str,
        local_diarization_model_id: str | None,
        openai_diarization_model: str | None,
        embedding_model_ref: str | None = None,
        compute_device: str = "auto",
        cuda_device_id: int = 0,
        remote_model_config: dict[str, str] | None = None,
    ) -> ProfileRefreshTask:
        """! @brief Start refresh task.
        @param diarization_execution Value for diarization execution.
        @param local_diarization_model_id Value for local diarization model id.
        @param openai_diarization_model Value for openai diarization model.
        @param embedding_model_ref Value for embedding model ref.
        @param compute_device Requested compute device preference.
        @param cuda_device_id CUDA device index to prefer when GPU execution is enabled.
        @return Result produced by the operation.
        """
        now = datetime.now(timezone.utc)
        task = ProfileRefreshTask(
            task_id=str(uuid4()),
            status="queued",
            diarization_execution=diarization_execution,
            local_diarization_model_id=local_diarization_model_id,
            openai_diarization_model=openai_diarization_model,
            created_at=now,
            updated_at=now,
        )
        with self._refresh_lock:
            self._refresh_tasks[task.task_id] = task

        worker = Thread(
            target=self._run_refresh_task,
            kwargs={
                "task_id": task.task_id,
                "embedding_model_ref": embedding_model_ref,
                "compute_device": compute_device,
                "cuda_device_id": cuda_device_id,
                "remote_model_config": remote_model_config or {},
            },
            daemon=True,
        )
        worker.start()
        return task

    def get_refresh_task(self, task_id: str) -> ProfileRefreshTask:
        """! @brief Get refresh task.
        @param task_id Identifier of the background task.
        @return Result produced by the operation.
        """
        with self._refresh_lock:
            task = self._refresh_tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def _run_refresh_task(
        self,
        *,
        task_id: str,
        embedding_model_ref: str | None,
        compute_device: str,
        cuda_device_id: int,
        remote_model_config: dict[str, str],
    ) -> None:
        """! @brief Run refresh task.
        @param task_id Identifier of the background task.
        @param embedding_model_ref Value for embedding model ref.
        @param compute_device Requested compute device preference.
        @param cuda_device_id CUDA device index to prefer when GPU execution is enabled.
        """
        task = self.get_refresh_task(task_id)
        self._update_refresh_task(task_id, status="running", message="Refreshing saved profiles.")

        if task.diarization_execution == "api":
            self._update_refresh_task(
                task_id,
                status="completed",
                message=(
                    "OpenAI diarization uses saved reference clips directly. "
                    "No reusable embedding vectors were generated."
                ),
            )
            return

        if task.diarization_execution == "remote":
            if not task.local_diarization_model_id or not embedding_model_ref:
                self._update_refresh_task(task_id, status="failed", message="Missing remote diarization model details.")
                return
            client = RemoteWorkerClient(
                base_url=remote_model_config.get("base_url", ""),
                auth_token=remote_model_config.get("auth_token", ""),
                timeout_s=int(remote_model_config.get("timeout_s", "900") or "900"),
            )
            profiles = self.list_profiles()
            total_samples = sum(len(profile.samples) for profile in profiles)
            self._update_refresh_task(task_id, total_samples=total_samples, processed_samples=0)
            processed = 0
            try:
                for profile in profiles:
                    changed = False
                    updated_samples: list[VoiceProfileSample] = []
                    for sample in profile.samples:
                        if any(
                            item.model_key == self._model_key(task.local_diarization_model_id, embedding_model_ref)
                            for item in sample.embeddings
                        ):
                            updated_samples.append(sample)
                        else:
                            clip_ranges = [(item.start_s, item.end_s) for item in sample.clip_ranges]
                            embedding = client.embed(audio_path=Path(sample.reference_audio_path), clip_ranges=clip_ranges)
                            updated_samples.append(
                                sample.model_copy(
                                    update={
                                        "embeddings": [
                                            *sample.embeddings,
                                            self._build_embedding_entry(
                                                diarization_model_id=task.local_diarization_model_id,
                                                embedding_model_ref=embedding.embedding_model_ref or embedding_model_ref,
                                                vector=np.asarray(embedding.vector, dtype=np.float32),
                                                threshold=embedding.threshold,
                                                library_version=embedding.library_version,
                                                engine_kind=embedding.engine_kind,
                                            ),
                                        ]
                                    }
                                )
                            )
                            changed = True
                        processed += 1
                        self._update_refresh_task(task_id, processed_samples=processed)
                    if changed:
                        self._persist_profile(
                            VoiceProfile(
                                profile_id=profile.profile_id,
                                name=profile.name,
                                created_at=profile.created_at,
                                updated_at=datetime.now(timezone.utc),
                                sample_count=len(updated_samples),
                                samples=updated_samples,
                            )
                        )
                self._update_refresh_task(task_id, status="completed", message="Saved profile embeddings refreshed.")
            except Exception as exc:
                self._update_refresh_task(task_id, status="failed", message=str(exc))
            return

        if not task.local_diarization_model_id or not embedding_model_ref:
            self._update_refresh_task(task_id, status="failed", message="Missing local diarization model details.")
            return

        profiles = self.list_profiles()
        total_samples = sum(len(profile.samples) for profile in profiles)
        self._update_refresh_task(task_id, total_samples=total_samples, processed_samples=0)

        processed = 0
        try:
            for profile in profiles:
                changed = False
                updated_samples: list[VoiceProfileSample] = []
                for sample in profile.samples:
                    if any(
                        item.model_key == self._model_key(task.local_diarization_model_id, embedding_model_ref)
                        for item in sample.embeddings
                    ):
                        updated_samples.append(sample)
                    else:
                        vector = self.compute_embedding(
                            Path(sample.reference_audio_path),
                            diarization_model_id=task.local_diarization_model_id,
                            embedding_model_ref=embedding_model_ref,
                            compute_device=compute_device,
                            cuda_device_id=cuda_device_id,
                        )
                        updated_samples.append(
                            sample.model_copy(
                                update={
                                    "embeddings": [
                                        *sample.embeddings,
                                        self._build_embedding_entry(
                                            diarization_model_id=task.local_diarization_model_id,
                                            embedding_model_ref=embedding_model_ref,
                                            vector=vector,
                                        ),
                                    ]
                                }
                            )
                        )
                        changed = True
                    processed += 1
                    self._update_refresh_task(task_id, processed_samples=processed)
                if changed:
                    self._persist_profile(
                        VoiceProfile(
                            profile_id=profile.profile_id,
                            name=profile.name,
                            created_at=profile.created_at,
                            updated_at=datetime.now(timezone.utc),
                            sample_count=len(updated_samples),
                            samples=updated_samples,
                        )
                    )
            self._update_refresh_task(task_id, status="completed", message="Saved profile embeddings refreshed.")
        except Exception as exc:
            self._update_refresh_task(task_id, status="failed", message=str(exc))

    def _update_refresh_task(self, task_id: str, **changes: object) -> None:
        """! @brief Update refresh task.
        @param task_id Identifier of the background task.
        @param changes Value for changes.
        """
        with self._refresh_lock:
            task = self._refresh_tasks[task_id]
            payload = task.model_dump()
            payload.update(changes)
            payload["updated_at"] = datetime.now(timezone.utc)
            self._refresh_tasks[task_id] = ProfileRefreshTask(**payload)

    def _build_embedding_entry(
        self,
        *,
        diarization_model_id: str,
        embedding_model_ref: str,
        vector: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
        library_version: str | None = None,
        engine_kind: str = "local_pyannote",
    ) -> VoiceProfileEmbedding:
        """! @brief Build embedding entry.
        @param diarization_model_id Value for diarization model id.
        @param embedding_model_ref Value for embedding model ref.
        @param vector Value for vector.
        @param threshold Value for threshold.
        @return Result produced by the operation.
        """
        now = datetime.now(timezone.utc)
        return VoiceProfileEmbedding(
            embedding_id=str(uuid4()),
            engine_kind=str(engine_kind),
            diarization_model_id=diarization_model_id,
            embedding_model_ref=embedding_model_ref,
            library_version=library_version or pyannote_audio_version(),
            threshold=threshold,
            vector=self._normalize(vector).tolist(),
            created_at=now,
            model_key=self._model_key(diarization_model_id, embedding_model_ref),
        )

    def _persist_profile(self, profile: VoiceProfile) -> None:
        """! @brief Persist profile.
        @param profile Value for profile.
        """
        profile_dir = self._profile_dir(profile.profile_id, profile.name)
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest = profile_dir / "profile.json"
        manifest.write_text(json.dumps(profile.model_dump(mode="json"), indent=2), encoding="utf-8")
        self._delete_stale_paths(profile.profile_id, keep=profile_dir)

    def _new_profile(self, name: str) -> VoiceProfile:
        """! @brief New profile.
        @param name Value for name.
        @return Result produced by the operation.
        """
        now = datetime.now(timezone.utc)
        return VoiceProfile(
            profile_id=str(uuid4()),
            name=name,
            created_at=now,
            updated_at=now,
            sample_count=0,
            samples=[],
        )

    def _find_by_name(self, name: str) -> VoiceProfile | None:
        """! @brief Find by name.
        @param name Value for name.
        @return Result produced by the operation.
        """
        needle = name.strip().lower()
        for profile in self.list_profiles():
            if profile.name.strip().lower() == needle:
                return profile
        return None

    def _write_reference_audio(
        self,
        output_path: Path,
        audio_data: np.ndarray,
        sample_rate: int,
        clip_ranges: Sequence[tuple[float, float]],
    ) -> None:
        """! @brief Write reference audio.
        @param output_path Path to the output file.
        @param audio_data Value for audio data.
        @param sample_rate Value for sample rate.
        @param clip_ranges Value for clip ranges.
        """
        clips: list[np.ndarray] = []
        for start_s, end_s in clip_ranges:
            start_idx = max(0, int(start_s * sample_rate))
            end_idx = max(start_idx, int(end_s * sample_rate))
            clip = np.asarray(audio_data[start_idx:end_idx], dtype=np.float32)
            if clip.size:
                clips.append(clip)
        if not clips:
            raise ValueError("Unable to build profile reference audio from empty clips.")
        silence = np.zeros(int(sample_rate * 0.15), dtype=np.float32)
        merged = clips[0]
        for clip in clips[1:]:
            merged = np.concatenate([merged, silence, clip]).astype(np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, merged, sample_rate)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """! @brief Normalize operation.
        @param vector Value for vector.
        @return Result produced by the operation.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _slugify_name(name: str) -> str:
        """! @brief Slugify name.
        @param name Value for name.
        @return str result produced by the operation.
        """
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return normalized[:60] or "speaker"

    @classmethod
    def _profile_dir(cls, profile_id: str, name: str | None = None) -> Path:
        """! @brief Profile dir.
        @param profile_id Identifier of the voice profile.
        @param name Value for name.
        @return Path result produced by the operation.
        """
        if name:
            return SETTINGS.profiles_dir / f"{cls._slugify_name(name)}--{profile_id}"
        named = sorted(SETTINGS.profiles_dir.glob(f"*--{profile_id}"))
        if named:
            return named[0]
        return SETTINGS.profiles_dir / profile_id

    @classmethod
    def _profile_manifest(cls, profile_id: str) -> Path:
        """! @brief Profile manifest.
        @param profile_id Identifier of the voice profile.
        @return Path result produced by the operation.
        """
        return cls._profile_dir(profile_id) / "profile.json"

    @classmethod
    def _delete_stale_paths(cls, profile_id: str, keep: Path) -> None:
        """! @brief Delete stale paths.
        @param profile_id Identifier of the voice profile.
        @param keep Value for keep.
        """
        candidates = {SETTINGS.profiles_dir / profile_id, *SETTINGS.profiles_dir.glob(f"*--{profile_id}")}
        for candidate in candidates:
            if candidate == keep:
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.unlink(missing_ok=True)

    @staticmethod
    def _model_key(diarization_model_id: str, embedding_model_ref: str) -> str:
        """! @brief Model key.
        @param diarization_model_id Value for diarization model id.
        @param embedding_model_ref Value for embedding model ref.
        @return str result produced by the operation.
        """
        return f"local_pyannote::{diarization_model_id}::{embedding_model_ref}"


VOICE_PROFILE_MANAGER = VoiceProfileManager()
