from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

import numpy as np
import soundfile as sf

from backend.app.config import SETTINGS
from backend.app.schemas import MatchResponse, MatchResult, VoiceProfile


EMBEDDING_VERSION = "simple-spectrum-v1"
DEFAULT_THRESHOLD = 0.82


class VoiceProfileManager:
    def __init__(self) -> None:
        SETTINGS.profiles_dir.mkdir(parents=True, exist_ok=True)

    def load_mono_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(str(audio_path), always_2d=False)
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        return mono, int(sample_rate)

    def list_profiles(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        for path in sorted(SETTINGS.profiles_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            profiles.append(VoiceProfile(**payload))
        return profiles

    def get_profile(self, profile_id: str) -> VoiceProfile:
        path = self._path(profile_id)
        if not path.exists():
            raise FileNotFoundError(profile_id)
        return VoiceProfile(**json.loads(path.read_text(encoding="utf-8")))

    def delete(self, profile_id: str) -> None:
        self._path(profile_id).unlink(missing_ok=True)

    def upsert_profile(self, name: str, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> VoiceProfile:
        normalized = self._normalize(embedding)
        for existing in self.list_profiles():
            if existing.name.strip().lower() == name.strip().lower():
                payload = existing.model_dump(mode="json")
                existing_embedding = np.array(existing.embedding, dtype=np.float32)
                sample_count = max(1, int(existing.sample_count))
                merged = self._normalize(((existing_embedding * sample_count) + normalized) / (sample_count + 1))
                payload["embedding"] = merged.tolist()
                payload["threshold"] = threshold
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                payload["sample_count"] = sample_count + 1
                profile = VoiceProfile(**payload)
                path = self._path(existing.profile_id, profile.name)
                self._delete_stale_paths(existing.profile_id, keep=path)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return VoiceProfile(**payload)

        now = datetime.now(timezone.utc)
        profile = VoiceProfile(
            profile_id=str(uuid4()),
            name=name,
            created_at=now,
            updated_at=now,
            embedding=normalized.tolist(),
            model_version=EMBEDDING_VERSION,
            threshold=threshold,
            sample_count=1,
        )
        self._path(profile.profile_id, profile.name).write_text(
            json.dumps(profile.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return profile

    def compute_embedding_from_segments(
        self,
        audio_path: Path,
        segments: Iterable[tuple[float, float]],
        *,
        audio_data: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if audio_data is None or sample_rate is None:
            audio_data, sample_rate = self.load_mono_audio(audio_path)

        embeddings: list[np.ndarray] = []
        for start_s, end_s in segments:
            start_idx = int(max(0.0, start_s) * sample_rate)
            end_idx = int(max(start_s, end_s) * sample_rate)
            segment_audio = audio_data[start_idx:end_idx]
            emb = self._compute_embedding_from_array(segment_audio)
            if emb.size:
                embeddings.append(emb)
        if not embeddings:
            return np.zeros(20, dtype=np.float32)
        stacked = np.vstack(embeddings)
        return self._normalize(stacked.mean(axis=0))

    def compute_embedding(
        self,
        audio_path: Path,
        start_s: float | None = None,
        end_s: float | None = None,
        *,
        audio_data: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if audio_data is None or sample_rate is None:
            audio_data, sample_rate = self.load_mono_audio(audio_path)

        if start_s is not None or end_s is not None:
            start_idx = int((start_s or 0.0) * sample_rate)
            end_idx = int((end_s if end_s is not None else len(audio_data) / sample_rate) * sample_rate)
            audio_data = audio_data[max(0, start_idx) : max(0, end_idx)]

        return self._compute_embedding_from_array(audio_data)

    def _compute_embedding_from_array(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(20, dtype=np.float32)

        window = np.hanning(len(audio))
        spectrum = np.abs(np.fft.rfft(audio * window)) + 1e-8
        bands = np.array_split(np.log1p(spectrum), 20)
        features = np.array([band.mean() for band in bands], dtype=np.float32)
        energy = float(np.sqrt(np.mean(np.square(audio))) + 1e-8)
        features[0] = features[0] + energy
        return self._normalize(features)

    def match(
        self,
        embedding: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
        profiles: Sequence[VoiceProfile] | None = None,
    ) -> MatchResponse:
        embedding = self._normalize(embedding)
        scored: list[MatchResult] = []

        candidates = profiles if profiles is not None else self.list_profiles()
        for profile in candidates:
            profile_embedding = np.array(profile.embedding, dtype=np.float32)
            score = float(np.dot(embedding, self._normalize(profile_embedding)))
            if score >= threshold:
                scored.append(MatchResult(profile_id=profile.profile_id, name=profile.name, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        if not scored:
            return MatchResponse(best_match=None, ambiguous_matches=[])

        best = scored[0]
        ambiguous = [item for item in scored[1:] if abs(best.score - item.score) <= 0.03]
        return MatchResponse(best_match=best, ambiguous_matches=ambiguous)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _slugify_name(name: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return normalized[:60] or "speaker"

    @classmethod
    def _path(cls, profile_id: str, name: str | None = None) -> Path:
        if name:
            return SETTINGS.profiles_dir / f"{cls._slugify_name(name)}--{profile_id}.json"
        named = sorted(SETTINGS.profiles_dir.glob(f"*--{profile_id}.json"))
        if named:
            return named[0]
        return SETTINGS.profiles_dir / f"{profile_id}.json"

    @classmethod
    def _delete_stale_paths(cls, profile_id: str, keep: Path) -> None:
        candidates = {SETTINGS.profiles_dir / f"{profile_id}.json", *SETTINGS.profiles_dir.glob(f"*--{profile_id}.json")}
        for candidate in candidates:
            if candidate == keep:
                continue
            candidate.unlink(missing_ok=True)


VOICE_PROFILE_MANAGER = VoiceProfileManager()
