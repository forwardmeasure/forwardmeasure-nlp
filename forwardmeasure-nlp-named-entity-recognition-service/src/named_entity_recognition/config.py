from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROFILE = {
    "name": "general-entities-v1",
    "version": "1.0.0",
    "description": "General-purpose people, organizations, locations, dates, and products.",
    "labels": ["person", "organization", "location", "date", "product"],
    "engines": ["gliner"],
    "strategy": "single_engine",
    "threshold": 0.5,
    "allow_nested_entities": True,
}


@dataclass(frozen=True)
class Settings:
    service_version: str = "1.0.0"
    model_path: str = "/models/urchade--gliner_medium-v2.1"
    model_name: str = "urchade/gliner_medium-v2.1"
    model_revision: str | None = "40ec419335d09393f298636f471328b722c6da9e"
    backbone_config_path: Path | None = None
    backbone_tokenizer_path: Path | None = None
    device: str = "cpu"
    local_files_only: bool = True
    max_segments: int = 100
    max_segment_characters: int = 100_000
    max_total_characters: int = 250_000
    max_labels: int = 100
    profiles_path: Path | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        profiles_value = os.getenv("NER_PROFILES_PATH")
        model_path = os.getenv("NER_MODEL_PATH") or _kserve_model_path() or cls.model_path
        return cls(
            service_version=os.getenv("NER_SERVICE_VERSION", cls.service_version),
            model_path=model_path,
            model_name=os.getenv("NER_MODEL_NAME", cls.model_name),
            model_revision=os.getenv("NER_MODEL_REVISION", cls.model_revision) or None,
            backbone_config_path=_optional_path("NER_BACKBONE_CONFIG_PATH"),
            backbone_tokenizer_path=_optional_path("NER_BACKBONE_TOKENIZER_PATH"),
            device=os.getenv("NER_DEVICE", cls.device),
            local_files_only=_boolean("NER_LOCAL_FILES_ONLY", cls.local_files_only),
            max_segments=_integer("NER_MAX_SEGMENTS", cls.max_segments),
            max_segment_characters=_integer(
                "NER_MAX_SEGMENT_CHARACTERS", cls.max_segment_characters
            ),
            max_total_characters=_integer(
                "NER_MAX_TOTAL_CHARACTERS", cls.max_total_characters
            ),
            max_labels=_integer("NER_MAX_LABELS", cls.max_labels),
            profiles_path=Path(profiles_value) if profiles_value else None,
        )

    def load_profiles(self) -> dict[str, dict[str, Any]]:
        if self.profiles_path is None:
            profile = dict(DEFAULT_PROFILE)
            return {str(profile["name"]): profile}
        payload = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", payload)
        if not isinstance(profiles, list):
            raise ValueError("NER profile configuration must contain a profiles array")
        result: dict[str, dict[str, Any]] = {}
        for profile in profiles:
            if not isinstance(profile, dict) or not profile.get("name"):
                raise ValueError("Each NER profile must be an object with a name")
            result[str(profile["name"])] = profile
        return result


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _kserve_model_path() -> str | None:
    """Resolve the model directory injected by a KServe custom predictor chart."""
    base_path = os.getenv("MODEL_BASE_PATH")
    model_subdir = os.getenv("MODEL_SUBDIR")
    if not base_path or not model_subdir:
        return None
    return str(Path(base_path) / model_subdir)
