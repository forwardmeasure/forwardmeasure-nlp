from __future__ import annotations

from fastapi import HTTPException
from named_entity_recognition_api.apis.profiles_api_base import BaseProfilesApi
from named_entity_recognition_api.models.ensemble_strategy import EnsembleStrategy
from named_entity_recognition_api.models.ner_profile import NerProfile
from named_entity_recognition_api.models.ner_profile_list import NerProfileList

from named_entity_recognition.runtime import get_runtime


class ProfilesApi(BaseProfilesApi):
    async def get_ner_profile(self, profile_name: str) -> NerProfile:
        profile = get_runtime().service.profiles.get(profile_name)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Unknown NER profile: {profile_name}")
        return _model(profile)

    async def list_ner_profiles(self) -> NerProfileList:
        profiles = get_runtime().service.profiles.values()
        return NerProfileList(profiles=[_model(profile) for profile in profiles])


def _model(profile: dict) -> NerProfile:
    return NerProfile(
        name=str(profile["name"]),
        version=str(profile["version"]),
        description=profile.get("description"),
        labels=list(profile["labels"]),
        engines=list(profile["engines"]),
        strategy=EnsembleStrategy(str(profile["strategy"])),
        threshold=float(profile.get("threshold", 0.5)),
        allow_nested_entities=bool(profile.get("allow_nested_entities", True)),
    )

