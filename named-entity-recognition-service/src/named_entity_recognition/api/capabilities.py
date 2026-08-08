from __future__ import annotations

from named_entity_recognition_api.apis.capabilities_api_base import BaseCapabilitiesApi
from named_entity_recognition_api.models.ensemble_strategy import EnsembleStrategy
from named_entity_recognition_api.models.ner_capabilities import NerCapabilities
from named_entity_recognition_api.models.ner_limits import NerLimits

from named_entity_recognition.api.common import engine_descriptor
from named_entity_recognition.runtime import get_runtime


class CapabilitiesApi(BaseCapabilitiesApi):
    async def get_ner_capabilities(self) -> NerCapabilities:
        runtime = get_runtime()
        settings = runtime.settings
        return NerCapabilities(
            service_version=settings.service_version,
            synchronous_extraction=True,
            asynchronous_jobs=False,
            limits=NerLimits(
                max_segments=settings.max_segments,
                max_segment_characters=settings.max_segment_characters,
                max_total_characters=settings.max_total_characters,
                max_labels=settings.max_labels,
            ),
            engines=[engine_descriptor(runtime)],
            ensemble_strategies=[EnsembleStrategy.SINGLE_ENGINE],
        )

