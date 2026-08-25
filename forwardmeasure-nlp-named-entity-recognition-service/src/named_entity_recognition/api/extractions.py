from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException
from named_entity_recognition_api.apis.extractions_api_base import BaseExtractionsApi
from named_entity_recognition_api.models.engine_mention_result import EngineMentionResult
from named_entity_recognition_api.models.entity_mention import EntityMention
from named_entity_recognition_api.models.ner_extraction_request import NerExtractionRequest
from named_entity_recognition_api.models.ner_extraction_response import NerExtractionResponse
from named_entity_recognition_api.models.ner_warning import NerWarning

from named_entity_recognition.api.common import engine_descriptor
from named_entity_recognition.domain import ExtractionOptions, TextSegment
from named_entity_recognition.runtime import get_runtime
from named_entity_recognition.service import RequestLimitError, RequestValidationError


class ExtractionsApi(BaseExtractionsApi):
    async def extract_entities(
        self, ner_extraction_request: NerExtractionRequest
    ) -> NerExtractionResponse:
        runtime = get_runtime()
        request = ner_extraction_request
        options_model = request.options
        options = ExtractionOptions(
            threshold=float(options_model.threshold if options_model else 0.5),
            overlap_policy=(
                options_model.overlap_policy.value
                if options_model and options_model.overlap_policy
                else "highest_confidence"
            ),
            include_engine_results=(
                bool(options_model.include_engine_results)
                if options_model and options_model.include_engine_results is not None
                else True
            ),
            allow_nested_entities=(
                bool(options_model.allow_nested_entities)
                if options_model and options_model.allow_nested_entities is not None
                else True
            ),
        )
        segments = [
            TextSegment(
                segment_id=segment.segment_id,
                text=segment.text,
                language=segment.language,
                base_offset=int(segment.base_offset or 0),
                source_ref=segment.source_ref,
            )
            for segment in request.segments
        ]

        started = time.perf_counter()
        try:
            result, profile = await asyncio.to_thread(
                runtime.service.extract,
                segments,
                list(request.labels) if request.labels else None,
                request.profile,
                options,
            )
        except RequestLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except RequestValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        descriptor = engine_descriptor(runtime)
        mentions = []
        for mention in result.mentions:
            engine_results = None
            if options.include_engine_results:
                engine_results = [
                    EngineMentionResult(
                        engine=runtime.engine.name,
                        model=runtime.engine.model_name,
                        model_revision=runtime.engine.model_revision,
                        label=mention.engine_label,
                        confidence=mention.confidence,
                    )
                ]
            mentions.append(
                EntityMention(
                    segment_id=mention.segment_id,
                    source_ref=mention.source_ref,
                    label=mention.label,
                    text=mention.text,
                    start=mention.start,
                    end=mention.end,
                    absolute_start=mention.absolute_start,
                    absolute_end=mention.absolute_end,
                    confidence=mention.confidence,
                    engine_results=engine_results,
                )
            )

        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return NerExtractionResponse(
            extraction_id=uuid4(),
            request_id=request.request_id,
            profile=profile.get("name") if profile else None,
            profile_version=profile.get("version") if profile else None,
            mentions=mentions,
            engines=[descriptor],
            warnings=[
                NerWarning(code=code, message=message, segment_id=segment_id)
                for code, message, segment_id in result.warnings
            ],
            created_at=datetime.now(timezone.utc),
            processing_time_ms=elapsed_ms,
        )

