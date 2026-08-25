from __future__ import annotations

from named_entity_recognition.config import Settings
from named_entity_recognition.domain import (
    ExtractionOptions,
    ExtractionResult,
    Mention,
    TextSegment,
)
from named_entity_recognition.engines.base import NerEngine


class RequestValidationError(ValueError):
    pass


class RequestLimitError(RequestValidationError):
    pass


class NerService:
    def __init__(
        self,
        settings: Settings,
        engine: NerEngine,
        profiles: dict[str, dict],
    ):
        self.settings = settings
        self.engine = engine
        self.profiles = profiles

    def extract(
        self,
        segments: list[TextSegment],
        labels: list[str] | None,
        profile_name: str | None,
        options: ExtractionOptions,
    ) -> tuple[ExtractionResult, dict | None]:
        profile = self._resolve_profile(profile_name)
        effective_labels = _normalize_labels(labels or (profile or {}).get("labels", []))
        if not effective_labels:
            raise RequestValidationError("At least one label or a profile is required")
        self._validate_limits(segments, effective_labels)
        if not self.engine.ready:
            raise RuntimeError("NER engine is not ready")

        mentions: list[Mention] = []
        for segment in segments:
            predictions = self.engine.predict(segment.text, effective_labels, options.threshold)
            for prediction in predictions:
                if prediction.start < 0 or prediction.end <= prediction.start:
                    continue
                if prediction.end > len(segment.text):
                    continue
                mentions.append(
                    Mention(
                        segment_id=segment.segment_id,
                        source_ref=segment.source_ref,
                        text=prediction.text,
                        label=prediction.label.strip().lower(),
                        start=prediction.start,
                        end=prediction.end,
                        absolute_start=segment.base_offset + prediction.start,
                        absolute_end=segment.base_offset + prediction.end,
                        confidence=max(0.0, min(1.0, prediction.confidence)),
                        engine_label=prediction.label,
                    )
                )
        mentions = _reconcile_mentions(
            mentions,
            policy=options.overlap_policy,
            allow_nested=options.allow_nested_entities,
        )
        return ExtractionResult(mentions=mentions), profile

    def _resolve_profile(self, name: str | None) -> dict | None:
        if name is None:
            return None
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise RequestValidationError(f"Unknown NER profile: {name}") from exc

    def _validate_limits(self, segments: list[TextSegment], labels: list[str]) -> None:
        if len(segments) > self.settings.max_segments:
            raise RequestLimitError(f"At most {self.settings.max_segments} segments are allowed")
        if len(labels) > self.settings.max_labels:
            raise RequestLimitError(f"At most {self.settings.max_labels} labels are allowed")
        total = 0
        seen_ids: set[str] = set()
        for segment in segments:
            if segment.segment_id in seen_ids:
                raise RequestValidationError(f"Duplicate segment_id: {segment.segment_id}")
            seen_ids.add(segment.segment_id)
            if not segment.text.strip():
                raise RequestValidationError(f"Segment {segment.segment_id} contains no text")
            if len(segment.text) > self.settings.max_segment_characters:
                raise RequestLimitError(
                    f"Segment {segment.segment_id} exceeds the character limit"
                )
            total += len(segment.text)
        if total > self.settings.max_total_characters:
            raise RequestLimitError(
                f"Request exceeds the {self.settings.max_total_characters} character limit"
            )


def _normalize_labels(labels: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in labels:
        label = value.strip().lower()
        if label and label not in seen:
            result.append(label)
            seen.add(label)
    return result


def _reconcile_mentions(
    mentions: list[Mention], policy: str, allow_nested: bool
) -> list[Mention]:
    if policy == "preserve":
        return sorted(mentions, key=lambda item: (item.segment_id, item.start, item.end, item.label))

    if policy == "longest_span":
        ranked = sorted(
            mentions,
            key=lambda item: (-(item.end - item.start), -item.confidence, item.start),
        )
    else:
        ranked = sorted(
            mentions,
            key=lambda item: (-item.confidence, -(item.end - item.start), item.start),
        )

    selected: list[Mention] = []
    for candidate in ranked:
        conflicts = False
        for existing in selected:
            if candidate.segment_id != existing.segment_id:
                continue
            overlaps = candidate.start < existing.end and existing.start < candidate.end
            nested = (
                candidate.start >= existing.start and candidate.end <= existing.end
            ) or (
                existing.start >= candidate.start and existing.end <= candidate.end
            )
            if overlaps and not (allow_nested and nested and candidate.label != existing.label):
                conflicts = True
                break
        if not conflicts:
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item.segment_id, item.start, item.end, item.label))

