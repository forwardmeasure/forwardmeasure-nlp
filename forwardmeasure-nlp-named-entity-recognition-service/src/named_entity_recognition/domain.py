from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextSegment:
    segment_id: str
    text: str
    language: str | None = None
    base_offset: int = 0
    source_ref: str | None = None


@dataclass(frozen=True)
class ExtractionOptions:
    threshold: float = 0.5
    overlap_policy: str = "highest_confidence"
    include_engine_results: bool = True
    allow_nested_entities: bool = True


@dataclass(frozen=True)
class EngineMention:
    text: str
    label: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class Mention:
    segment_id: str
    source_ref: str | None
    text: str
    label: str
    start: int
    end: int
    absolute_start: int
    absolute_end: int
    confidence: float
    engine_label: str


@dataclass(frozen=True)
class ExtractionResult:
    mentions: list[Mention] = field(default_factory=list)
    warnings: list[tuple[str, str, str | None]] = field(default_factory=list)

