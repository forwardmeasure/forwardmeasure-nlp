from __future__ import annotations

from typing import Protocol

from named_entity_recognition.domain import EngineMention


class NerEngine(Protocol):
    name: str
    version: str | None
    model_name: str
    model_revision: str | None
    device: str

    @property
    def ready(self) -> bool: ...

    def load(self) -> None: ...

    def predict(self, text: str, labels: list[str], threshold: float) -> list[EngineMention]: ...

