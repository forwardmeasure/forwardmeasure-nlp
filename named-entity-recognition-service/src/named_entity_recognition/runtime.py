from __future__ import annotations

from dataclasses import dataclass

from named_entity_recognition.config import Settings
from named_entity_recognition.engines.base import NerEngine
from named_entity_recognition.engines.gliner import GlinerEngine
from named_entity_recognition.service import NerService


@dataclass
class Runtime:
    settings: Settings
    engine: NerEngine
    service: NerService


_runtime: Runtime | None = None


def configure_runtime(settings: Settings | None = None, engine: NerEngine | None = None) -> Runtime:
    global _runtime
    resolved_settings = settings or Settings.from_environment()
    resolved_engine = engine or GlinerEngine(resolved_settings)
    profiles = resolved_settings.load_profiles()
    _runtime = Runtime(
        settings=resolved_settings,
        engine=resolved_engine,
        service=NerService(resolved_settings, resolved_engine, profiles),
    )
    return _runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        return configure_runtime()
    return _runtime

