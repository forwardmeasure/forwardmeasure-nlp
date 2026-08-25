from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from named_entity_recognition_api.apis.capabilities_api import router as capabilities_router
from named_entity_recognition_api.apis.extractions_api import router as extractions_router
from named_entity_recognition_api.apis.jobs_api import router as jobs_router
from named_entity_recognition_api.apis.profiles_api import router as profiles_router

# Importing these modules registers implementations with the generated bases.
from named_entity_recognition.api import capabilities, extractions, jobs, profiles  # noqa: F401
from named_entity_recognition.config import Settings
from named_entity_recognition.engines.base import NerEngine
from named_entity_recognition.runtime import configure_runtime, get_runtime


API_PREFIX = "/api/v1/named-entity-recognition"
LOGGER = logging.getLogger(__name__)
_startup_error: Exception | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _startup_error
    runtime = get_runtime()
    try:
        await asyncio.to_thread(runtime.engine.load)
        _startup_error = None
    except Exception as exc:  # readiness remains false and the pod can retry
        _startup_error = exc
        LOGGER.exception("Failed to load the NER engine during application startup")
    yield


def create_app(settings: Settings | None = None, engine: NerEngine | None = None) -> FastAPI:
    configure_runtime(settings=settings, engine=engine)
    application = FastAPI(
        title="ForwardMeasure Named Entity Recognition Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(extractions_router, prefix=API_PREFIX)
    application.include_router(jobs_router, prefix=API_PREFIX)
    application.include_router(profiles_router, prefix=API_PREFIX)
    application.include_router(capabilities_router, prefix=API_PREFIX)

    @application.get("/q/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "UP"}

    @application.get("/q/health/ready", include_in_schema=False)
    async def ready() -> dict[str, str]:
        if not get_runtime().engine.ready:
            detail = str(_startup_error) if _startup_error else "NER engine is not ready"
            raise HTTPException(status_code=503, detail=detail)
        return {"status": "UP"}

    @application.get("/q/health/started", include_in_schema=False)
    async def started() -> dict[str, str]:
        return await ready()

    return application


app = create_app()
