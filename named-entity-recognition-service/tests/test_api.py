from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from named_entity_recognition.config import Settings
from named_entity_recognition.domain import EngineMention
from named_entity_recognition.main import create_app


class FakeEngine:
    name = "gliner"
    version = "0.2.28"
    model_name = "test/gliner"
    model_revision = "revision"
    device = "cpu"

    def __init__(self) -> None:
        self.ready = False

    def load(self) -> None:
        self.ready = True

    def predict(self, text: str, labels: list[str], threshold: float) -> list[EngineMention]:
        return [EngineMention("Alice", "person", 0, 5, 0.94)]


@pytest.mark.asyncio
async def test_generated_api_dispatches_to_service_implementation() -> None:
    app = create_app(settings=Settings(), engine=FakeEngine())
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://ner.test"
        ) as client:
            response = await client.post(
                "/api/v1/named-entity-recognition/extractions",
                json={
                    "labels": ["person"],
                    "segments": [
                        {
                            "segment_id": "page-1",
                            "text": "Alice arrived",
                            "base_offset": 12,
                        }
                    ],
                },
            )

            assert response.status_code == 200
            payload = response.json()
            assert payload["mentions"][0]["absolute_start"] == 12
            assert payload["mentions"][0]["absolute_end"] == 17
            assert payload["mentions"][0]["engine_results"][0]["engine"] == "gliner"
            assert (await client.get("/q/health/ready")).status_code == 200


@pytest.mark.asyncio
async def test_async_jobs_are_explicitly_not_enabled() -> None:
    app = create_app(settings=Settings(), engine=FakeEngine())
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://ner.test"
        ) as client:
            response = await client.post(
                "/api/v1/named-entity-recognition/jobs",
                json={
                    "extraction": {
                        "labels": ["person"],
                        "segments": [{"segment_id": "one", "text": "Alice"}],
                    }
                },
            )

            assert response.status_code == 501
