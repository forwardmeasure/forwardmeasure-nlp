from __future__ import annotations

import json

import pytest

from named_entity_recognition.config import Settings
from named_entity_recognition.domain import EngineMention, ExtractionOptions, TextSegment
from named_entity_recognition.engines.gliner import _offline_model_view
from named_entity_recognition.service import NerService, RequestLimitError, RequestValidationError


class FakeEngine:
    name = "fake"
    version = "1.0"
    model_name = "fake/model"
    model_revision = "test"
    device = "cpu"
    ready = True

    def load(self) -> None:
        pass

    def predict(self, text: str, labels: list[str], threshold: float) -> list[EngineMention]:
        return [
            EngineMention(text="Alice", label="Person", start=0, end=5, confidence=0.91),
            EngineMention(text="Acme", label="Organization", start=15, end=19, confidence=0.87),
        ]


def service(**overrides) -> NerService:
    settings = Settings(**overrides)
    profiles = settings.load_profiles()
    return NerService(settings, FakeEngine(), profiles)


def test_extract_preserves_segment_and_absolute_offsets() -> None:
    result, profile = service().extract(
        segments=[TextSegment("page-1", "Alice works at Acme", base_offset=100)],
        labels=["Person", "Organization"],
        profile_name=None,
        options=ExtractionOptions(),
    )

    assert profile is None
    assert [(mention.label, mention.absolute_start, mention.absolute_end) for mention in result.mentions] == [
        ("person", 100, 105),
        ("organization", 115, 119),
    ]


def test_profile_supplies_labels() -> None:
    result, profile = service().extract(
        segments=[TextSegment("segment", "Alice works at Acme")],
        labels=None,
        profile_name="general-entities-v1",
        options=ExtractionOptions(),
    )

    assert len(result.mentions) == 2
    assert profile and profile["version"] == "1.0.0"


def test_duplicate_segment_ids_are_rejected() -> None:
    with pytest.raises(RequestValidationError, match="Duplicate segment_id"):
        service().extract(
            segments=[TextSegment("same", "Alice"), TextSegment("same", "Acme")],
            labels=["person"],
            profile_name=None,
            options=ExtractionOptions(),
        )


def test_total_character_limit_is_enforced() -> None:
    with pytest.raises(RequestLimitError, match="character limit"):
        service(max_total_characters=5).extract(
            segments=[TextSegment("segment", "Alice works at Acme")],
            labels=["person"],
            profile_name=None,
            options=ExtractionOptions(),
        )


def test_kserve_model_cache_environment_resolves_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NER_MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_BASE_PATH", "/mnt/models/cache")
    monkeypatch.setenv("MODEL_SUBDIR", "example--specialized-ner")

    settings = Settings.from_environment()

    assert settings.model_path == "/mnt/models/cache/example--specialized-ner"


def test_explicit_model_path_overrides_kserve_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NER_MODEL_PATH", "/models/pinned")
    monkeypatch.setenv("MODEL_BASE_PATH", "/mnt/models/cache")
    monkeypatch.setenv("MODEL_SUBDIR", "example--specialized-ner")

    settings = Settings.from_environment()

    assert settings.model_path == "/models/pinned"


def test_offline_model_view_injects_backbone_config_without_changing_cache(tmp_path) -> None:
    model_path = tmp_path / "model-cache"
    model_path.mkdir()
    original_config = {
        "model_name": "microsoft/deberta-v3-base",
        "hidden_size": 512,
    }
    (model_path / "gliner_config.json").write_text(
        json.dumps(original_config), encoding="utf-8"
    )
    (model_path / "model.safetensors").write_text("weights", encoding="utf-8")
    backbone_path = tmp_path / "backbone.json"
    backbone_path.write_text(
        json.dumps(
            {
                "_name_or_path": "microsoft/deberta-v3-base",
                "model_type": "deberta-v2",
                "hidden_size": 768,
            }
        ),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "spm.model").write_text("tokenizer", encoding="utf-8")
    (tokenizer_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    view_path = _offline_model_view(model_path, backbone_path, tokenizer_path)

    patched = json.loads(
        (view_path / "gliner_config.json").read_text(encoding="utf-8")
    )
    assert patched["encoder_config"]["model_type"] == "deberta-v2"
    assert (view_path / "model.safetensors").is_symlink()
    assert (view_path / "spm.model").is_symlink()
    assert (view_path / "tokenizer_config.json").is_symlink()
    assert json.loads(
        (model_path / "gliner_config.json").read_text(encoding="utf-8")
    ) == original_config
