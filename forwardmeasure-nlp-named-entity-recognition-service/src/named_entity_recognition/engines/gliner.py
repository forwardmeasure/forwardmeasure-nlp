from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from typing import Any

from named_entity_recognition.config import Settings
from named_entity_recognition.domain import EngineMention


class GlinerEngine:
    name = "gliner"

    def __init__(self, settings: Settings):
        self.model_name = settings.model_name
        self.model_path = settings.model_path
        self.model_revision = settings.model_revision
        self.backbone_config_path = settings.backbone_config_path
        self.backbone_tokenizer_path = settings.backbone_tokenizer_path
        self.device = settings.device
        self.local_files_only = settings.local_files_only
        self._model: Any | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()
        try:
            self.version = version("gliner")
        except PackageNotFoundError:
            self.version = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from gliner import GLiNER

            kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
            if self.model_revision and not Path(self.model_path).is_dir():
                kwargs["revision"] = self.model_revision
            load_path = _offline_model_view(
                Path(self.model_path),
                self.backbone_config_path,
                self.backbone_tokenizer_path,
            )
            model = GLiNER.from_pretrained(str(load_path), **kwargs)
            if self.device:
                model = model.to(self.device)
            model.eval()
            self._model = model

    def predict(self, text: str, labels: list[str], threshold: float) -> list[EngineMention]:
        if self._model is None:
            raise RuntimeError("GLiNER model is not loaded")
        with self._inference_lock:
            entities = self._model.predict_entities(text, labels, threshold=threshold)
        return [
            EngineMention(
                text=str(entity["text"]),
                label=str(entity["label"]),
                start=int(entity["start"]),
                end=int(entity["end"]),
                confidence=float(entity["score"]),
            )
            for entity in entities
        ]


def _offline_model_view(
    model_path: Path,
    backbone_config_path: Path | None,
    backbone_tokenizer_path: Path | None = None,
) -> Path:
    """Inject offline backbone assets without changing the read-only model cache."""
    if backbone_config_path is None and backbone_tokenizer_path is None:
        return model_path

    gliner_config_path = model_path / "gliner_config.json"
    config = json.loads(gliner_config_path.read_text(encoding="utf-8"))
    if backbone_config_path is not None and config.get("encoder_config") is None:
        encoder_config = json.loads(backbone_config_path.read_text(encoding="utf-8"))
        configured_model = str(config.get("model_name", ""))
        configured_backbone = str(encoder_config.get("_name_or_path", ""))
        if configured_model and configured_backbone and configured_model != configured_backbone:
            raise ValueError(
                f"Backbone config {configured_backbone} does not match GLiNER model {configured_model}"
            )
        config["encoder_config"] = encoder_config

    view_path = Path(mkdtemp(prefix="ner-model-"))
    for source in model_path.iterdir():
        if source.name == "gliner_config.json":
            continue
        (view_path / source.name).symlink_to(
            source.resolve(), target_is_directory=source.is_dir()
        )
    if backbone_tokenizer_path is not None:
        for source in backbone_tokenizer_path.iterdir():
            destination = view_path / source.name
            if destination.exists():
                raise ValueError(f"Backbone tokenizer asset conflicts with model asset: {source.name}")
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    (view_path / "gliner_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return view_path
