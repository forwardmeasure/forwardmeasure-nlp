ARG PYTHON_RUNTIME_VERSION=3.12.12
ARG PYTHON_RUNTIME_IMAGE_DIGEST=sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
FROM docker.io/library/python:${PYTHON_RUNTIME_VERSION}-slim-bookworm@${PYTHON_RUNTIME_IMAGE_DIGEST} AS base

ARG PYTORCH_VERSION=2.13.0
ARG DEBERTA_V3_BASE_REVISION=8ccc9b6f36199bec6961081d44eb72fb3f7353f3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NER_LOCAL_FILES_ONLY=true \
    NER_MODEL_PATH=/mnt/models/cache/urchade--gliner_medium-v2.1 \
    NER_MODEL_NAME=urchade/gliner_medium-v2.1 \
    NER_MODEL_REVISION=40ec419335d09393f298636f471328b722c6da9e \
    NER_BACKBONE_CONFIG_PATH=/app/model-configs/microsoft--deberta-v3-base.json \
    NER_BACKBONE_TOKENIZER_PATH=/app/model-backbones/microsoft--deberta-v3-base \
    NER_DEVICE=cpu \
    HF_HOME=/tmp/huggingface

RUN groupadd --gid 10013 ner \
 && useradd --uid 10013 --gid 10013 --create-home ner

WORKDIR /app
COPY target/generated-server /tmp/generated-server
COPY target/python-runtime/requirements.txt /tmp/requirements.txt
COPY pyproject.toml README.md /app/
COPY model-configs /app/model-configs
COPY src /app/src

RUN python -m pip install --upgrade pip \
 && python -m pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==${PYTORCH_VERSION}" \
 && python -m pip install -r /tmp/requirements.txt \
 && python -m pip install --no-deps /tmp/generated-server \
 && python -m pip install --no-deps /app \
 && python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', revision='${DEBERTA_V3_BASE_REVISION}').save_pretrained('/app/model-backbones/microsoft--deberta-v3-base')" \
 && rm -rf /tmp/generated-server /tmp/requirements.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

FROM base AS test
COPY target/python-runtime/requirements-test.txt /tmp/requirements-test.txt
COPY tests /app/tests
RUN python -m pip install -r /tmp/requirements-test.txt \
 && rm -f /tmp/requirements-test.txt \
 && python -m pytest /app/tests

FROM base AS runtime
USER 10013:10013
EXPOSE 8080
ENTRYPOINT ["python", "-m", "named_entity_recognition"]

