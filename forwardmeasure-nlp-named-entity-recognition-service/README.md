# Named Entity Recognition Service

This module is an engine-neutral KServe custom predictor. GLiNER is its first
in-process provider, but callers depend only on the repository-owned OpenAPI
contract and generated clients.

The service identifies text mentions and preserves caller-provided segment and
source offsets. It deliberately does not resolve identities, create canonical
entities, infer relationships, or attach Entity Intelligence domain meaning.

The synchronous extraction, profile, capability, and Kubernetes health APIs
are implemented. The asynchronous job resources remain explicitly fail-closed
with HTTP 501 until they are backed by a durable workflow rather than an
in-memory queue.

## Runtime configuration

- `NER_MODEL_PATH`: local model directory mounted read-only into the pod.
- `NER_MODEL_NAME`: stable model identifier returned as provenance.
- `NER_MODEL_REVISION`: pinned model revision.
- `NER_BACKBONE_CONFIG_PATH`: optional local Transformer configuration.
- `NER_BACKBONE_TOKENIZER_PATH`: optional local tokenizer directory.
- `NER_DEVICE`: PyTorch device, normally `cpu` or `cuda`.
- `NER_LOCAL_FILES_ONLY`: defaults to `true` in the production image.
- `NER_PROFILES_PATH`: optional JSON profile configuration.

`NER_MODEL_PATH` takes precedence over the `MODEL_BASE_PATH` and
`MODEL_SUBDIR` pair supplied by the shared model-cache chart.

## Build and test

From the repository root:

```bash
mvn clean verify
docker build --target test \
  -f named-entity-recognition-service/Dockerfile \
  named-entity-recognition-service
```

Generate and build the production image through Maven:

```bash
mvn package -Pcontainer-image
```

