# ForwardMeasure NLP

Open-source, reusable natural-language processing contracts, generated clients,
and model-serving implementations shared across ForwardMeasure verticals.

The first capability is implementation-neutral named entity recognition:

- `named-entity-recognition-contracts` publishes the canonical OpenAPI contract.
- `named-entity-recognition-client-java` generates a Java 25 client.
- `named-entity-recognition-client-typescript` generates a TypeScript fetch client.
- `named-entity-recognition-service` generates the Python client and FastAPI edge,
  and supplies the GLiNER-backed KServe predictor.
- `forwardmeasure-nlp-bom` provides the supported Maven dependency surface.

All dependency and build-plugin versions are selected in the root POM. Leaf
modules contain no independent library or plugin versions.

## Verification

```bash
mvn clean verify
docker build --target test \
  -f named-entity-recognition-service/Dockerfile \
  named-entity-recognition-service
```

The equivalent complete local gate is `./scripts/verify.sh`. Version tags and
manual release dispatches execute the same gate before publishing the immutable
Docker Hub image. The release workflow requires `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN` repository secrets.

The shared Kubernetes deployment is owned by the `forwardmeasure-platform`
repository. Vertical applications consume its stable cluster-local predictor
service and do not deploy their own copy.
