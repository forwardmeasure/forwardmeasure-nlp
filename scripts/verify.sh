#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPOSITORY_DIR}"
mvn --batch-mode --no-transfer-progress clean verify
docker build --progress=plain --target test \
  --file named-entity-recognition-service/Dockerfile \
  named-entity-recognition-service

