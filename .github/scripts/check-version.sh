#!/usr/bin/env bash
set -euo pipefail

echo "Ref: $GITHUB_REF"
TAG="${GITHUB_REF#refs/tags/}"
VERSION=$(python - <<'PY'
import tomllib
with open('pyproject.toml','rb') as f:
    data = tomllib.load(f)
print(data['project']['version'])
PY
)
echo "Tag:    ${TAG}"
echo "Version: ${VERSION}"
if [ "${TAG#v}" != "$VERSION" ]; then
  echo "Error: Tag (${TAG}) does not match project version (${VERSION})."
  exit 1
fi
