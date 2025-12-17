#!/usr/bin/env bash
set -euo pipefail

# Create .pypirc from environment variables
if [ -n "${PYPI_API_TOKEN:-}" ]; then
  cat > ~/.pypirc <<EOF
[distutils]
index-servers =
    pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = ${PYPI_API_TOKEN}
EOF
  chmod 600 ~/.pypirc
  echo "✓ .pypirc created for PyPI"
fi

if [ -n "${TEST_PYPI_API_TOKEN:-}" ]; then
  cat >> ~/.pypirc <<EOF

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = ${TEST_PYPI_API_TOKEN}
EOF
  echo "✓ TestPyPI added to .pypirc"
fi

# Determine target repository
REPO="${1:-pypi}"
echo "Publishing to: $REPO"

# Install twine if not available
if ! command -v twine &> /dev/null; then
  echo "Installing twine..."
  pip install twine
fi

# Upload to specified repository
twine upload --repository "$REPO" dist/* --non-interactive --skip-existing
