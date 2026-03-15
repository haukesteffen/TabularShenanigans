#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

uv sync --extra boosters --extra gpu
uv pip install \
  --python .venv/bin/python \
  --reinstall-package lightgbm \
  --no-binary lightgbm \
  -C cmake.define.USE_CUDA=ON \
  "lightgbm>=4.6.0"
PYTHONPATH=src uv run python scripts/validate_lightgbm_cuda_build.py
