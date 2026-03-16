#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${CUDACXX:-}" ]]; then
  for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-12/bin/nvcc /usr/local/cuda-12.4/bin/nvcc; do
    if [[ -x "$candidate" ]]; then
      export CUDACXX="$candidate"
      break
    fi
  done
fi

if [[ -n "${CUDACXX:-}" ]]; then
  CUDA_HOME="$(cd "$(dirname "$(dirname "$CUDACXX")")" && pwd)"
  export CUDA_HOME
  export CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

uv sync --extra boosters --extra gpu
uv pip install \
  --python .venv/bin/python \
  --reinstall-package lightgbm \
  --no-binary lightgbm \
  -C cmake.define.USE_CUDA=ON \
  "lightgbm>=4.6.0"
PYTHONPATH=src uv run python scripts/validate_lightgbm_cuda_build.py
