# Target GPU Smoke Validation

`scripts/validate_gpu_target_matrix.py` is the target-host smoke harness for `#193`.
It validates the repo-owned Linux NVIDIA install path on a real GPU machine and is
intentionally separate from the broader parity/performance audit tracked in `#182`.

Until the Linux image work in `#173` lands, this validation runs against the
repo-owned host install path rather than a container image.

## What It Covers
- bootstrap/install validation:
  - `uv sync --extra boosters --extra gpu`
  - `./scripts/install_lightgbm_cuda.sh`
- end-to-end `uv run python main.py train` smoke runs for representative GPU tuples:
  - `logistic_regression` `gpu_native`
  - `logistic_regression` `gpu_patch`
  - `xgboost` `gpu_native`
  - `xgboost` `gpu_patch`
  - `catboost` `gpu_native`
  - `ridge` `gpu_native`
  - `elasticnet` `gpu_native`
  - `random_forest` `gpu_native` through `gpu_cuml`
  - `random_forest` `gpu_native` through `gpu_native_frequency`
  - `lightgbm` `gpu_native`
- per-case validation of:
  - runtime resolution
  - preprocessing backend selection
  - fit/predict success
  - candidate artifact generation
  - MLflow logging
  - sampled GPU utilization and memory

## How To Run
```bash
uv run python scripts/validate_gpu_target_matrix.py
```

Useful flags:
- `--cases <comma-separated-case-ids>` to run a subset of the matrix
- `--cv-splits <n>` to adjust the smoke-time CV depth
- `--skip-sync` when the environment is already synced
- `--skip-lightgbm-install` when the CUDA-enabled LightGBM build is already in place
- `--output-root <path>` to redirect the session output

## Output Contract
Each run writes a timestamped session directory under
`reports/gpu_target_validation/<timestamp>/` containing:
- `validation_summary.json`
- `validation_report.md`
- `bootstrap/` command logs
- one directory per validation case with train stdout/stderr and `gpu_samples.csv`
- a local file-based `mlruns/` store used only for the smoke session

The harness temporarily rewrites repository-root `config.yaml` from the tracked
binary/regression example configs so it can keep the single-config runtime
contract intact, then restores the original `config.yaml` at the end.

## Interpretation
- A failed case means the target machine exposed an environment mismatch or a real
  support-matrix gap. File a follow-up issue before treating the GPU workstream as done.
- This harness is a smoke/integration check only. It does not claim CPU/GPU score
  parity or performance improvement; that work remains in `#182`.
