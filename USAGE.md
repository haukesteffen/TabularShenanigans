# Usage

Operator guide for running TabularShenanigans.

## Prerequisites

- Kaggle CLI access configured for your user.
- An MLflow tracking server available and its tracking URI noted.
- `uv` installed.

## Environment Setup

CPU and local development:
```bash
uv sync
# or, with booster model families (LightGBM, CatBoost, XGBoost):
uv sync --extra boosters
# or, with RealMLP:
uv sync --extra neural
# or, with both booster and neural model families:
uv sync --extra boosters --extra neural
```

Linux GPU hosts (RAPIDS acceleration, Python 3.13, `x86_64`, CUDA 12):
```bash
uv sync --extra boosters --extra gpu
```

LightGBM `gpu_native` additionally requires a CUDA-enabled source build:
```bash
./scripts/install_lightgbm_cuda.sh
PYTHONPATH=src uv run python scripts/validate_lightgbm_cuda_build.py
```

The repository lockfile pins the RAPIDS-compatible `numpy`/`pandas` range, so `uv run python main.py` is safe after syncing the correct extras.

## Config Setup

Copy a tracked example to repository-root `config.yaml`:
```bash
cp config.binary.example.yaml config.yaml
# or
cp config.regression.example.yaml config.yaml
```

Set `experiment.tracking.tracking_uri` in `config.yaml`.

`config.yaml` is the only runtime config source. It is intentionally ignored by Git.

### Config Reference

Tracked example configs:
- `config.binary.example.yaml`
- `config.regression.example.yaml`

Required top-level sections: `competition`, `experiment`.

#### `competition`

| Key | Required | Values / Notes |
| --- | --- | --- |
| `slug` | yes | Kaggle competition slug |
| `task_type` | yes | `regression` or `binary` |
| `primary_metric` | yes | `rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, or `accuracy` |
| `positive_label` | no | needed when binary labels are not `[0,1]`, `[False,True]`, or `["No","Yes"]` |
| `id_column` | no | override auto-inferred ID column |
| `label_column` | no | override auto-inferred label column |
| `cv.n_splits` | yes | |
| `cv.shuffle` | yes | |
| `cv.random_state` | yes | |
| `features.force_categorical` | no | |
| `features.force_numeric` | no | |
| `features.drop_columns` | no | |
| `features.low_cardinality_int_threshold` | no | |

#### `experiment`

| Key | Required | Notes |
| --- | --- | --- |
| `tracking.tracking_uri` | yes | MLflow tracking URI |
| `runtime.compute_target` | no | `auto` (default), `cpu`, or `gpu` |
| `runtime.gpu_backend` | no | `auto` (default), `patch`, or `native`; advanced override |
| `candidates` | yes | one or more candidate entries (see below) |

`train` drains `experiment.candidates` in order unless narrowed with `--candidate-id` or `--index`. `submit --index <n>` uses a 1-based index into this list.

Deprecated: `experiment.candidate` (singular) is still accepted as a one-entry list and emits a deprecation notice.

#### Candidate Shapes

**Model candidate:**
- `candidate_type: model`
- `model_family`:
  - regression: `ridge`, `elasticnet`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `realmlp`, `lightgbm`, `catboost`, `xgboost`
  - binary: `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `realmlp`, `lightgbm`, `catboost`, `xgboost`
- `representation_id`: registered representation (e.g., `median-native`, `standardize-onehot`, `kbins-frequency`, or competition-specific like `s6e3_fr1-median-native`)
- optional `model_params`: manual estimator overrides
  - `logistic_regression` is `saga`-only; `model_params` uses `l1_ratio` only; `penalty` and `solver` are not supported
- optional `optimization`
  - logistic regression Optuna trials fix `solver="saga"` and `max_iter=1000`, tune `C`, `tol`, `class_weight`, and `l1_ratio`

Hard-invalid: representations with `native` categorical preprocessor and any `model_family` other than `catboost` or `realmlp`.

**Blend candidate:**
- `candidate_type: blend`
- `base_candidate_ids`: at least two existing compatible candidate IDs from the same competition experiment
- optional `weights`

`candidate_id` is derived automatically:
- model: `<model_registry_key>-<representation_id>-<hash8>`
- blend: `blend__<hash8>`
- rerunning the same spec derives the same ID and hard-fails only when a canonical completed run for that `candidate_id` already exists in MLflow

## Commands

`uv run python main.py` runs the default pipeline: `fetch -> prepare -> train`.

### Stage Commands

| Command | Description |
| --- | --- |
| `uv run python main.py fetch` | ensure the competition zip is present locally |
| `uv run python main.py prepare` | fetch if needed, materialize context in memory, write EDA reports |
| `uv run python main.py eda` | write EDA reports only |
| `uv run python main.py train` | train all configured candidates sequentially |
| `uv run python main.py train --candidate-id <id>` | train one candidate by ID |
| `uv run python main.py train --index <n>` | train one candidate by 1-based index |
| `uv run python main.py train --skip-existing` | skip candidates that already exist in MLflow |
| `uv run python main.py submit` | dry-run validation for the first configured candidate |
| `uv run python main.py submit --candidate-id <id>` | dry-run validation for a candidate by ID |
| `uv run python main.py submit --index <n>` | dry-run validation for a candidate by 1-based index |
| `uv run python main.py submit --candidate-id <id> --execute` | real Kaggle submission |
| `uv run python main.py submit --candidate-id <id> --execute --message-prefix <prefix>` | real Kaggle submission with a description prefix |
| `uv run python main.py refresh-submissions` | refresh Kaggle submission scores onto MLflow runs |

The default pipeline stops after `train`; `submit` is always a separate explicit command. Without `--execute`, `submit` performs dry-run validation only.

### Utility Scripts

| Command | Description |
| --- | --- |
| `uv run python scripts/benchmark_gpu_checkpoint.py` | CPU vs `gpu_patch` vs `gpu_native` benchmark matrix |
| `uv run python scripts/validate_gpu_target_matrix.py` | target-host GPU smoke validation |
| `PYTHONPATH=src uv run python scripts/validate_lightgbm_cuda_build.py` | validate LightGBM CUDA build |
| `./scripts/install_lightgbm_cuda.sh` | reinstall LightGBM with CUDA support |

### Stage Behavior

- **fetch**: ensures the competition zip is present locally.
- **prepare**: fetches if needed, materializes the competition context in memory, and writes EDA reports under `reports/<competition_slug>/`.
- **eda**: writes EDA reports only.
- **train**: trains all configured candidates sequentially by default, loading one shared dataset context per invocation. Use `--candidate-id` or `--index` to train one configured candidate. Model candidates stage artifacts in a temp bundle and upload to MLflow. Blend candidates download their base candidates from MLflow, materialize blended predictions, then upload the blended candidate run.
- **submit**: downloads the candidate from MLflow and validates `test_predictions.csv` against `sample_submission.csv`. With `--execute`, submits to Kaggle and records the submission event on the candidate run. Without `--execute`, performs dry-run validation only.
- **refresh-submissions**: scans Kaggle submission history, matches `submit=<submission_event_id>` descriptions, and can recover missing MLflow submission events from `candidate=<candidate_id>` metadata before appending score observations.

## Outputs

### Local Filesystem

- `data/<competition_slug>/` — downloaded competition data.
- `reports/<competition_slug>/` — optional EDA reports.
- Temp directories are used during live commands but are not kept afterward.

### MLflow

- One MLflow experiment per `competition.slug`.
- One canonical top-level MLflow run per `candidate_id`.
- Failed or incomplete retry attempts may coexist as non-canonical top-level runs for the same `candidate_id`.
- There are no stage-specific MLflow runs for `prepare`, `submit`, or `refresh-submissions`.
- There is no local canonical `artifacts/` workflow.

Real Kaggle submissions use auto-generated messages:
```text
candidate=<candidate_id> | submit=<submission_event_id> | <metric>=<value>
```

## Runtime Notes

### GPU Execution

- `compute_target: auto` picks the best registered GPU implementation per model/preprocessing tuple, falling back to CPU when nothing is registered.
- `compute_target: gpu` requires a registered GPU implementation and fails fast otherwise.
- `extra_trees` and `hist_gradient_boosting` always fall back to CPU; no GPU backend is registered.
- `realmlp` can use representations with `native` categorical preprocessing or the existing non-native preprocessors.
- Native RealMLP keeps repository numeric preprocessing (`median`, `standardize`, or `kbins`) and passes raw categorical columns plus categorical metadata into the upstream estimator.
- `realmlp` still uses the standard CPU preprocessing path and can train on PyTorch CUDA internally when model routing resolves `compute_target` to GPU.
- Mixed `gpu_patch` and non-`gpu_patch` batches are rejected because RAPIDS hook installation is process-global; split with `train --index <n>`.
- GPU preprocessing can resolve independently from model routing; hybrid CPU-model + GPU-preprocessing cases coerce outputs back to CPU before fit.

### Verification Targets

Preferred competitions for manual testing:
- `playground-series-s5e12`: binary development smoke test
- `playground-series-s6e3`: binary production target
- `playground-series-s5e10`: regression smoke test

Suggested checks:
- Run `uv run python main.py train` and confirm one MLflow run appears per configured candidate.
- Inspect a candidate run and confirm `logs/`, `candidate/`, `config/`, and `context/` artifacts exist.
- Trigger one intentionally failing candidate and confirm the run is marked failed but still has `logs/runtime.log`.
- Rerun the same candidate config after a failed attempt and confirm training retries without manual MLflow cleanup.
- Train a blend candidate and confirm it downloads base candidates from MLflow.
- Run `uv run python main.py submit --index 1` and confirm dry-run validation succeeds.
- Run `uv run python main.py refresh-submissions` after a real submission and confirm MLflow metrics update, including recovery when the post-submit MLflow write was interrupted.

### Current Limits

- Kaggle authentication must be preconfigured.
- RAPIDS acceleration is Linux-only; macOS stays on CPU.
- LightGBM GPU routing requires a CUDA-enabled source build.
- Candidate lookup is keyed by derived `candidate_id`; reusing the same spec while a canonical completed run already exists is a hard error.
