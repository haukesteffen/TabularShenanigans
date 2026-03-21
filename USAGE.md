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

Optional top-level section: `screening`.

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

`screening` evaluates the explicit list of `screening.candidates`, then prints a copy/paste-ready YAML snippet for the top `screening.promote_top_k` candidates to paste into `experiment.candidates`.

#### Candidate Shapes

**Model candidate:**
- `candidate_type: model`
- `model_family`:
  - regression: `ridge`, `elasticnet`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `realmlp`, `lightgbm`, `catboost`, `xgboost`
  - binary: `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `realmlp`, `lightgbm`, `catboost`, `xgboost`
- `representation`: explicit operator/pruner spec
  - `operators`: one or more operator entries like `standardize_numeric`, `native_numeric`, `native_categorical`, `ordinal_encode_categoricals`, `frequency_encode_categoricals`, `onehot_encode_low_cardinality_categoricals`, `row_missing_count`
  - `pruners`: optional pruner entries like `high_correlation_prune`
- optional `model_params`: manual estimator overrides
  - `logistic_regression` is `saga`-only; `model_params` uses `l1_ratio` only; `penalty` and `solver` are not supported
- optional `optimization`
  - logistic regression Optuna trials fix `solver="saga"` and `max_iter=1000`, tune `C`, `tol`, `class_weight`, and `l1_ratio`

Hard-invalid: representations with `native_categorical` and any `model_family` other than `catboost` or `realmlp`.

**Blend candidate:**
- `candidate_type: blend`
- `base_candidate_ids`: at least two existing compatible candidate IDs from the same competition experiment
- optional `weights`

`candidate_id` is derived automatically:
- model: `<model_registry_key>-<representation_id>-<hash8>`
- blend: `blend__<hash8>`
- the same logical model candidate keeps the same `candidate_id` across screening and canonical evaluation
- rerunning the same canonical candidate hard-fails only when a canonical completed run for that `candidate_id` already exists in MLflow
- schema v4 changes the candidate-id fingerprint inputs versus older experiments; existing schema-v3 MLflow data is archival and does not share IDs with new runs

#### `screening`

| Key | Required | Notes |
| --- | --- | --- |
| `candidates` | yes | explicit list of `(model_family, representation)` pairs to screen |
| `optimization` | no | optional shared tuning budget applied to all candidates; per-candidate `optimization` overrides this |
| `cv.n_splits` | no | defaults to `2` |
| `cv.shuffle` | no | defaults to `true` |
| `cv.random_state` | no | defaults to `42` |
| `promote_top_k` | no | defaults to `3`; must be `<=` the number of configured candidates |

## Commands

`uv run python main.py` runs the default pipeline: `fetch -> prepare -> train`.

### Stage Commands

| Command | Description |
| --- | --- |
| `uv run python main.py fetch` | ensure the competition zip is present locally |
| `uv run python main.py prepare` | fetch if needed, materialize context in memory, write EDA reports |
| `uv run python main.py eda` | write EDA reports only |
| `uv run python main.py train` | train all configured candidates sequentially |
| `uv run python main.py screening` | run all configured screening candidates sequentially |
| `uv run python main.py screening --candidate-id <id>` | screen one candidate by configured screening candidate ID |
| `uv run python main.py screening --index <n>` | screen one candidate by 1-based screening index |
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
- **screening**: runs model-only screening candidates under `screening.cv`, writes screening runs to the screening MLflow experiment, and prints a promotion snippet for the top-ranked candidates.
- **train**: trains canonical candidates sequentially by default, loading one shared dataset context per invocation. Use `--candidate-id` or `--index` to train one configured candidate. Model candidates stage artifacts in a temp bundle and upload to the candidates MLflow experiment. Blend candidates download their base candidates from the candidates experiment, materialize blended predictions, then upload the canonical candidate run.
- **submit**: downloads the canonical candidate from MLflow and validates `test_predictions.csv` against `sample_submission.csv`. With `--execute`, submits to Kaggle and records the submission event on a submission run in the submissions MLflow experiment. Without `--execute`, performs dry-run validation only.
- **refresh-submissions**: scans Kaggle submission history, matches `submit=<submission_event_id>` descriptions, and updates submission runs in the submissions experiment.

## Outputs

### Local Filesystem

- `data/<competition_slug>/` — downloaded competition data.
- `reports/<competition_slug>/` — optional EDA reports.
- Temp directories are used during live commands but are not kept afterward.

### MLflow

- Three MLflow experiments per `competition.slug`:
  - `<competition_slug>__screening`
  - `<competition_slug>__candidates`
  - `<competition_slug>__submissions`
- Screening runs live only in the screening experiment.
- Re-screening the same logical candidate is allowed and creates another screening run; screening runs are exploratory, not canonical.
- One canonical top-level candidate run per `candidate_id` lives in the candidates experiment.
- Submission event runs live in the submissions experiment.
- Failed or incomplete retry attempts may coexist as non-canonical top-level runs for the same canonical `candidate_id`.
- There are no stage-specific MLflow runs for `prepare` or `refresh-submissions`.
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
- Run `uv run python main.py screening` and confirm one MLflow run appears per configured screening candidate in `<competition_slug>__screening`.
- Run `uv run python main.py train` and confirm one MLflow run appears per configured canonical candidate in `<competition_slug>__candidates`.
- Inspect a candidate run and confirm `logs/`, `candidate/`, `config/`, and `context/` artifacts exist.
- Trigger one intentionally failing candidate and confirm the run is marked failed but still has `logs/runtime.log`.
- Rerun the same candidate config after a failed attempt and confirm training retries without manual MLflow cleanup.
- Train a blend candidate and confirm it downloads base candidates from MLflow.
- Run `uv run python main.py submit --index 1` and confirm dry-run validation succeeds.
- Run `uv run python main.py submit --index 1 --execute` and confirm a submission run appears in `<competition_slug>__submissions` while the canonical candidate run remains unchanged.
- Run `uv run python main.py refresh-submissions` after a real submission and confirm submission-run metrics update, including recovery when the original submission-run write was interrupted.

### Current Limits

- Kaggle authentication must be preconfigured.
- RAPIDS acceleration is Linux-only; macOS stays on CPU.
- LightGBM GPU routing requires a CUDA-enabled source build.
- Candidate lookup is keyed by derived `candidate_id`; reusing the same spec while a canonical completed run already exists is a hard error.
