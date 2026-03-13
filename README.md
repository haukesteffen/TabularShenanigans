# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Primary target: Kaggle Playground Series tabular competitions.
- Task scope: regression and binary classification.
- Canonical experiment store: MLflow.
- Local persistent outputs: downloaded competition data under `data/<competition_slug>/` and optional EDA reports under `reports/<competition_slug>/`.

## Current Capabilities
- Load and validate one repository-root `config.yaml`.
- Fetch Kaggle competition data into `data/<competition_slug>/` when the zip is missing.
- Infer the Playground-style submission schema from `train.csv`, `test.csv`, and `sample_submission.csv`, with optional config overrides for `id_column` and `label_column`.
- Build deterministic competition-level fold assignments in memory from the configured CV settings.
- Run deterministic feature recipes for model candidates.
- Train one model candidate at a time with split numeric and categorical preprocessing.
- Train one blend candidate at a time by downloading compatible base candidates from MLflow and combining their saved predictions.
- Run Optuna inside `train` for model candidates when optimization is enabled.
- Validate prediction artifacts against `sample_submission.csv` before submission.
- Submit a candidate to Kaggle from MLflow-backed candidate artifacts.
- Refresh Kaggle submission outcomes back onto the same MLflow candidate runs.

## MLflow Canonical Model
- One MLflow experiment per `competition.slug`.
- One top-level MLflow run per `candidate_id`.
- Candidate runs are the source of truth for:
  - candidate manifest
  - OOF/test predictions
  - blend metadata
  - optimization metadata
  - submission event history
  - submission score observations
- There are no stage-specific MLflow runs for `prepare`, `submit`, or `refresh-submissions`.
- There is no local canonical `artifacts/` workflow anymore.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- Optuna for hyperparameter tuning
- MLflow for canonical run metadata and artifact storage
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart
1. Ensure Kaggle CLI access is already configured for your user.
2. Ensure an MLflow tracking server is available and note its tracking URI.
3. Install dependencies with `uv sync`.
4. If you want LightGBM, CatBoost, or XGBoost model recipes, install the optional booster dependencies with `uv sync --extra boosters`.
5. If you want RAPIDS-backed GPU execution on a Linux `x86_64` CUDA 12 host, install the GPU extra with `uv sync --extra boosters --extra gpu`.
6. Copy a tracked example config to repository-root `config.yaml`.
7. Set `experiment.tracking.tracking_uri` in `config.yaml`.
8. Run `uv run python main.py`.

```bash
cp config.binary.example.yaml config.yaml
# or
cp config.regression.example.yaml config.yaml
```

`config.yaml` is the only runtime config source. It is intentionally ignored by Git.

`main.py` is a thin wrapper around a bootstrap module. The bootstrap runs before the
runtime imports application modules that depend on `pandas` or `sklearn`, so GPU
setups can install RAPIDS acceleration hooks before the training stack loads.

Supported environment matrix:
- CPU and local development: `uv sync` or `uv sync --extra boosters`
- Linux GPU hosts: `uv sync --extra boosters --extra gpu`
- RAPIDS GPU support currently targets Python 3.13 on Linux `x86_64` CUDA 12 hosts with NVIDIA-visible devices
- The repository lockfile is now the source of truth for the RAPIDS-compatible `numpy` / `pandas` range, so `uv run python main.py ...` is safe after syncing the correct extras

## Stage Commands
`uv run python main.py` runs the default pipeline: `fetch -> prepare -> train -> submit`.

Available stage-specific commands:
- `uv run python main.py fetch`
- `uv run python main.py prepare`
- `uv run python main.py eda`
- `uv run python main.py train`
- `uv run python main.py submit`
- `uv run python main.py submit --candidate-id <candidate_id>`
- `uv run python main.py refresh-submissions`

Stage behavior:
- `fetch`: ensures the competition zip is present locally.
- `prepare`: fetches if needed, materializes the competition context in memory, and writes EDA reports under `reports/<competition_slug>/`.
- `eda`: writes EDA reports only.
- `train`: trains one candidate and logs it to MLflow. Model candidates stage candidate artifacts in a temp bundle and upload them. Blend candidates download their base candidates from MLflow, materialize blended predictions, then upload the blended candidate run.
- `submit`: downloads one candidate from MLflow, validates `test_predictions.csv` against `sample_submission.csv`, and when `experiment.submit.enabled=true` submits to Kaggle and records the submission event under that same candidate run.
- `refresh-submissions`: scans Kaggle submission history, matches `submit=<submission_event_id>` descriptions, and appends new score observations back onto the matching candidate runs.

`submit` defaults to the derived `candidate_id` for the current config. Use `--candidate-id` when you want to submit another existing candidate for the same competition experiment.

## Config Overview
Tracked example configs:
- `config.binary.example.yaml`
- `config.regression.example.yaml`

Required top-level sections:
- `competition`
- `experiment`

`competition` keys:
- `slug`
- `task_type`: `regression` or `binary`
- `primary_metric`: `rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, or `accuracy`
- optional `positive_label` for binary tasks when the training labels do not match one of the safe auto-resolved pairs `[0, 1]`, `[False, True]`, or `["No", "Yes"]`
- optional `id_column`
- optional `label_column`
- `cv`:
  - `n_splits`
  - `shuffle`
  - `random_state`
- optional `features`:
  - `force_categorical`
  - `force_numeric`
  - `drop_columns`
  - `low_cardinality_int_threshold`

`experiment` keys:
- required `tracking`
- optional `runtime`
- required `candidate`
- optional `submit`

`experiment.tracking` keys:
- `tracking_uri`: MLflow tracking URI. This is required.

`experiment.runtime` keys:
- `compute_target`: `auto`, `cpu`, or `gpu`
  - `auto`: prefer GPU only when the runtime exposes visible NVIDIA devices and RAPIDS hooks are available, otherwise fall back to CPU
  - `cpu`: force CPU execution
  - `gpu`: require GPU execution and fail fast when no GPU runtime or RAPIDS hook path is available
  - when GPU execution is active, `xgboost`, `lightgbm`, and `catboost` also switch to their GPU-specific estimator params automatically
  - the RAPIDS-backed GPU path currently expects the project environment to be installed with `uv sync --extra boosters --extra gpu` on a Python 3.13 Linux `x86_64` CUDA 12 host

`experiment.candidate` keys:
- shared:
  - `candidate_type`: `model` or `blend`
- model candidate:
  - `feature_recipe_id`: built-in values are `fr0`, `fr1`, `fr2`, `fr3`, and the `fr2_ablate_*` / `fr3_ablate_*` study variants for `playground-series-s6e3`
    - `fr3` is the reduced stable `s6e3` engineered recipe
    - `fr2_ablate_*` and `fr3_ablate_*` variants are study recipes used for grouped ablation work
  - `model_family`
    - regression: `ridge`, `elasticnet`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
    - binary: `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
  - `numeric_preprocessor`: `median`, `standardize`, or `kbins`
  - `categorical_preprocessor`: `onehot`, `ordinal`, `frequency`, or `native`
  - optional `model_params`: manual estimator overrides; when omitted, the runtime uses repo defaults plus the estimator library defaults
    - `logistic_regression` is `saga`-only in this runtime
    - logistic `model_params` use `l1_ratio` only; `penalty` and `solver` are not supported
  - optional `optimization`
    - logistic regression Optuna trials fix `solver="saga"` and `max_iter=1000`
    - logistic regression Optuna trials tune `C`, `tol`, `class_weight`, and `l1_ratio`
  - `categorical_preprocessor: onehot` is an internal matrix-output decision, not a config knob:
    - sparse CSR output: `ridge`, `elasticnet`, `logistic_regression`, `random_forest`, `extra_trees`, `lightgbm`, `catboost`, `xgboost`
    - dense array output: `hist_gradient_boosting`
    - `numeric_preprocessor: kbins` follows the same sparse-versus-dense decision when combined with `onehot`
- blend candidate:
  - `base_candidate_ids`: at least two existing compatible candidate IDs from the same competition experiment
  - optional `weights`

`candidate_id` is derived automatically and is not configured directly:
- model candidates derive `<feature_recipe_id>--<preprocessing_scheme_id>--<model_registry_key>--<hash8>`
- blend candidates derive `blend__<hash8>`
- rerunning the exact same candidate spec derives the same `candidate_id` and hard-fails if that run already exists in MLflow

Hard-invalid preprocessing combination:
- `categorical_preprocessor: native` with any `model_family` other than `catboost`

`experiment.submit` keys:
- `enabled`
- optional `message_prefix`

Real Kaggle submissions use auto-generated messages shaped like:

```text
candidate=<candidate_id> | submit=<submission_event_id> | <metric>=<value>
```

## Candidate Artifact Contract
Each candidate MLflow run stores:
- `logs/runtime.log`
- `config/runtime_config.json`
- `context/competition.json`
- `context/folds.csv`
- `candidate/candidate.json`
- `candidate/fold_metrics.csv`
- `candidate/oof_predictions.csv`
- `candidate/test_predictions.csv`
- `candidate/test_prediction_probabilities.csv` for binary `accuracy` candidates and blends
- `candidate/blend_summary.csv` for blend candidates
- `candidate/optimization_summary.json`, `candidate/optimization_trials.csv`, and `candidate/optimization_best_params.json` for optimized model candidates

Real Kaggle submissions also add:
- `submissions/history.json`
- `submissions/<submission_event_id>/event.json`
- `submissions/<submission_event_id>/submission.csv`
- `submissions/<submission_event_id>/observations.json`

Current candidate-run metrics include:
- `cv_score_mean`
- `cv_score_std`
- `feature_count`
- `fit_wall_seconds`
- `train_rows`
- `test_rows`
- `optimization_best_value` and `optimization_trial_count` when present
- `submit_count`, `latest_public_score`, `best_public_score`, `latest_private_score`, and `best_private_score` when submission history exists

Current candidate-run tags include:
- `run_kind=candidate`
- `tracking_schema_version=2`
- `competition_slug`
- `candidate_id`
- `candidate_type`
- `task_type`
- `primary_metric`
- `config_fingerprint`
- git metadata when available

The MLflow schema is intentionally lean in this iteration. After a few real runs, trim low-value params/metrics/artifacts and add anything still missing based on actual usage.

## Prediction Contracts
- Binary `roc_auc` and `log_loss`: `test_predictions.csv` stores positive-class probabilities in `[0, 1]`.
- Binary `accuracy`: `test_predictions.csv` stores class labels from the observed binary label pair.
- Binary `accuracy` candidates and blends also store `test_prediction_probabilities.csv` so blends can average positive-class probabilities before applying a `0.5` threshold.
- Regression submissions must be numeric, non-missing, and finite.

## Manual Verification Targets
Preferred targets:
- `playground-series-s5e12`: binary development smoke test
- `playground-series-s6e3`: binary production target
- `playground-series-s5e10`: regression smoke test

Suggested checks:
- run `uv run python main.py train` and confirm one MLflow candidate run appears under the competition experiment
- inspect the candidate run and confirm `logs/`, `candidate/`, `config/`, and `context/` artifacts exist
- trigger one intentionally failing candidate run and confirm the MLflow run is marked failed but still has `logs/runtime.log`
- rerun the exact same candidate config and confirm `uv run python main.py train` fails because it derives the same `candidate_id`
- train one blend candidate and confirm it downloads base candidates from MLflow instead of reading local artifact directories
- run `uv run python main.py submit` with `experiment.submit.enabled: false` and confirm dry-run validation succeeds without creating local candidate artifacts or submission ledgers
- run `uv run python main.py refresh-submissions` after at least one real Kaggle submission and confirm candidate-run submission metrics update in MLflow

## Current Limits
- Kaggle authentication is expected to be preconfigured.
- Competition downloads still live under `data/<competition_slug>/`.
- RAPIDS acceleration is a Linux GPU runtime concern. Native macOS runs stay on CPU.
- LightGBM GPU routing still depends on a CUDA-enabled LightGBM runtime build in the target image or host environment.
- EDA reports still live under `reports/<competition_slug>/`.
- Local temp directories are used during a running command, but candidate state is not kept there after the command finishes.
- Candidate lookup is keyed by derived `candidate_id` inside the competition MLflow experiment. Reusing the same candidate spec without deleting the existing run is a hard error.
