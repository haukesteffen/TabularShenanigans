# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the runtime contracts as they exist now.

## System Flow
1. Load and validate the repository-root `config.yaml`.
2. Normalize and validate `competition.task_type`, `competition.primary_metric`, and the candidate contract.
3. Resolve the MLflow tracking URI from `experiment.tracking.tracking_uri`.
4. Download the competition zip into `data/<competition_slug>/` when it is missing.
5. Load one shared dataset context from `train.csv`, `test.csv`, and `sample_submission.csv`.
6. Resolve `id_column` and `label_column`, then prepare raw feature frames with the resolved ID column excluded from modeled features.
7. For model candidates, apply the selected deterministic feature recipe.
8. Build the competition fold assignments in memory from the configured CV settings.
9. For model candidates, fit the selected preprocessing + model combination fold-locally and produce OOF/test predictions.
10. For blend candidates, download compatible base candidates from the competition MLflow experiment, validate compatibility, and combine their saved predictions without retraining the base candidates.
11. Stage the candidate bundle into a temp directory:
    - `config/runtime_config.json`
    - `context/competition.json`
    - `context/folds.csv`
    - `candidate/*`
12. Create one MLflow run for the candidate and upload the staged bundle.
13. For real Kaggle submissions, download the candidate from MLflow, validate `test_predictions.csv` against `sample_submission.csv`, submit `submission.csv`, and append submission history artifacts back onto that same candidate run.
14. For submission refresh, scan Kaggle submissions once, match `submit=<submission_event_id>` descriptions, and update candidate-run submission history plus scoreboard metrics in place.

## Canonical Storage Model
- MLflow is the canonical experiment store.
- One MLflow experiment per `competition.slug`.
- One top-level MLflow run per `candidate_id`.
- There are no stage-specific MLflow runs.
- There are no local canonical candidate directories or local submission ledgers.

Local persistent filesystem usage is limited to:
- `data/<competition_slug>/` for downloaded competition zips
- `reports/<competition_slug>/` for optional EDA CSVs
- temp directories during live commands

## MLflow Run Schema
Each candidate run is named with `candidate_id`.

### Tags
Current candidate-run tags:
- `run_kind=candidate`
- `tracking_schema_version=2`
- `competition_slug`
- `candidate_id`
- `candidate_type`
- `task_type`
- `primary_metric`
- `config_fingerprint`
- `git_commit` when available
- `git_branch` when available

### Params
Current candidate-run params:
- `cv__n_splits`
- `cv__shuffle`
- `cv__random_state`
- model candidates:
  - `feature_recipe_id`
  - `numeric_preprocessor`
  - `categorical_preprocessor`
  - `preprocessing_scheme_id`
  - `model_family`
  - `model_registry_key`
  - `model__*` for resolved model params
  - `opt__*` for optimization settings
- blend candidates:
  - `blend__base_candidate_ids_json`
  - `blend__configured_weights_json`

### Metrics
Current candidate-run metrics:
- `cv_score_mean`
- `cv_score_std`
- `train_rows`
- `test_rows`
- `feature_count`
- `fit_wall_seconds`
- `optimization_best_value` when present
- `optimization_trial_count` when present
- submission metrics when submission history exists:
  - `submit_count`
  - `latest_public_score`
  - `best_public_score`
  - `latest_private_score`
  - `best_private_score`

### Artifacts
Every candidate run stores:
- `logs/runtime.log`
- `config/runtime_config.json`
- `context/competition.json`
- `context/folds.csv`
- `candidate/candidate.json`
- `candidate/fold_metrics.csv`
- `candidate/oof_predictions.csv`
- `candidate/test_predictions.csv`

Optional candidate artifacts:
- `candidate/test_prediction_probabilities.csv`
- `candidate/blend_summary.csv`
- `candidate/optimization_summary.json`
- `candidate/optimization_trials.csv`
- `candidate/optimization_best_params.json`

Submission artifacts on the same candidate run:
- `submissions/history.json`
- `submissions/<submission_event_id>/event.json`
- `submissions/<submission_event_id>/submission.csv`
- `submissions/<submission_event_id>/observations.json`

## CLI Stages
- `uv run python main.py`: `fetch -> prepare -> train -> submit`
- `uv run python main.py fetch`
- `uv run python main.py prepare`
- `uv run python main.py eda`
- `uv run python main.py train`
- `uv run python main.py submit`
- `uv run python main.py submit --candidate-id <candidate_id>`
- `uv run python main.py refresh-submissions`

Stage notes:
- `prepare` no longer persists canonical competition metadata. It only prepares the context in memory and writes EDA reports.
- `train` is the only stage that creates candidate runs.
- `submit` and `refresh-submissions` mutate existing candidate runs by appending submission history and score metrics.

## Module Responsibilities
- [main.py](/Users/hs/dev/TabularShenanigans/main.py): CLI entrypoint and linear stage dispatch.
- [competition.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/competition.py): in-memory competition preparation, fold assignment materialization, and prepared-context construction.
- [config.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/config.py): nested config validation, metric normalization, candidate-id derivation, and resolved model lookup.
- [candidate_artifacts.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/candidate_artifacts.py): shared manifest/config helpers and temp-bundle file writers for candidate/context artifacts.
- [data.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/data.py): Kaggle downloads, zip access, schema inference, and sample-submission loading.
- [eda.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/eda.py): local EDA report generation.
- [feature_recipes](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/feature_recipes): deterministic feature recipes such as `fr0`, `fr1`, `fr2`, and the `fr2_ablate_*` grouped ablation variants used for `s6e3` recipe studies.
- [model_evaluation.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/model_evaluation.py): shared prepared training context and reusable CV evaluation logic for train/tune.
- [models.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/models.py): model registry, capability checks, estimator construction, and tuning space definitions.
- [preprocess.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/preprocess.py): raw feature-frame preparation and split preprocessing components.
- [cv.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/cv.py): splitters and task-aware metric scoring.
- [train.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/train.py): model training workflow, candidate manifest construction, temp bundle staging, and MLflow candidate logging.
- [blend.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/blend.py): MLflow-backed base-candidate loading, compatibility checks, weighted blending, and blended candidate logging.
- [tune.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/tune.py): Optuna orchestration on top of the shared model-evaluation layer.
- [submit.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/submit.py): MLflow-backed candidate resolution, submission validation, Kaggle submit orchestration, and submission refresh.
- [submission_history.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/submission_history.py): candidate-run submission event/observation models and Kaggle refresh helpers.
- [mlflow_store.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/mlflow_store.py): MLflow experiment/run lookup, candidate-run creation, candidate download, artifact upload, and submission-history persistence.

## Configuration Contract
Input:
- one local `config.yaml`
- tracked starting points:
  - `config.binary.example.yaml`
  - `config.regression.example.yaml`

Required top-level keys:
- `competition`
- `experiment`

`competition` keys:
- `slug`
- `task_type`
- `primary_metric`
- optional `positive_label`
- optional `id_column`
- optional `label_column`
- `cv`
- optional `features`

`experiment` keys:
- required `tracking`
- required `candidate`
- optional `submit`

`experiment.tracking`:
- `tracking_uri` only

Model candidate contract:
- `candidate_type: model`
- `feature_recipe_id`
- `model_family`
- `numeric_preprocessor`
- `categorical_preprocessor`
- optional `model_params` (manual estimator overrides; when omitted, the runtime uses repo defaults plus estimator library defaults)
  - `logistic_regression` is `saga`-only
  - logistic `model_params` use `l1_ratio` only; `penalty` and `solver` are invalid
- optional `optimization`

Optimization note:
- logistic regression Optuna trials fix `solver="saga"` and `max_iter=1000`
- logistic regression Optuna trials tune `C`, `tol`, `class_weight`, and `l1_ratio`

Blend candidate contract:
- `candidate_type: blend`
- `base_candidate_ids`
- optional `weights`

Naming contract:
- model candidates derive `<feature_recipe_id>--<preprocessing_scheme_id>--<model_registry_key>--<hash8>`
- blend candidates derive `blend__<hash8>`
- identical candidate specs derive the same `candidate_id`

Hard-invalid preprocessing combination:
- `categorical_preprocessor: native` with any model family other than `catboost`

Sparse onehot runtime contract:
- `categorical_preprocessor: onehot` stays an internal runtime choice rather than a user-facing dense/sparse switch
- sparse CSR output is used for `ridge`, `elasticnet`, `logistic_regression`, `random_forest`, `extra_trees`, `lightgbm`, `catboost`, and `xgboost`
- dense array output remains in place for `hist_gradient_boosting`
- `numeric_preprocessor: kbins` follows the same sparse-versus-dense output contract when composed with `onehot`

## Candidate Manifest Contract
Model candidate manifests currently record:
- identity: `candidate_id`, `candidate_type`, `competition_slug`, `task_type`, `primary_metric`
- provenance: `config_fingerprint`, `config_snapshot`, `mlflow_run_id`
- model info: `model_family`, `model_registry_key`, `estimator_name`
- feature/preprocessing info: `feature_recipe_id`, `feature_columns`, `numeric_preprocessor`, `categorical_preprocessor`, `preprocessing_scheme_id`
- CV summary: `cv_summary`
- schema/label metadata: `id_column`, `label_column`, `positive_label`, `negative_label`, `observed_label_pair`
- dataset metadata: `target_summary`, `train_rows`, `train_cols`, `test_rows`, `test_cols`
- optional tuning provenance

Blend candidate manifests currently record:
- the same identity/provenance/schema fields
- `model_registry_key=blend_weighted_average`
- `estimator_name=WeightedAverageBlend`
- `preprocessing_scheme_id=blend`
- `component_candidates` with candidate IDs, MLflow run IDs, normalized weights, and component CV summaries

## Submission Contract
Submission preparation uses the selected candidate manifest as the source of truth for:
- `competition_slug`
- `task_type`
- `primary_metric`
- `id_column`
- `label_column`
- binary label metadata

Validation rules:
- `test_predictions.csv` columns must exactly match `[id_column, label_column]`
- row count must match `sample_submission.csv`
- ID values and order must exactly match `sample_submission.csv`
- regression values must be numeric, non-missing, and finite
- binary `roc_auc`/`log_loss` values must be numeric probabilities in `[0, 1]`
- binary `accuracy` values must stay within the observed label pair

Real Kaggle submissions:
- generate one `submission_event_id`
- use description format `candidate=<candidate_id> | submit=<submission_event_id> | <metric>=<value>`
- append the event into `submissions/history.json` on the candidate run
- upload `submission.csv` under `submissions/<submission_event_id>/`
- attempt an immediate refresh without creating a separate run

Refresh behavior:
- scan Kaggle submissions once
- extract `submission_event_id` from the Kaggle description
- match it to candidate-run submission history
- append only new observations
- update candidate-run score metrics in place

## Runtime Invariants
- MLflow is required. The runtime does not support a no-tracking mode.
- Candidate state is canonical in MLflow, not on local disk.
- Reusing an existing derived `candidate_id` within a competition experiment is a hard error.
- `prepare` is not a persisted source of truth anymore.
- `train` and `blend` must produce exactly one candidate run keyed by `candidate_id`.
- Candidate runs should upload `logs/runtime.log` on both success and failure once the run exists.
- `submit` resolves candidates from MLflow, not from local artifact directories.
- `refresh-submissions` updates existing candidate runs and does not create standalone tracking runs.
- Feature recipes must be deterministic, leakage-safe, and schema-preserving across train/test.
- Binary probability blends require matching saved class metadata across all base candidates.
- Binary `accuracy` blends require the saved probability sidecar and current probability-average blend rule metadata across all base candidates.
- Dry-run submit validates predictions but does not persist submission history.
- Kaggle downloads and submissions assume local CLI auth is already configured.

## Verification Notes
Recommended manual checks:
- one real candidate on the current competition target
- one synthetic or smaller smoke workflow covering:
  - two model candidates
  - one blend candidate
  - one intentionally failing candidate run with `logs/runtime.log` plus traceback uploaded before run termination
  - one dry-run submit
  - one submission-refresh path against seeded submission history

After a few real runs, revisit:
- which params are actually worth showing in the runs table
- which metrics are redundant
- whether some artifacts should be dropped or renamed
- whether candidate-level submission history should expose more derived leaderboard metadata
