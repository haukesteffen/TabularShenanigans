# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the runtime contracts as they exist now.

For setup, commands, and config reference, see [USAGE.md](/USAGE.md).

## System Flow

1. Enter the bootstrap entrypoint before importing runtime modules that depend on `pandas` or `sklearn`.
2. Read `experiment.runtime.compute_target` and the optional advanced `experiment.runtime.gpu_backend` override from repository-root `config.yaml`, resolve hardware capability separately from tuple routing, inspect the selected train or screening candidates before importing the runtime stack, install RAPIDS hooks only when the selected batch resolves entirely to `gpu_patch`, and route GPU-capable booster families onto their GPU parameter paths. Screening config uses explicit `screening.candidates`; each candidate is validated for compatibility and fails hard if incompatible.
3. Load and validate the repository-root `config.yaml`.
4. Normalize and validate `competition.task_type`, `competition.primary_metric`, and the full `experiment.candidates` contract.
5. Resolve the MLflow tracking URI from `experiment.tracking.tracking_uri`.
6. Download the competition zip into `data/<competition_slug>/` when it is missing.
7. Load one shared dataset context from `train.csv`, `test.csv`, and `sample_submission.csv`.
8. Resolve `id_column` and `label_column`, then prepare raw feature frames with the resolved ID column excluded from modeled features.
9. For each selected configured candidate, resolve its candidate-specific runtime execution context.
10. For model candidates, compile the selected representation (resolves feature schema, validates model compatibility, selects preprocessing backend).
11. Build the competition fold assignments in memory from the configured CV settings.
12. For model candidates, fit the compiled representation per fold (all steps fit inside the CV loop) and produce OOF/test predictions.
13. For blend candidates, download compatible base candidates from the competition MLflow experiment, validate compatibility, and combine their saved predictions without retraining the base candidates.
14. Stage the candidate bundle into a temp directory:
    - `config/runtime_config.json`
    - `context/competition.json`
    - `context/folds.csv`
    - `candidate/*`
15. For screening, create one screening MLflow run attempt in `<competition_slug>__screening` and upload the staged bundle.
16. For canonical training, create one canonical candidate MLflow run attempt in `<competition_slug>__candidates` and upload the staged bundle.
17. Treat only a `FINISHED` canonical candidate run with the full candidate artifact contract as the canonical run for that `candidate_id`.
18. For real Kaggle submissions, download the explicitly selected canonical candidate from MLflow, validate `test_predictions.csv` against `sample_submission.csv`, submit `submission.csv`, and create a submission run in `<competition_slug>__submissions`.
19. For submission refresh, scan Kaggle submissions once, match `submit=<submission_event_id>` descriptions, recover missing submission runs from `candidate=<candidate_id>` metadata when needed, and update submission-run history plus scoreboard metrics in place.

## Runtime Invariants

- MLflow is required. The runtime does not support a no-tracking mode.
- Candidate state is canonical in MLflow, not on local disk.
- A completed run with the full candidate artifact contract is canonical for a `candidate_id` only in the candidates experiment.
- Failed or incomplete candidate attempts are non-canonical and may remain in MLflow until cleaned up.
- Starting a new canonical attempt while another canonical run for the same `candidate_id` is still active is a hard error.
- `prepare` is not a persisted source of truth anymore.
- `screening` writes non-canonical screening runs into the screening experiment.
- Re-screening the same logical candidate is expected behavior and creates another screening run; screening runs are exploratory history, not canonical state.
- `train` and `blend` must produce exactly one canonical candidate run keyed by `candidate_id` in the candidates experiment.
- Candidate runs should upload `logs/runtime.log` on both success and failure once the run exists.
- `submit` resolves candidates from MLflow, not from local artifact directories.
- `submit` and `refresh-submissions` must not mutate canonical candidate runs.
- `refresh-submissions` updates existing submission runs and may recover missing submission runs.
- Untuned model candidates use upstream library estimator defaults for all hyperparameters. The repo sets only runtime and task-contract params: determinism (`random_state`/`random_seed`), parallelism (`n_jobs`/`thread_count`), logging/file-writing controls (`verbosity`, `verbose`, `allow_writing_files`), problem-definition params (`objective`, `eval_metric`, `loss_function`), and GPU routing. Users who want a stronger untuned baseline should set `model_params` explicitly or enable optimization.
- Representation steps must be deterministic and schema-preserving across train/test. All steps fit per fold inside the CV loop.
- Binary probability blends require matching saved class metadata across all base candidates.
- Binary `accuracy` blends require the saved probability sidecar and current probability-average blend rule metadata across all base candidates.
- Dry-run submit validates predictions but does not persist submission history.
- Kaggle downloads and submissions assume local CLI auth is already configured.

## Canonical Storage Model

- MLflow is the canonical experiment store.
- Three MLflow experiments per `competition.slug`: screening, candidates, submissions.
- One canonical top-level MLflow candidate run per `candidate_id` in the candidates experiment.
- Screening runs live in the screening experiment and are not canonical.
- Submission runs live in the submissions experiment and track leaderboard events independently of candidate runs.
- Tracking schema v4 changes candidate-id derivation versus archival schema-v3 data; existing experiments should be treated as read-only history rather than mixed with new runs.
- Failed or incomplete retry attempts may coexist as non-canonical top-level runs for the same canonical `candidate_id`.
- There are no local canonical candidate directories or local submission ledgers.

Local persistent filesystem usage is limited to:
- `data/<competition_slug>/` for downloaded competition zips
- `reports/<competition_slug>/` for optional EDA CSVs
- temp directories during live commands

## MLflow Run Schema

Each candidate run is named with `candidate_id`.

### Tags

### Screening Runs

Screening runs use:
- `run_kind=screening`
- `tracking_schema_version=5`

Screening runs use the same candidate artifact bundle contract as canonical model candidates, but they live in the screening experiment and are not used for canonical lookup or submissions.

### Candidate Runs

Canonical candidate runs use:
- `run_kind=canonical`
- `tracking_schema_version=5`
- `competition_slug`
- `candidate_id`
- `candidate_type`
- `task_type`
- `primary_metric`
- model candidates also tag:
  - `model_family`
  - `model_registry_key`
  - `representation_id`
  - `representation_operator_ids`
  - `representation_pruner_ids`
  - `hyperparameter_source`
- `runtime_requested_compute_target`
- `runtime_resolved_compute_target`
- `runtime_requested_gpu_backend`
- `runtime_resolved_gpu_backend`
- `runtime_acceleration_backend`
- `runtime_preprocessing_backend`
- `config_fingerprint`
- `git_commit` when available
- `git_branch` when available

### Params

- `cv__n_splits`
- `cv__shuffle`
- `cv__random_state`
- `runtime__requested_compute_target`
- `runtime__resolved_compute_target`
- `runtime__requested_gpu_backend`
- `runtime__resolved_gpu_backend`
- `runtime__gpu_available`
- `runtime__acceleration_backend`
- `runtime__preprocessing_backend`
- `runtime__rapids_hooks_installed`
- `runtime__fallback_reason` when CPU fallback happened under `compute_target=auto`
- model candidates:
  - `representation_id`
  - `model_family`
  - `model_registry_key`
  - `representation__summary`
  - `representation__operator_ids`
  - `representation__operator_ids_json`
  - `representation__operators_json`
  - `representation__pruner_ids`
  - `representation__pruner_ids_json`
  - `representation__pruners_json`
  - `representation__matrix_output_kind`
  - `representation__routing_numeric_preprocessor`
  - `representation__routing_categorical_preprocessor`
  - `representation__has_native_categorical`
  - `representation__has_sparse_numeric`
  - `representation__has_dense_numeric`
  - `representation__has_native_numeric`
  - `hp__source`
  - `hp__override_count`
  - `hp__overrides_json`
  - `hp__*` for resolved configured or tuned hyperparameter overrides
  - `model__resolved_param_count`
  - `model__resolved_params_json`
  - `opt__*` for optimization settings (logged only when optimization block is present)
- blend candidates:
  - `blend__base_candidate_ids_json`
  - `blend__configured_weights_json`

### Metrics

- `cv_score_mean`
- `cv_score_std`
- `train_rows`
- `test_rows`
- `feature_count`
- `fit_wall_seconds`
- `optimization_best_value` when present
- `optimization_trial_count` when present
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

### Optimization Trial Child Runs

Optuna-backed canonical model candidates also create nested MLflow child runs named `trial_<n>` with `run_kind=optimization_trial`.

Trial child-run tags:
- `run_kind=optimization_trial`
- `mlflow.parentRunId=<candidate_run_id>`
- `candidate_id`
- `model_family`
- `model_registry_key`
- `representation_id`
- `trial_state`, transitioning from `RUNNING` to `COMPLETE` or `FAIL`

### Submission Runs

Submission runs use:
- `run_kind=submission`
- `tracking_schema_version=5`
- `competition_slug`
- `candidate_id`
- `submission_event_id`
- `git_commit` when available
- `git_branch` when available

Submission run params include:
- `candidate_id`
- `submission_file_name`

Submission run metrics:
- `submit_count`
- `latest_public_score`
- `best_public_score`
- `latest_private_score`
- `best_private_score`

Submission run artifacts:
- `submissions/history.json`
- `submissions/event.json`
- `submissions/submission.csv`
- `submissions/observations.json`

Trial child-run params:
- `trial_number`
- `representation_id`
- `model_family`
- `model_registry_key`
- `runtime__resolved_compute_target`
- `runtime__resolved_gpu_backend`
- `runtime__preprocessing_backend`
- `representation__summary`
- `representation__operator_ids`
- `representation__operator_ids_json`
- `representation__operators_json`
- `representation__pruner_ids`
- `representation__pruner_ids_json`
- `representation__pruners_json`
- `representation__matrix_output_kind`
- `representation__routing_numeric_preprocessor`
- `representation__routing_categorical_preprocessor`
- `representation__has_native_categorical`
- `representation__has_sparse_numeric`
- `representation__has_dense_numeric`
- `representation__has_native_numeric`
- `hp__source`
- `hp__override_count`
- `hp__overrides_json`
- `hp__*` for sampled Optuna hyperparameters
- `model__resolved_param_count` on completed trials
- `model__resolved_params_json` on completed trials

Trial child-run metrics:
- `duration_seconds`
- `cv_score_mean` on completed trials
- `cv_score_std` on completed trials

## Prediction Contracts

- Binary `roc_auc` and `log_loss`: `test_predictions.csv` stores positive-class probabilities in `[0, 1]`.
- Binary `accuracy`: `test_predictions.csv` stores class labels from the observed binary label pair.
- Binary `accuracy` candidates and blends also store `test_prediction_probabilities.csv` so blends can average positive-class probabilities before applying a `0.5` threshold.
- Regression submissions must be numeric, non-missing, and finite.

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
- create one submission run in the submissions experiment
- write `submissions/history.json`, `submissions/event.json`, and `submissions/submission.csv` on that submission run
- attempt an immediate refresh against that submission run

Refresh behavior:
- scan Kaggle submissions once
- extract `submission_event_id` from the Kaggle description
- match it to submission-run history
- when `submission_event_id` is missing from MLflow history but the Kaggle description includes `candidate=<candidate_id>`, recover the submission event into a new submission run linked to that canonical candidate
- append only new observations
- update submission-run score metrics in place

## Candidate Manifest Contract

Model candidate manifests currently record:
- identity: `candidate_id`, `candidate_type`, `competition_slug`, `task_type`, `primary_metric`
- provenance: `config_fingerprint`, `config_snapshot`, `mlflow_run_id`
- runtime execution: requested/resolved compute target, acceleration backend, RAPIDS hook status, and hardware capability snapshot
- runtime profiling: training-context build time, fold-stage CV timings, artifact staging time, and first-fold matrix residency snapshots when collected
- model info: `model_family`, `model_registry_key`, `estimator_name`
- representation info: `representation_id`, `feature_columns`
- runtime-selected preprocessing backend: `preprocessing_backend`
- CV summary: `cv_summary`
- schema/label metadata: `id_column`, `label_column`, `positive_label`, `negative_label`, `observed_label_pair`
- dataset metadata: `target_summary`, `train_rows`, `train_cols`, `test_rows`, `test_cols`
- optional tuning provenance

Blend candidate manifests currently record:
- the same identity/provenance/schema fields
- `model_registry_key=blend_weighted_average`
- `estimator_name=WeightedAverageBlend`
- `component_candidates` with candidate IDs, MLflow run IDs, normalized weights, and component CV summaries

## GPU Runtime Contracts

### Execution Routing

`compute_target` and `gpu_backend` are resolved during bootstrap before `pandas`/`sklearn` imports. Tuple routing is registry-driven.

Current `compute_target: auto` routing on supported Linux NVIDIA hosts:

| Model family | Auto-selected model backend | Notes |
| --- | --- | --- |
| `logistic_regression` (binary) | `gpu_native` for `frequency + standardize`; `gpu_patch` for `frequency + median\|kbins`; CPU fallback otherwise | GPU logistic is still `frequency`-only |
| `lightgbm` | `gpu_native` | `onehot` keeps the existing sparse CSR boundary; the model trains through the explicit CUDA adapter |
| `xgboost` | `gpu_native` for `frequency + median\|standardize`; `gpu_patch` for the remaining registered `ordinal`/`frequency` tuples; CPU fallback otherwise | sparse GPU-native inputs remain unsupported |
| `catboost` | `gpu_native` for `categorical_preprocessor: native`; CPU fallback otherwise | preprocessing stays on `cpu_native_frame` |
| `realmlp` | `cpu` backend with `compute_target=gpu` semantics for all registered tuples, including `categorical_preprocessor: native` | preprocessing stays on CPU; the estimator still receives `device="cuda"` when runtime routing resolves to GPU |
| `ridge`, `elasticnet` | `gpu_native` for `frequency + median\|standardize`; CPU fallback otherwise | explicit cuML regressors stay on dense inputs only |
| `random_forest` | `gpu_native` for `onehot + median\|standardize\|kbins` and `frequency + median\|standardize`; CPU fallback otherwise | explicit cuML random forest stays on dense inputs only |
| `extra_trees`, `hist_gradient_boosting` | CPU fallback | intentional fallback because no maintained official GPU backend is registered |

Current preprocessing selection on GPU hosts:

| Condition | Resolved preprocessing backend |
| --- | --- |
| `categorical_preprocessor: native` | `cpu_native_frame` |
| dense `onehot` with `median`, `standardize`, or `kbins` | `gpu_cuml` |
| `frequency` with `median` or `standardize` | `gpu_native_frequency` |
| no explicit GPU preprocessor matched and model routing resolved to `gpu_patch` | `gpu_patch` |
| CPU fallback tuples | `cpu_sklearn`, `cpu_frequency`, or `cpu_native_frame` depending on the categorical preprocessor |

### Sparse Onehot Contract

- `categorical_preprocessor: onehot` is an internal runtime choice, not a user-facing dense/sparse switch.
- sparse CSR output: `ridge`, `elasticnet`, `logistic_regression`, `extra_trees`, `lightgbm`, `catboost`, `xgboost`, and `random_forest` when runtime does not resolve to `gpu_native`.
- dense array output: `hist_gradient_boosting`, and `random_forest` when runtime resolves to `gpu_native`.
- `numeric_preprocessor: kbins` follows the same sparse-versus-dense output contract when composed with `onehot`.

### Booster GPU Routing

- When GPU is active, `xgboost` adds `device="cuda"` and `catboost` adds `task_type="GPU"`.
- When `gpu_native` for `lightgbm`, the repo builds a `RepositoryLightGbmEstimator`, validates the installed CUDA build once per process, and trains with `device_type="cuda"`.
- The repo-owned LightGBM adapter coerces fit and predict inputs onto the same NumPy/SciPy boundary before calling LightGBM.
- The LightGBM `gpu_native` path keeps the sparse-CSR contract for `onehot`; CUDA training is available but explicit GPU preprocessing is limited to `frequency + median|standardize`.

### GPU Logistic Regression

- `gpu_patch`: keeps the sklearn estimator surface via RAPIDS `cuml.accel` hooks; supports `categorical_preprocessor: frequency` only.
- `gpu_native`: builds an explicit `cuml.LogisticRegression`; supports `frequency + standardize` only; rejects `model_params.class_weight`.
- Unsupported preprocessing combinations (ordinal, onehot, sparse kbins) are rejected before training.

### GPU Linear Regression

- `gpu_native` for `ridge`/`elasticnet`: builds explicit `cuml.Ridge`/`cuml.ElasticNet` estimators for `frequency + median|standardize`.
- Dense `cudf.DataFrame` outputs stay GPU-resident through fit and predict; predictions are flattened to 1D before scoring.
- `gpu_native` ridge accepts: `alpha`, `copy_X`, `fit_intercept`, `solver`.
- `gpu_native` elasticnet accepts: `alpha`, `fit_intercept`, `l1_ratio`, `max_iter`, `selection`, `solver`, `tol`.

### GPU Random Forest

- `gpu_native`: builds explicit `cuml.RandomForestClassifier`/`cuml.RandomForestRegressor`.
- Supported tuples: `frequency + median|standardize`, `onehot + median|standardize|kbins`.
- Requires dense inputs only; `onehot` flips from sparse CSR to dense arrays for this path.
- Normalizes `criterion` to `split_criterion`, `max_leaf_nodes` to `max_leaves`; rejects `n_jobs`.
- Accepted `model_params` subset: `bootstrap`, `criterion`, `max_batch_size`, `max_depth`, `max_features`, `max_leaf_nodes`, `max_samples`, `min_impurity_decrease`, `min_samples_leaf`, `min_samples_split`, `n_bins`, `n_estimators`, `n_streams`, `oob_score`, `random_state`.
- `max_depth: null` is not supported; omit it or set an explicit positive integer.

### GPU XGBoost

- `gpu_native` for `frequency + median|standardize`: fold-local preprocessing via repo-owned `cudf` path; dense outputs stay GPU-resident through fit and predict.
- Sparse CSR preprocessing output is rejected (includes `onehot` and sparse `kbins`).
- Prediction outputs are coerced back to NumPy before scoring.

### GPU CatBoost

- `gpu_native` for `categorical_preprocessor: native` only; uses CatBoost's own GPU mode, not the RAPIDS patch layer.
- Preprocessing stays on `cpu_native_frame`.

### RealMLP Native Categorical Path

- `realmlp` now accepts `categorical_preprocessor: native` in both binary and regression configs.
- Native RealMLP uses the existing `cpu_native_frame` preprocessing backend: repository numeric preprocessing still runs first, while categorical columns stay as raw pandas object/string columns.
- During fold-local fit, the runtime passes `cat_col_names` so the upstream `RealMLP_TD_*` estimators can distinguish native categorical columns.
- Existing non-native RealMLP paths (`onehot`, `ordinal`, `frequency`) are unchanged.

### GPU Preprocessing

- `gpu_cuml`: explicit dense GPU preprocessing for `onehot` with `median`, `standardize`, or `kbins`.
- `gpu_native_frequency`: explicit repo-owned `frequency` preprocessing backend.
- `gpu_patch`: stays on existing sklearn/pandas constructors with RAPIDS hooks installed.
- Other schemes stay on CPU unless the runtime is using `gpu_patch`.
- On GPU hosts, preprocessing can resolve to GPU even when the model backend falls back to CPU; in hybrid cases, preprocessed outputs are coerced back to CPU before fit.

### GPU Dependency Contract

- Base dependencies pin `numpy` and `pandas` into the RAPIDS-compatible range.
- Optional GPU dependencies live behind the `gpu` extra.
- Currently targets Python `>=3.13,<3.14` on Linux `x86_64` CUDA 12.
- CPU and macOS installs use `uv sync` or `uv sync --extra boosters`.
- The LightGBM CUDA contract: sync with `--extra boosters --extra gpu`, run `install_lightgbm_cuda.sh`, validate with `validate_lightgbm_cuda_build.py`.

## Module Responsibilities

- [main.py](/main.py): thin repository-root wrapper that inserts `src/` on `sys.path` and forwards into the bootstrap entrypoint.
- [bootstrap.py](/src/tabular_shenanigans/bootstrap.py): pre-runtime bootstrap hook point that resolves execution mode and installs RAPIDS hooks before runtime modules import `pandas` or `sklearn`.
- [bootstrap_config.py](/src/tabular_shenanigans/bootstrap_config.py): lightweight YAML reader for the bootstrap-only runtime settings loaded before the full config model.
- [execution_routing.py](/src/tabular_shenanigans/execution_routing.py): repo-owned tuple support registry and routing resolver for `cpu`, `gpu_patch`, and `gpu_native`.
- [cli.py](/src/tabular_shenanigans/cli.py): CLI parser and linear stage dispatch after bootstrap completes.
- [runtime_execution.py](/src/tabular_shenanigans/runtime_execution.py): runtime capability detection, RAPIDS hook activation, requested-versus-resolved execution context, and bootstrap/runtime metadata helpers.
- [preprocess_execution.py](/src/tabular_shenanigans/preprocess_execution.py): preprocessing backend selection for CPU, `gpu_patch`, and explicit GPU-native paths.
- [gpu_cuml_preprocess.py](/src/tabular_shenanigans/gpu_cuml_preprocess.py): explicit dense GPU preprocessing adapters built on maintained cuML preprocessing constructors.
- [lightgbm_cuda_backend.py](/src/tabular_shenanigans/lightgbm_cuda_backend.py): repo-owned LightGBM adapter, input-coercion helpers, and CUDA build validation probe for the `gpu_native` LightGBM path.
- [competition.py](/src/tabular_shenanigans/competition.py): in-memory competition preparation, fold assignment materialization, and prepared-context construction.
- [config.py](/src/tabular_shenanigans/config.py): nested config validation, metric normalization, candidate-id derivation, and resolved model lookup.
- [candidate_artifacts.py](/src/tabular_shenanigans/candidate_artifacts.py): shared manifest/config helpers and temp-bundle file writers for candidate/context artifacts.
- [data.py](/src/tabular_shenanigans/data.py): Kaggle downloads, zip access, schema inference, and sample-submission loading.
- [eda.py](/src/tabular_shenanigans/eda.py): local EDA report generation.
- [representations/](/src/tabular_shenanigans/representations): representation runtime — operator/pruner specs, feature-bundle compilation, compatibility contract derivation, and final materialization. Replaces the former `feature_recipes/` and registered `representation_id` registry with explicit representation config.
- [model_evaluation.py](/src/tabular_shenanigans/model_evaluation.py): shared prepared training context, reusable CV evaluation logic for train/tune, and fold-stage runtime profiling for benchmark checkpoints.
- [models.py](/src/tabular_shenanigans/models.py): model registry, capability checks, estimator construction (runtime/task-contract params only; hyperparameter defaults are upstream), and tuning space definitions.
- [preprocess.py](/src/tabular_shenanigans/preprocess.py): raw feature-frame preparation, `NativeFramePreprocessor`, and `FrequencyFramePreprocessor` backends used by GPU preprocessing paths.
- [cv.py](/src/tabular_shenanigans/cv.py): splitters and task-aware metric scoring.
- [train.py](/src/tabular_shenanigans/train.py): model training workflow, candidate manifest construction, temp bundle staging, MLflow candidate logging, and training-stage runtime profiling capture.
- [training_orchestration.py](/src/tabular_shenanigans/training_orchestration.py): configured-candidate selection, sequential batch execution, optional skip-existing behavior, and batch summary reporting.
- [blend.py](/src/tabular_shenanigans/blend.py): MLflow-backed base-candidate loading, compatibility checks, weighted blending, and blended candidate logging.
- [tune.py](/src/tabular_shenanigans/tune.py): Optuna orchestration on top of the shared model-evaluation layer.
- [submit.py](/src/tabular_shenanigans/submit.py): MLflow-backed candidate resolution, submission validation, Kaggle submit orchestration, and submission refresh.
- [submission_history.py](/src/tabular_shenanigans/submission_history.py): candidate-run submission event/observation models and Kaggle refresh helpers.
- [mlflow_store.py](/src/tabular_shenanigans/mlflow_store.py): MLflow experiment/run lookup, candidate-run creation, candidate download, artifact upload, and submission-history persistence.
- [benchmark_gpu_checkpoint.py](/scripts/benchmark_gpu_checkpoint.py): issue-scoped benchmark harness for the early CPU vs `gpu_patch` vs `gpu_native` checkpoint, using a temporary file-based MLflow store and timestamped reports under `reports/benchmark_checkpoints/`.
- [validate_gpu_target_matrix.py](/scripts/validate_gpu_target_matrix.py): issue-scoped target-host smoke harness for `#193`, using a temporary file-based MLflow store, bootstrap/install validation, and timestamped reports under `reports/gpu_target_validation/`.
- [validate_lightgbm_cuda_build.py](/scripts/validate_lightgbm_cuda_build.py): repo-owned validation probe for the installed LightGBM CUDA build on the current host.
- [install_lightgbm_cuda.sh](/scripts/install_lightgbm_cuda.sh): source-build helper that reinstalls LightGBM with `USE_CUDA=ON` into the project virtualenv and immediately runs the validation probe.

## Representation Operator Catalog

### Numeric Operators
- `native_numeric` — passthrough numeric columns as native tabular
- `standardize_numeric` — median impute + standard scale
- `robust_scale_numeric` — median impute + robust scale
- `signed_log_expand_numeric` — median impute + signed log1p transform
- `quantile_bin_numeric` — median impute + quantile binning (sparse onehot)

### Categorical Operators
- `native_categorical` — passthrough categoricals as native tabular (normalized strings)
- `frequency_encode_categoricals` — per-column frequency encoding
- `ordinal_encode_categoricals` — ordinal encoding with unknown handling
- `onehot_encode_low_cardinality_categoricals` — sparse onehot for low-cardinality columns
- `target_encode_categoricals` — smoothed target encoding (supervised)
- `rare_category_bucket` — sparse onehot indicator for rare categories

### Interaction Operators
- `multiply_numeric_pairs` — pairwise product of all C(n,2) numeric column pairs
- `ratio_numeric_pairs` — pairwise ratio a/(b+eps) of all C(n,2) numeric column pairs
- `difference_numeric_pairs` — pairwise difference of all C(n,2) numeric column pairs
- `sum_numeric_pairs` — pairwise sum of all C(n,2) numeric column pairs
- `cross_low_cardinality_categoricals` — sparse onehot of all C(n,2) low-cardinality categorical crosses
- `cross_categorical_with_binned_numeric` — sparse onehot of (binned numeric x low-cardinality categorical) crosses
- `groupwise_deviation_features` — per-group z-score deviation for (numeric, categorical) pairs
- `frequency_encode_categorical_crosses` — frequency encoding of all C(n,2) low-cardinality categorical crosses

### Utility Operators
- `row_missing_count` — count of missing values per row

### Pruners
- `high_correlation_prune` — drop dense columns exceeding a correlation threshold

## Extension Notes

- After a few real runs, revisit which params are actually worth showing in the runs table, which metrics are redundant, whether some artifacts should be dropped or renamed, and whether candidate-level submission history should expose more derived leaderboard metadata.
- The MLflow schema is intentionally lean in this iteration.
- The runtime does not ship custom GPU implementations for `extra_trees` or `hist_gradient_boosting`; a future GPU path would need to come from a maintained upstream library and then be added to the registry.
