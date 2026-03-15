# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the runtime contracts as they exist now.

## System Flow
1. Enter the bootstrap entrypoint before importing runtime modules that depend on `pandas` or `sklearn`.
2. Read `experiment.runtime.compute_target` and the optional advanced `experiment.runtime.gpu_backend` override from repository-root `config.yaml`, resolve hardware capability separately from tuple routing, resolve preprocessing backend selection separately from model routing, install RAPIDS hooks only when the registry resolves the current tuple to `gpu_patch`, and route GPU-capable booster families onto their GPU parameter paths.
3. Load and validate the repository-root `config.yaml`.
4. Normalize and validate `competition.task_type`, `competition.primary_metric`, and the candidate contract.
5. Resolve the MLflow tracking URI from `experiment.tracking.tracking_uri`.
6. Download the competition zip into `data/<competition_slug>/` when it is missing.
7. Load one shared dataset context from `train.csv`, `test.csv`, and `sample_submission.csv`.
8. Resolve `id_column` and `label_column`, then prepare raw feature frames with the resolved ID column excluded from modeled features.
9. For model candidates, apply the selected deterministic feature recipe.
10. Build the competition fold assignments in memory from the configured CV settings.
11. For model candidates, fit the selected preprocessing + model combination fold-locally and produce OOF/test predictions.
12. For blend candidates, download compatible base candidates from the competition MLflow experiment, validate compatibility, and combine their saved predictions without retraining the base candidates.
13. Stage the candidate bundle into a temp directory:
    - `config/runtime_config.json`
    - `context/competition.json`
    - `context/folds.csv`
    - `candidate/*`
14. Create one MLflow run for the candidate and upload the staged bundle.
15. For real Kaggle submissions, download the candidate from MLflow, validate `test_predictions.csv` against `sample_submission.csv`, submit `submission.csv`, and append submission history artifacts back onto that same candidate run.
16. For submission refresh, scan Kaggle submissions once, match `submit=<submission_event_id>` descriptions, and update candidate-run submission history plus scoreboard metrics in place.

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
- `tracking_schema_version=3`
- `competition_slug`
- `candidate_id`
- `candidate_type`
- `task_type`
- `primary_metric`
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
Current candidate-run params:
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
- `main.py` keeps the existing user-facing command but now delegates into a bootstrap module before the runtime imports `pandas`- or `sklearn`-dependent modules.
- bootstrap resolves `experiment.runtime.compute_target` and optional `experiment.runtime.gpu_backend` for the current machine and installs RAPIDS hooks only when the resolved backend is `gpu_patch`.
- the GPU runtime contract assumes the environment was synced with `uv sync --extra boosters --extra gpu`; plain `uv run python main.py ...` is safe after that because the lockfile now pins the RAPIDS-compatible shared dependency range
- `prepare` no longer persists canonical competition metadata. It only prepares the context in memory and writes EDA reports.
- `train` is the only stage that creates candidate runs.
- `submit` and `refresh-submissions` mutate existing candidate runs by appending submission history and score metrics.

## Module Responsibilities
- [main.py](/Users/hs/dev/TabularShenanigans/main.py): thin repository-root wrapper that inserts `src/` on `sys.path` and forwards into the bootstrap entrypoint.
- [bootstrap.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/bootstrap.py): pre-runtime bootstrap hook point that resolves execution mode and installs RAPIDS hooks before runtime modules import `pandas` or `sklearn`.
- [bootstrap_config.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/bootstrap_config.py): lightweight YAML reader for the bootstrap-only runtime settings loaded before the full config model.
- [execution_routing.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/execution_routing.py): repo-owned tuple support registry and routing resolver for `cpu`, `gpu_patch`, and `gpu_native`.
- [cli.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/cli.py): CLI parser and linear stage dispatch after bootstrap completes.
- [runtime_execution.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/runtime_execution.py): runtime capability detection, RAPIDS hook activation, requested-versus-resolved execution context, and bootstrap/runtime metadata helpers.
- [preprocess_execution.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/preprocess_execution.py): preprocessing backend selection and preprocessor construction for CPU, `gpu_patch`, and the current explicit GPU-native frequency path.
- [gpu_cuml_preprocess.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/gpu_cuml_preprocess.py): explicit dense GPU preprocessing adapters built on maintained cuML preprocessing constructors.
- [lightgbm_cuda_backend.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/lightgbm_cuda_backend.py): repo-owned LightGBM adapter, input-coercion helpers, and CUDA build validation probe for the `gpu_native` LightGBM path.
- [competition.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/competition.py): in-memory competition preparation, fold assignment materialization, and prepared-context construction.
- [config.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/config.py): nested config validation, metric normalization, candidate-id derivation, and resolved model lookup.
- [candidate_artifacts.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/candidate_artifacts.py): shared manifest/config helpers and temp-bundle file writers for candidate/context artifacts.
- [data.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/data.py): Kaggle downloads, zip access, schema inference, and sample-submission loading.
- [eda.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/eda.py): local EDA report generation.
- [feature_recipes](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/feature_recipes): deterministic feature recipes such as `fr0`, `fr1`, `fr2`, `fr3`, and the `fr2_ablate_*` / `fr3_ablate_*` grouped ablation variants used for `s6e3` recipe studies.
- [model_evaluation.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/model_evaluation.py): shared prepared training context, reusable CV evaluation logic for train/tune, and fold-stage runtime profiling for benchmark checkpoints.
- [models.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/models.py): model registry, capability checks, estimator construction, and tuning space definitions.
- [preprocess.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/preprocess.py): raw feature-frame preparation and split preprocessing components.
- [cv.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/cv.py): splitters and task-aware metric scoring.
- [train.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/train.py): model training workflow, candidate manifest construction, temp bundle staging, MLflow candidate logging, and training-stage runtime profiling capture.
- [blend.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/blend.py): MLflow-backed base-candidate loading, compatibility checks, weighted blending, and blended candidate logging.
- [tune.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/tune.py): Optuna orchestration on top of the shared model-evaluation layer.
- [submit.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/submit.py): MLflow-backed candidate resolution, submission validation, Kaggle submit orchestration, and submission refresh.
- [submission_history.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/submission_history.py): candidate-run submission event/observation models and Kaggle refresh helpers.
- [mlflow_store.py](/Users/hs/dev/TabularShenanigans/src/tabular_shenanigans/mlflow_store.py): MLflow experiment/run lookup, candidate-run creation, candidate download, artifact upload, and submission-history persistence.
- [benchmark_gpu_checkpoint.py](/Users/hs/dev/TabularShenanigans/scripts/benchmark_gpu_checkpoint.py): issue-scoped benchmark harness for the early CPU vs `gpu_patch` vs `gpu_native` checkpoint, using a temporary file-based MLflow store and timestamped reports under `reports/benchmark_checkpoints/`.
- [validate_lightgbm_cuda_build.py](/Users/hs/dev/TabularShenanigans/scripts/validate_lightgbm_cuda_build.py): repo-owned validation probe for the installed LightGBM CUDA build on the current host.
- [install_lightgbm_cuda.sh](/Users/hs/dev/TabularShenanigans/scripts/install_lightgbm_cuda.sh): source-build helper that reinstalls LightGBM with `USE_CUDA=ON` into the project virtualenv and immediately runs the validation probe.

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
- optional `runtime`
- required `candidate`
- optional `submit`

`experiment.tracking`:
- `tracking_uri` only

`experiment.runtime`:
- `compute_target`: `auto`, `cpu`, or `gpu`
  - `auto`: choose the best registered implementation for the current tuple on the current machine: `gpu_native`, then `gpu_patch`, otherwise CPU fallback
  - `cpu`: stay on CPU
  - `gpu`: require a registered GPU implementation for the current tuple and fail fast otherwise
- optional `gpu_backend`: `auto`, `patch`, or `native`
  - advanced/transitional override; leave this at `auto` for normal use
  - `auto`: let the registry choose between `gpu_native` and `gpu_patch`
  - `patch`: require the registered RAPIDS hook-based `gpu_patch` path for the current tuple
  - `native`: require the registered explicit `gpu_native` path for the current tuple
  - tuple routing is registry-driven and is resolved during bootstrap for model candidates before `pandas` / `sklearn` imports happen
  - preprocessing backend selection is resolved separately from model routing and currently emits one of:
    - `cpu_sklearn`
    - `cpu_frequency`
    - `cpu_native_frame`
    - `gpu_cuml`
    - `gpu_patch`
    - `gpu_native_frequency`
  - current supported `gpu_native` tuple:
    - `model_family: logistic_regression`
    - `categorical_preprocessor: frequency`
    - `numeric_preprocessor: standardize`
    - `model_family: lightgbm`
    - `categorical_preprocessor: onehot`, `ordinal`, or `frequency`
    - `numeric_preprocessor: median`, `standardize`, or `kbins`
    - `model_family: xgboost`
    - `categorical_preprocessor: frequency`
    - `numeric_preprocessor: median` or `standardize`
    - `model_family: catboost`
    - `categorical_preprocessor: native`
    - `numeric_preprocessor: median`, `standardize`, or `kbins`
  - unsupported explicit `gpu_backend: patch` / `gpu_backend: native` requests fail fast with repo-owned errors
  - under `compute_target: auto`, tuples with no registered GPU implementation intentionally fall back to CPU
  - `extra_trees` and `hist_gradient_boosting` are currently intentional CPU-fallback families on GPU hosts because no maintained official GPU implementation is registered for them in this runtime
  - `#181` does not authorize custom GPU implementations for those algorithms; a future GPU path would need to come from a maintained upstream library and then be added to the registry

GPU dependency contract:
- base project dependencies pin `numpy` and `pandas` into the RAPIDS-compatible range used by both CPU and GPU installs
- optional GPU dependencies live behind the `gpu` extra
- the project currently supports Python `>=3.13,<3.14`
- the `gpu` extra currently targets Python 3.13 Linux `x86_64` CUDA 12 hosts via NVIDIA's Python package index
- expected install command on GPU hosts: `uv sync --extra boosters --extra gpu`
- `lightgbm` is still declared in the `boosters` extra, but the stock wheel is not accepted for the explicit `gpu_native` CUDA path
- the repo-owned LightGBM CUDA contract is:
  - sync the shared dependencies with `uv sync --extra boosters --extra gpu`
  - run `./scripts/install_lightgbm_cuda.sh` on the target Linux GPU host
  - validate the result with `PYTHONPATH=src uv run python scripts/validate_lightgbm_cuda_build.py`
- CPU and macOS installs should continue using `uv sync` or `uv sync --extra boosters`

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

Booster GPU routing contract:
- when runtime execution resolves to GPU, `xgboost` adds `device="cuda"`
- when runtime execution resolves to GPU, `catboost` adds `task_type="GPU"`
- when runtime execution resolves to `gpu_native` for `lightgbm`, the repo builds a `RepositoryLightGbmEstimator`, validates the installed CUDA build once per process, and trains with `device_type="cuda"`
- on GPU hosts, preprocessing can resolve to an explicit GPU backend even when the model backend still resolves to CPU; in those hybrid cases the runtime converts the preprocessed outputs back to CPU arrays before fit/predict
- `gpu_cuml` is the current maintained explicit preprocessing backend and currently supports dense `categorical_preprocessor: onehot` with numeric `median`, `standardize`, or `kbins`
- when runtime execution resolves to `gpu_patch`, preprocessing currently stays on the existing sklearn/pandas constructors and relies on the installed RAPIDS hooks rather than explicit repo-owned GPU preprocessing adapters
- when runtime execution resolves to `gpu_native` for `catboost`, the repo keeps CatBoost on the existing native categorical frame path and does not route it through the RAPIDS patch layer or the repo-owned `cudf` frequency preprocessor
- the current explicit GPU preprocessing surface is intentionally narrow:
  - `gpu_cuml` for dense `onehot` plus numeric `median`, `standardize`, or `kbins`
  - `gpu_native_frequency` for `categorical_preprocessor: frequency` with numeric `median` or `standardize`
  - other preprocessing schemes currently stay on CPU-backed constructors unless the runtime is using `gpu_patch`
- the LightGBM `gpu_native` path keeps the current sparse-CSR contract for `onehot`, so CUDA training is available there even though explicit GPU preprocessing is currently limited to the `frequency + median|standardize` slice
- the repo-owned LightGBM adapter coerces fit and predict inputs onto the same NumPy / SciPy boundary before calling LightGBM
  - this removes the current fit/predict feature-name mismatch instead of suppressing it
- when runtime execution resolves to GPU, `logistic_regression` continues to use the sklearn estimator surface but runs through the RAPIDS `cuml.accel` hook path
- user `model_params` still override repo defaults

XGBoost GPU-native input contract:
- when runtime execution resolves to GPU for `xgboost`, fold-local preprocessing still happens per fold so CV remains leakage-safe
- when runtime execution resolves to `gpu_native` for the supported XGBoost tuple, the repo uses the explicit native preprocessing path instead of CPU-side transformed folds plus conversion
- the current native XGBoost support matrix remains intentionally narrow:
  - `categorical_preprocessor: frequency`
  - `numeric_preprocessor: median` or `standardize`
- after preprocessing, supported dense fold outputs stay on GPU through `fit` and `predict`
  - the current supported slice produces `cudf.DataFrame` fold outputs
- prediction outputs are coerced back to NumPy before scoring and artifact assembly
- sparse CSR preprocessing output is rejected before training for the XGBoost GPU-native path
  - this currently covers `categorical_preprocessor: onehot` and related sparse `kbins` compositions
  - rationale: XGBoost does not support `cupyx` CSR inputs in this runtime

GPU logistic regression contract:
- when runtime execution resolves to `gpu_patch` for `logistic_regression`, the builder wraps the sklearn estimator in the existing binary-label encoding adapter before fit
- this keeps original competition labels in `model.classes_` while ensuring cuML sees numeric binary targets during fit
- the `gpu_patch` path currently supports `categorical_preprocessor: frequency` only
- unsupported preprocessing combinations are rejected before training for the GPU logistic path
  - this currently covers `categorical_preprocessor: ordinal`, `categorical_preprocessor: onehot`, and related sparse `kbins` compositions
  - rationale: the RAPIDS-hooked sklearn preprocessing stack is not stable yet for these branches in the current runtime
- when runtime execution resolves to `gpu_native` for `logistic_regression`, the repo builds an explicit `cuml.LogisticRegression` estimator instead of relying on sklearn interception
- the `gpu_native` logistic path currently supports:
  - `categorical_preprocessor: frequency`
  - `numeric_preprocessor: standardize`
- `gpu_native` logistic currently rejects `model_params.class_weight`; use `gpu_backend: patch` or CPU execution when class weighting is required

## Candidate Manifest Contract
Model candidate manifests currently record:
- identity: `candidate_id`, `candidate_type`, `competition_slug`, `task_type`, `primary_metric`
- provenance: `config_fingerprint`, `config_snapshot`, `mlflow_run_id`
- runtime execution: requested/ resolved compute target, acceleration backend, RAPIDS hook status, and hardware capability snapshot
- runtime profiling: training-context build time, fold-stage CV timings, artifact staging time, and first-fold matrix residency snapshots when collected
- model info: `model_family`, `model_registry_key`, `estimator_name`
- feature/preprocessing info: `feature_recipe_id`, `feature_columns`, `numeric_preprocessor`, `categorical_preprocessor`, `preprocessing_scheme_id`
- runtime-selected preprocessing backend: `preprocessing_backend`
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

## Runtime Constraints
- RAPIDS acceleration is only expected on Linux GPU runtimes.
- `auto` can fall back to CPU for either missing GPU hardware or unavailable RAPIDS hook modules.
- once RAPIDS hook installation starts, rollback is not guaranteed; install failures are treated as hard errors.
- LightGBM GPU routing still requires a CUDA-enabled LightGBM runtime build in the target environment.

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
