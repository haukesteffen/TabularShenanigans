# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the system as it exists and the contracts it is expected to preserve.

The intended operating scope is Kaggle Playground Series tabular competitions. Current default development targets are `playground-series-s5e12` for binary classification with `primary_metric: roc_auc` and `playground-series-s5e10` for regression with `primary_metric: mse`. The current binary production target is `playground-series-s6e3` with `primary_metric: roc_auc`.

## System Flow
1. Load and validate the local repository-root `config.yaml`.
2. Use explicit `task_type` and `primary_metric` from config.
3. Download the competition zip into `data/<competition_slug>/` when it is missing.
4. Load one shared dataset context from `train.csv`, `test.csv`, and `sample_submission.csv`.
5. Run `prepare` to write report CSVs under `reports/<competition_slug>/`, persist `artifacts/<competition_slug>/competition.json`, and freeze `artifacts/<competition_slug>/folds.csv`.
6. Resolve `id_column` and `label_column` from `train.csv`, `test.csv`, and `sample_submission.csv`, then prepare raw feature frames from the train/test data with the resolved `id_column` excluded from modeled features.
7. For model candidates during training and tuning, apply the selected deterministic feature recipe to the raw feature frames. The default `identity` recipe leaves the features unchanged.
8. For model candidates during training, load the frozen fold assignments from `folds.csv` and build fold-local preprocessing from the selected model recipe:
   - `onehot`: numeric median imputation + `StandardScaler`; categorical most-frequent imputation + `OneHotEncoder`
   - `ordinal`: numeric median imputation; categorical most-frequent imputation + `OrdinalEncoder`
   - `native`: numeric median imputation inside a pandas frame; categorical missing-value fill with native categorical columns preserved for CatBoost
   - `frequency`: numeric median imputation; categorical values replaced by fold-local relative frequencies, with unseen categories mapped to `0.0`
9. For model candidates, resolve the current `experiment.candidate.model_family + preprocessor` combination to one internal canonical model recipe, then train that one configured candidate:
  - regression: `ridge + onehot`, `elasticnet + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `hist_gradient_boosting + frequency`, `lightgbm + ordinal`, `lightgbm + frequency`, `catboost + native`, `xgboost + ordinal`, `xgboost + frequency`
  - binary classification: `logistic_regression + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `hist_gradient_boosting + frequency`, `lightgbm + ordinal`, `lightgbm + frequency`, `catboost + native`, `xgboost + ordinal`, `xgboost + frequency`
10. For blend candidates, load compatible base candidate artifacts from `artifacts/<competition_slug>/candidates/<base_candidate_id>/`, validate shared schema plus frozen-fold alignment, validate the binary probability label contract when `primary_metric` is `roc_auc` or `log_loss`, and materialize blended OOF plus test predictions without retraining the base candidates.
11. When `experiment.candidate.optimization.enabled=true` for a model candidate, `train` runs an Optuna study on the frozen fold assignments, retrains the best trial into the standard candidate artifact layout, and writes optimization metadata inside the candidate directory.
12. Write one candidate artifact directory under `artifacts/<competition_slug>/candidates/<candidate_id>/` with `candidate.json`, `fold_metrics.csv`, `oof_predictions.csv`, `test_predictions.csv`, and optional candidate-type-specific files such as `blend_summary.csv` or optimization metadata.
13. Validate predictions against `sample_submission.csv`, including exact ID content and order, using `candidate.json` as the submission metadata contract, apply metric-aware binary prediction validation, write `submission.csv` in the selected candidate directory, and optionally submit to Kaggle.
14. When `experiment.tracking.enabled=true`, publish stage-local artifacts and metadata for `prepare`, `train`, and `submit` to the configured MLflow server after the stage succeeds.

## CLI Stages
- `uv run python main.py`: default full pipeline (`fetch` -> `prepare` -> `train` -> `submit`)
- `uv run python main.py fetch`: ensure competition data is present
- `uv run python main.py prepare`: fetch if needed, load the shared dataset context, write EDA reports, persist competition metadata, and freeze folds
- `uv run python main.py eda`: fetch if needed, load the shared dataset context, and write EDA reports
- `uv run python main.py train`: fetch if needed, load the shared dataset context, auto-run `prepare` when needed, then either train one model candidate or materialize one blend candidate on the frozen folds; when optimization is enabled for a model candidate, run Optuna plus candidate-local optimization artifact writing before the final retrain
- `uv run python main.py submit`: prepare or submit the configured `candidate_id`
- `uv run python main.py submit --candidate-id <candidate_id>`: prepare or submit another existing candidate for the configured competition

The default `submit` path supports current candidate artifacts only. Unsupported or missing candidate artifacts fail directly.

## Module Responsibilities
- `main.py`: orchestration entrypoint for config loading plus stage-specific CLI dispatch across fetch, prepare, EDA, training, and submission.
- `src/tabular_shenanigans/competition.py`: competition-level preparation, `competition.json` persistence, `folds.csv` persistence, prepared-context validation, and split reconstruction from frozen folds.
- `src/tabular_shenanigans/config.py`: Pydantic-backed nested config schema for `competition` plus `experiment`, metric normalization, candidate-to-model resolution, and runtime contract validation.
- `src/tabular_shenanigans/data.py`: competition download, zip access, metric helpers, dataset schema resolution, and sample-submission template loading.
- `src/tabular_shenanigans/eda.py`: competition-scan EDA summaries written to CSV from the shared dataset context, including missingness, categorical cardinality, target summary, and feature-type counts.
- `src/tabular_shenanigans/feature_recipes/*`: deterministic experiment-scoped feature transforms, including the `identity` default and tracked competition-specific recipe modules.
- `src/tabular_shenanigans/models.py`: model-recipe registry, candidate `model_family + preprocessor` resolution, tunable-model search spaces, optional booster loading, and estimator construction for supported presets.
- `src/tabular_shenanigans/preprocess.py`: feature frame preparation, column typing, scheme-specific preprocessing pipelines, and native-frame support for CatBoost.
- `src/tabular_shenanigans/cv.py`: task-aware CV splitters and metric scoring helpers.
- `src/tabular_shenanigans/blend.py`: blend-candidate validation, base-candidate artifact loading, weighted prediction combination, and `blend_summary.csv` writing.
- `src/tabular_shenanigans/train.py`: config-selected training from the shared dataset context, frozen-fold loading, candidate artifact writing, candidate manifest generation, and optimization-aware training orchestration.
- `src/tabular_shenanigans/tune.py`: internal Optuna helper used by `train` when candidate optimization is enabled.
- `src/tabular_shenanigans/submit.py`: submission schema validation, candidate selection by `candidate_id`, submission message creation, Kaggle submission, and submission ledger updates.
- `src/tabular_shenanigans/tracking.py`: optional MLflow run creation, tag/metric logging, config snapshot logging, and post-stage artifact publishing.

## Configuration Contract
Input:
- One local runtime config file: `config.yaml` (single source of truth)
- Tracked example configs: `config.binary.example.yaml` and `config.regression.example.yaml`
- Expected workflow: copy one example file to `config.yaml`, then edit `config.yaml` for the local run
- Required top-level keys:
  - `competition`
  - `experiment`
- `competition` keys:
  - `slug`
  - `task_type` (`regression` or `binary`)
  - `primary_metric` (`rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`)
  - optional `positive_label` (explicit positive class for binary competitions; required unless observed labels match one of the documented safe conventions `[0, 1]`, `[False, True]`, or `["No", "Yes"]`)
  - optional schema overrides:
    - `id_column`
    - `label_column`
  - `cv`:
    - `n_splits` (integer >= 2, default 7)
    - `shuffle` (boolean, default true)
    - `random_state` (integer, default 42)
  - optional `features` block:
    - `force_categorical` (list of column names)
    - `force_numeric` (list of column names)
    - `drop_columns` (list of additional modeled-feature columns to remove after the ID column is excluded by default)
    - `low_cardinality_int_threshold` (positive integer)
- `experiment` keys:
  - `name`
  - optional `notes`
  - required `candidate`
  - optional `submit`
  - optional `tracking`
- Current `experiment.candidate` keys:
  - shared:
    - `candidate_type` (`model` or `blend`)
    - `candidate_id`
  - model candidate:
    - `feature_recipe_id` (string, default `identity`; resolves to a tracked deterministic feature recipe applied before preprocessing)
    - `preprocessor` (`onehot`, `ordinal`, `native`, or `frequency`)
    - `model_family`
      - regression: `ridge`, `elasticnet`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
      - binary classification: `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
    - optional `model_params`
    - optional `optimization`:
      - `enabled` (boolean, default false)
      - `method` (currently only `optuna`)
      - `n_trials` (integer >= 1, optional)
      - `timeout_seconds` (integer >= 1, optional)
      - `random_state` (integer, default 42)
  - blend candidate:
    - `base_candidate_ids` (list of at least two existing compatible candidate IDs)
    - optional `weights` (positive numeric weights with equal-weight default)
- Current `experiment.submit` keys:
  - `enabled` (boolean, default false)
  - `message_prefix` (string, optional)
- Current `experiment.tracking` keys:
  - `enabled` (boolean, default false)
  - `tracking_uri` (string, required when enabled)
  - `experiment_name` (string, required when enabled)

Binary prediction artifact contract:
- `roc_auc` and `log_loss` write positive-class probabilities to `test_predictions.csv` and `submission.csv`
- `accuracy` writes predicted class labels to `test_predictions.csv` and `submission.csv`

The config is validated by Pydantic with `extra="forbid"`. Unknown keys, schema mismatches, and missing required fields are hard errors.
Configured metrics are normalized to the internal metric names during config validation.
The old flat config layout is unsupported and fails fast.
The current runtime resolves `experiment.candidate.feature_recipe_id` to one tracked feature recipe for model candidates; built-in recipe IDs are `identity` and `s6e3_v1`.
The current runtime resolves `experiment.candidate.model_family + experiment.candidate.preprocessor` to one internal canonical `model_id` for model candidates.
Blend candidates consume compatible existing candidate artifacts and materialize a synthetic `blend_weighted_average` artifact model ID.
optimization requires at least one stopping condition: `experiment.candidate.optimization.n_trials` or `experiment.candidate.optimization.timeout_seconds`.
enabled optimization is consumed by `train` and applies to model candidates only.
Frequency encoding is fold-local and maps unseen categorical values to `0.0`.
LightGBM, CatBoost, and XGBoost require the optional booster dependencies installed via `uv sync --extra boosters`.
MLflow tracking requires the optional tracking dependencies installed via `uv sync --extra tracking`.

## Preferred Verification Targets
- `playground-series-s5e12`: binary development smoke test with `task_type: binary` and `primary_metric: roc_auc`
- `playground-series-s6e3`: binary production-target smoke test with `task_type: binary` and `primary_metric: roc_auc`
- `playground-series-s5e10`: regression smoke test with `task_type: regression` and `primary_metric: mse`

Manual verification steps for each target:
- copy the corresponding tracked example config to `config.yaml`
- verify the competition assets include `train.csv`, `test.csv`, and `sample_submission.csv`
- run `uv run python main.py prepare`
- confirm `artifacts/<competition_slug>/competition.json` and `artifacts/<competition_slug>/folds.csv` are generated
- run the workflow from a clean repo state with explicit `competition` plus one current `experiment.candidate`
- confirm inferred `id_column` and `label_column`
- confirm `candidate.json` is generated in the candidate directory
- confirm `test_predictions.csv` is generated in the candidate directory
- run `uv run python main.py submit`
- confirm `submission.csv` validates against `sample_submission.csv`, including exact ID values and order, for the selected candidate directory
- confirm binary outputs match the configured metric contract: probabilities for `roc_auc`/`log_loss`, labels for `accuracy`
- when optimization is enabled, run `uv run python main.py train` and confirm `optimization_summary.json`, `optimization_trials.csv`, and `optimization_best_params.json` are generated in the candidate directory
- when optimization is enabled, confirm the best-trial retrain writes a standard candidate artifact and records `tuning_provenance` in `candidate.json`

## Artifact Contract
- A validated in-memory config object from Pydantic
- Competition files downloaded under `data/<competition_slug>/`
- EDA summary printed to terminal
- Competition context artifacts under `artifacts/<competition_slug>/`:
  - `competition.json`
  - `folds.csv`
- EDA report CSV files under `reports/<competition_slug>/`
  - `columns_train.csv`
  - `columns_test.csv`
  - `missingness_summary.csv`
  - `categorical_cardinality_summary.csv`
  - `target_summary.csv`
  - `feature_type_counts.csv`
  - `run_summary.csv`
- Candidate artifacts under `artifacts/<competition_slug>/candidates/<candidate_id>/`:
  - `candidate.json`
  - `fold_metrics.csv`
  - `oof_predictions.csv`
  - `test_predictions.csv`
  - `submission.csv` when prepared or submitted
- model candidates also record `feature_recipe_id` plus the engineered `feature_columns`
- blend candidates also include `blend_summary.csv` and record component candidate provenance plus normalized weights in `candidate.json`
- optimized model candidates record `tuning_provenance` in `candidate.json`
- optimized candidates also include:
  - `optimization_summary.json`
  - `optimization_trials.csv`
  - `optimization_best_params.json`
- Append-only submission ledger at `artifacts/<competition_slug>/submissions.csv` keyed by `candidate_id`
- Optional MLflow-published stage artifacts:
  - `prepare`: config snapshot, `competition.json`, `folds.csv`, and reports
  - `train`: config snapshot and the full candidate artifact directory
  - `submit`: config snapshot, `submission.csv`, and submission metadata JSON

## Runtime Invariants And Failure Behavior
- One runtime config source only: local repository-root `config.yaml`
- No config overrides via CLI or environment variables
- Tracked example config files are documentation and starting points only; they are never read automatically at runtime
- top-level `competition` and `experiment` sections must be present in config for every run
- the old flat config layout is unsupported
- when tracking is enabled, MLflow publishing is part of stage success and happens after local files are written
- `competition.task_type` and `competition.primary_metric` must be present in config for every run
- `prepare` is the competition-level source of truth for `competition.json` and `folds.csv`
- `train` must consume the prepared fold assignments and fail if the prepared context no longer matches the current config or resolved dataset schema
- the current runtime supports one `experiment.candidate` of type `model` or `blend`
- model candidates: `experiment.candidate.feature_recipe_id` must resolve to one supported tracked recipe; `identity` is the default path for new competitions
- model candidates: `experiment.candidate.model_family + experiment.candidate.preprocessor` must resolve to one supported canonical recipe for the configured task
- blend candidates: `experiment.candidate.base_candidate_ids` must resolve to existing compatible candidate artifacts for the same competition context
- binary probability blend candidates: all base candidates must share the same saved `positive_label`, `negative_label`, and `observed_label_pair` contract
- feature recipes are experiment-scoped, deterministic, leakage-safe transforms applied after raw feature extraction and before preprocessing
- feature recipes must preserve row counts and row order and must produce identical train/test feature columns
- training must write exactly one candidate artifact directory keyed by `candidate_id`
- rerunning an existing `candidate_id` must fail instead of mutating an existing artifact directory
- enabled optimization is part of `train`, applies to model candidates only, and retrains exactly one tuned candidate into the normal candidate artifact layout
- enabled optimization must have at least one stopping condition: `experiment.candidate.optimization.n_trials` or `experiment.candidate.optimization.timeout_seconds`
- `submit` must resolve one candidate by `candidate_id`, defaulting to `config.candidate_id`
- `native_catboost` must preserve categorical feature positions through preprocessing so CatBoost can receive `cat_features`
- Kaggle CLI and authentication are expected to be preconfigured
- Competition zip contents are expected to include `train.csv`, `test.csv`, and `sample_submission.csv`
- Binary classification supports any two-class target labels
- The positive class is resolved from the training target and used consistently for diagnostics and scoring
- Binary `roc_auc` and `log_loss` artifacts use positive-class probabilities
- Binary `accuracy` artifacts use class labels from the observed binary label pair
- Binary classification must have an explicit positive-class contract; when `positive_label` is omitted, the workflow only auto-resolves the positive class for `[0, 1]`, `[False, True]`, or `["No", "Yes"]`
- `id_column` inference must resolve to exactly one column present in `train.csv`, `test.csv`, and `sample_submission.csv`
- The resolved `id_column` is identifier metadata and must be excluded from preprocessing and model fitting by default
- `label_column` inference must resolve to exactly one column present in `train.csv` and `sample_submission.csv` but not `test.csv`
- Submission must resolve `competition_slug`, `task_type`, `id_column`, and `label_column` from `candidate.json` rather than re-inferring them from raw train/test data
- `sample_submission.csv` must match the resolved schema exactly as `[id_column, label_column]`
- The selected candidate artifact `test_predictions.csv[id_column]` must match `sample_submission.csv[id_column]` exactly in both values and row order
- Submission requires the current candidate prediction layout at `artifacts/<competition_slug>/candidates/<candidate_id>/test_predictions.csv`
- Feature override columns must exist and cannot overlap between forced numeric and forced categorical sets
- Configured metric must normalize to a supported metric compatible with the configured task type
- CV splitter construction must support both `cv_shuffle=true` and `cv_shuffle=false`
- Submission output must match the schema, row count, and ID ordering/content of `sample_submission.csv`
- Fail fast is the default behavior; errors are surfaced directly with minimal wrapping

Hard-error cases include:
- Missing config file -> hard error
- Schema/type violation -> hard error
- Any attempt to use additional config sources -> hard error
- Kaggle command failure -> hard error (bubble up with minimal wrapping)
- Missing `task_type` or `primary_metric` -> hard error
- Unknown/unsupported configured `primary_metric` -> hard error
- Invalid task/metric pairing (for example `binary` + `rmse`) -> hard error
- Missing required top-level `competition` or `experiment` sections -> hard error
- Any use of the old flat config layout -> hard error
- Unsupported `experiment.candidate.candidate_type` -> hard error
- Unknown `experiment.candidate.feature_recipe_id` for a model candidate -> hard error
- Invalid configured `experiment.candidate.model_family + preprocessor` combination for a model candidate -> hard error
- enabled optimization without `experiment.candidate.optimization.n_trials` or `experiment.candidate.optimization.timeout_seconds` -> hard error
- enabled optimization for an unsupported model candidate combination -> hard error
- Missing/invalid competition zip contents -> hard error
- Missing `competition.json` or `folds.csv` during `train` -> auto-run `prepare`
- Prepared competition context that no longer matches the current config or resolved dataset schema -> hard error
- `id_column` inference not exactly one column -> hard error
- `label_column` inference not exactly one column -> hard error
- Invalid `id_column` or `label_column` override -> hard error
- Unknown columns in `force_categorical`, `force_numeric`, or `drop_columns` -> hard error
- Any overlap between `force_categorical` and `force_numeric` -> hard error
- No modeled feature columns remaining after excluding `id_column` and applying `drop_columns` -> hard error
- Feature recipe output that changes row counts, row order, or train/test feature-column alignment -> hard error
- Preprocessing fit/transform failure -> hard error
- Unsupported task type for CV/model selection -> hard error
- Unsupported metric for chosen task -> hard error
- Any CV/training fit or scoring failure -> hard error
- Blend base candidate artifact missing `candidate.json`, `oof_predictions.csv`, or `test_predictions.csv` -> hard error
- Blend base candidate mismatch in competition slug, task type, primary metric, schema, binary probability label contract, OOF row order, fold assignments, or test ID order -> hard error
- Fold assignment gaps in OOF generation -> hard error
- Candidate artifact directory already exists for the configured `candidate_id` -> hard error
- Missing configured candidate artifacts at submit time -> hard error
- Requested `candidate_id` with no matching candidate artifact directory -> hard error
- Submission schema or ID mismatch against `sample_submission.csv` -> hard error
- Binary probability artifact outside `[0, 1]` for `roc_auc` or `log_loss` -> hard error
- Binary label artifact containing values outside the observed label pair for `accuracy` -> hard error
- Kaggle submission command failure when `experiment.submit.enabled=true` -> hard error

## Design Guardrails
- Keep implementation simple and avoid speculative abstractions.
- Stay CPU-first by default; treat GPU support as an optional later backend.
- Keep data and modeling logic behind small internal interfaces so backend swaps stay localized.
- Avoid scattering direct pandas/sklearn calls across many modules.
- Prefer columnar and vectorized transformations over row-wise Python loops.
- Use portable artifacts such as `csv` and `json`.
- Keep runs reproducible from config and append-only ledgers.
- Keep EDA script-driven and CSV-oriented rather than notebook- or plot-first.

## Extension Notes
- New user-facing config keys should be added to the relevant nested config model in `config.py` and documented in both this file and `README.md`.
- If the user-facing config workflow changes, update `config.binary.example.yaml` and `config.regression.example.yaml` alongside the docs.
- New metrics should be normalized and validated during config loading, then scored in `cv.py`.
- New model families should be introduced in `models.py` with explicit task compatibility, canonical recipe IDs, and matching artifact outputs.
- New feature recipes should be added under `src/tabular_shenanigans/feature_recipes/`, registered explicitly, and kept deterministic plus leakage-safe.
- New preprocessing modes should be added in `preprocess.py` without breaking the existing feature-frame contract or the preprocessing-first recipe naming convention.
- New candidate or submission artifacts should be reflected in both the artifact contract above and the corresponding ledger rows.
