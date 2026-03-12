# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Primary target: Kaggle Playground Series tabular competitions.
- Task scope: regression and binary classification.
- Scalability direction: CPU-first local workflow now, with a future path to cloud GPU execution (RAPIDS `cudf`/`cuml`) without rewriting the full pipeline.

## Development Defaults
The repository is developed and manually verified primarily against these Playground Series competitions:
- default classification development target: `playground-series-s5e12` with `primary_metric: roc_auc`
- classification production target: `playground-series-s6e3` with `primary_metric: roc_auc`
- default regression target: `playground-series-s5e10` with `primary_metric: mse`

## Current Capabilities
- Load and validate a single local repository-root `config.yaml`.
- Fetch Kaggle competition data into `data/<competition_slug>/` when the zip is missing.
- Require explicit `task_type` and `primary_metric` in config.
- Separate stable competition settings from the current experiment candidate in config via top-level `competition` and `experiment` sections.
- Ship tracked binary and regression example configs that can be copied into the local runtime config.
- Persist a competition-level context under `artifacts/<competition_slug>/` with `competition.json` and `folds.csv`.
- Infer Playground-style submission schema from dataset files:
  - `id_column` as the only column shared by `train.csv`, `test.csv`, and `sample_submission.csv`
  - `label_column` as the only column shared by `train.csv` and `sample_submission.csv` but not `test.csv`
- Exclude the resolved `id_column` from modeled features by default; identifier columns are treated as metadata, not training signal.
- Generate terminal and CSV EDA summaries under `reports/<competition_slug>/`, including missingness, categorical cardinality, target summary, and feature-type counts.
- Freeze CV assignments once per prepared competition context and reuse them across `train`.
- Run stage-specific CLI entrypoints for `fetch`, `prepare`, `eda`, `train`, and submit-only flows.
- Materialize one explicit experiment candidate at a time:
  - train one cross-validated model candidate using an optional config-selected feature recipe plus config-selected preprocessing and model-family choices that resolve internally to the current canonical recipe IDs
  - or build one blend candidate from existing compatible candidate artifacts under the same prepared competition context
- Write one candidate artifact directory under `artifacts/<competition_slug>/candidates/<candidate_id>/`, including `candidate.json`, `fold_metrics.csv`, `oof_predictions.csv`, and `test_predictions.csv`, plus optional candidate-type-specific metadata such as `blend_summary.csv` or optimization files.
- When `experiment.candidate.optimization.enabled=true` for a model candidate, run Optuna inside `train`, retrain the best trial into the candidate artifact directory, and keep optimization metadata next to that candidate.
- Validate predictions against `sample_submission.csv`, including exact ID content and order, with task-aware binary prediction checks, and optionally submit to Kaggle from the current candidate artifact selected by `candidate_id`.
- Optionally publish `prepare`, `train`, and `submit` runs plus generated artifacts to a remote MLflow tracking server while preserving the current local file-based workflow.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- Optuna for local hyperparameter tuning
- MLflow for optional remote run tracking and artifact publishing
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart
1. Ensure Kaggle CLI access is already configured for your user.
2. Install dependencies with `uv sync`.
3. If you want LightGBM, CatBoost, or XGBoost model recipes, install the optional booster dependencies with `uv sync --extra boosters`.
4. If you want optional MLflow tracking, install the tracking dependencies with `uv sync --extra tracking`.
5. Copy a tracked example config to a local repository-root `config.yaml`.
6. Run `uv run python main.py`.

```bash
cp config.binary.example.yaml config.yaml
# or
cp config.regression.example.yaml config.yaml
```

`config.yaml` is the only runtime config source. It is intentionally ignored by Git so you can keep local competition-specific settings without committing them.

The current pipeline fetches competition data if needed, prepares competition metadata plus frozen folds, trains the current experiment candidate with fold-local preprocessing and task-aware diagnostics, writes prediction artifacts, and prepares a validated submission file.

## Stage Commands
`uv run python main.py` still runs the full default pipeline: fetch, prepare, train, and submit.

Available stage-specific commands:
- `uv run python main.py fetch`
- `uv run python main.py prepare`
- `uv run python main.py eda`
- `uv run python main.py train`
- `uv run python main.py submit`
- `uv run python main.py submit --candidate-id <candidate_id>`

Stage behavior:
- `fetch`: ensures competition data is present locally
- `prepare`: fetches if needed, writes EDA report CSVs, persists `competition.json`, and freezes `folds.csv`
- `eda`: fetches if needed, then writes EDA report CSVs
- `train`: fetches if needed, prepares competition context when it is missing, then writes one candidate artifact directory on the frozen folds; model candidates train on raw data plus preprocessing, while blend candidates combine existing compatible candidate artifacts without retraining base models; when optimization is enabled for a model candidate, `train` first runs Optuna and stores optimization metadata inside that candidate directory
- `submit`: resolves one candidate artifact by `candidate_id` and never retrains implicitly

`submit` defaults to `config.candidate_id`. Use `--candidate-id` only when you want to submit another existing candidate for the same competition.

## Config Overview
Tracked example configs:
- `config.binary.example.yaml`: binary-classification starting point using the default dev smoke-test target
- `config.regression.example.yaml`: regression starting point using the default regression smoke-test target

Runtime config workflow:
- copy one of the tracked examples to repository-root `config.yaml`
- edit only `config.yaml` for local runs
- keep `config.yaml` untracked; the application does not read the example files directly

The flat config layout is no longer supported. `config.yaml` must now contain top-level `competition` and `experiment` sections.

Required top-level sections:
- `competition`
- `experiment`

`competition` keys:
- `slug`
- `task_type`: `regression` or `binary`
- `primary_metric`: one of `rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`
- optional `positive_label`: explicit positive class for binary competitions; required unless the observed training labels follow one of the documented safe conventions: `[0, 1]`, `[False, True]`, or `["No", "Yes"]`
- optional schema overrides:
  - `id_column`
  - `label_column`
- `cv`:
  - `n_splits` (default `7`)
  - `shuffle` (default `true`)
  - `random_state` (default `42`)
- optional `features` block:
  - `force_categorical`
  - `force_numeric`
  - `drop_columns`
  - `low_cardinality_int_threshold`

`experiment` keys:
- `name`
- optional `notes`
- required `candidate` block
- optional `submit` block
- optional `tracking` block

Current `experiment.tracking` keys:
- `enabled`: if `true`, publish `prepare`, `train`, and `submit` runs to MLflow (default `false`)
- `tracking_uri`: remote MLflow tracking URI; required when tracking is enabled
- `experiment_name`: remote MLflow experiment name; required when tracking is enabled

Current `experiment.candidate` contract:
- shared keys:
  - `candidate_type`: `model` or `blend`
  - `candidate_id`
- model candidate keys:
  - optional `feature_recipe_id`: tracked deterministic feature recipe applied after raw feature extraction and before preprocessing/model fitting; defaults to `identity`
  - `preprocessor`: `onehot`, `ordinal`, `native`, or `frequency`
  - `model_family`
    - regression: `ridge`, `elasticnet`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
    - binary classification: `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
  - optional `model_params`
  - optional `optimization`:
    - `enabled`
    - `method`: currently only `optuna`
    - `n_trials`
    - `timeout_seconds`
    - `random_state`
- blend candidate keys:
  - `base_candidate_ids`: list of at least two existing compatible candidate IDs under the same competition
  - optional `weights`: positive numeric weights with equal-weight default

Blend candidates validate that all base candidates share the same competition slug, task type, primary metric, resolved schema, frozen fold assignments, OOF row order, and test ID order. For binary `roc_auc` and `log_loss` blends, base candidates must also share the same saved `positive_label`, `negative_label`, and `observed_label_pair` contract.

Supported `model_family + preprocessor` combinations for model candidates:
- regression: `ridge + onehot`, `elasticnet + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `hist_gradient_boosting + frequency`, `lightgbm + ordinal`, `lightgbm + frequency`, `catboost + native`, `xgboost + ordinal`, `xgboost + frequency`
- binary classification: `logistic_regression + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `hist_gradient_boosting + frequency`, `lightgbm + ordinal`, `lightgbm + frequency`, `catboost + native`, `xgboost + ordinal`, `xgboost + frequency`

Built-in `feature_recipe_id` values for model candidates:
- `identity`: default pass-through recipe for new competitions
- `s6e3_v1`: a first competition-specific feature set for `playground-series-s6e3`

Feature recipes live in tracked Python modules under `src/tabular_shenanigans/feature_recipes/`. They are intended for deterministic, leakage-safe competition-specific feature transforms. New competitions should start with `identity`; add a tracked recipe module only when the generic baseline plateaus.

`frequency` encodes each categorical value as its fold-local relative frequency in the training fold. Unseen categories at transform time map to `0.0`.

Current `experiment.submit` keys:
- `enabled`: if `true`, submit to Kaggle after training (default `false`)
- `message_prefix`: optional prefix used in auto-generated submission messages

Binary prediction artifact contract:
- `roc_auc` and `log_loss`: `test_predictions.csv` and `submission.csv` contain positive-class probabilities in `[0, 1]`
- `accuracy`: `test_predictions.csv` and `submission.csv` contain predicted class labels from the observed binary label set

If `id_column` or `label_column` are omitted, the training pipeline infers them from `train.csv`, `test.csv`, and `sample_submission.csv`. The resolved `id_column` is preserved for prediction outputs and in `candidate.json`, but it is not part of the model feature matrix. Submission preparation consumes the selected artifact manifest as the schema/task source of truth and uses `sample_submission.csv` only for validation. Invalid overrides, ambiguous inference, a `sample_submission.csv` shape that does not exactly match `[id_column, label_column]`, or a submission ID column that differs from `sample_submission.csv` in values or ordering are hard errors.

The current runtime resolves `experiment.candidate.model_family + experiment.candidate.preprocessor` into one internal canonical `model_id` for model candidates. Blend candidates materialize a synthetic `blend_weighted_average` artifact model ID and use their component candidate metadata as provenance.

`task_type` and `primary_metric` are always config-driven. The pipeline does not infer them from Kaggle metadata.

## Preferred Manual Verification Targets
Use these Playground competitions as the primary smoke tests:
- binary classification (dev): `playground-series-s5e12` with `task_type: binary` and `primary_metric: roc_auc`
- binary classification (prod target): `playground-series-s6e3` with `task_type: binary` and `primary_metric: roc_auc`
- regression: `playground-series-s5e10` with `task_type: regression` and `primary_metric: mse`

Tracked example configs for those targets:
- `config.binary.example.yaml`
- `config.regression.example.yaml`

Manual verification for each target:
- copy the corresponding example file to `config.yaml`
- confirm the competition archive includes `train.csv`, `test.csv`, and `sample_submission.csv`
- run `uv run python main.py prepare`
- confirm `artifacts/<competition_slug>/competition.json` and `artifacts/<competition_slug>/folds.csv` are written
- confirm the pipeline infers `id_column` and `label_column` without overrides
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/candidate.json` is written
- for model candidates, confirm `candidate.json` records the selected `feature_recipe_id`
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/test_predictions.csv` is written for the selected candidate artifact
- run `uv run python main.py submit`
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/submission.csv` is written and validated against `sample_submission.csv`, including exact ID values and order
- confirm binary outputs match the configured metric contract: probabilities for `roc_auc`/`log_loss`, labels for `accuracy`

Manual verification for optimization:
- run `uv run python main.py train` with `experiment.candidate.optimization.enabled: true` and at least one stopping condition
- supported tunable combinations are:
  - binary: `logistic_regression + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `hist_gradient_boosting + frequency`, `lightgbm + ordinal`, `lightgbm + frequency`, `catboost + native`, `xgboost + ordinal`, `xgboost + frequency`
  - regression: `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `hist_gradient_boosting + frequency`, `lightgbm + ordinal`, `lightgbm + frequency`, `catboost + native`, `xgboost + ordinal`, `xgboost + frequency`
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/optimization_summary.json` is written
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/optimization_trials.csv` records trial state, score, and params
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/candidate.json` records `tuning_provenance`

Manual verification for blend candidates:
- ensure at least two compatible base candidates already exist under `artifacts/<competition_slug>/candidates/`
- configure `experiment.candidate.candidate_type: blend` with `base_candidate_ids` and optional `weights`
- run `uv run python main.py train`
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/candidate.json` is written with `candidate_type: blend` and component provenance
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/blend_summary.csv` records component candidate IDs, weights, component CV scores, and OOF correlation hints
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/test_predictions.csv` is written without retraining the base candidates
- for binary `roc_auc` and `log_loss` blends, confirm mismatched base-candidate label contracts fail before predictions are averaged
- run `uv run python main.py submit`
- confirm `artifacts/<competition_slug>/candidates/<candidate_id>/submission.csv` is written and validated against `sample_submission.csv`

## Outputs
- Competition data: `data/<competition_slug>/`
- Competition context artifacts: `artifacts/<competition_slug>/`
  - `competition.json`
  - `folds.csv`
- EDA reports: `reports/<competition_slug>/`
- Candidate artifacts: `artifacts/<competition_slug>/candidates/<candidate_id>/`
  - includes `candidate.json`, `fold_metrics.csv`, `oof_predictions.csv`, `test_predictions.csv`, and `submission.csv` when prepared or submitted
  - model candidates record the selected `feature_recipe_id`, engineered `feature_columns`, and optional `tuning_provenance`
  - blend candidates also include `blend_summary.csv` and record their component candidates plus normalized weights in `candidate.json`
  - optimized model candidates also include `optimization_summary.json`, `optimization_trials.csv`, and `optimization_best_params.json`
- Submission ledger: `artifacts/<competition_slug>/submissions.csv` as an append-only submission event table keyed by `candidate_id`
- Optional remote MLflow artifacts:
  - `prepare` uploads `competition.json`, `folds.csv`, and reports
  - `train` uploads the full candidate artifact directory
  - `submit` uploads `submission.csv` plus submission metadata

## Current Assumptions
- Kaggle CLI is installed, authenticated, and has access to the configured competition.
- Competition zip contents include `train.csv`, `test.csv`, and `sample_submission.csv`.
- The competition follows a simple two-column Playground submission contract: `sample_submission.csv` must be exactly `[id_column, label_column]`.
- The resolved `id_column` is identifier metadata and is excluded from preprocessing and model fitting by default.
- `config.yaml` must use top-level `competition` and `experiment` sections; the old flat layout is unsupported.
- The current runtime supports `experiment.candidate.candidate_type: model` and `experiment.candidate.candidate_type: blend`.
- Model candidates resolve `experiment.candidate.model_family + experiment.candidate.preprocessor` to one canonical internal `model_id`; candidate artifacts record that resolved `model_id` and `preprocessing_scheme_id`.
- Model candidates default `experiment.candidate.feature_recipe_id` to `identity`; feature recipes are tracked Python modules rather than runtime scripts or YAML transform blocks.
- Blend candidates consume existing compatible candidate artifacts and write one new blend candidate artifact without retraining the base candidates.
- `prepare` is the competition-level source of truth for `competition.json` and `folds.csv`; `train` consumes that frozen context and auto-runs `prepare` only when it is missing.
- Enabled optimization is part of `train`, applies to model candidates only, and writes candidate-local metadata alongside the tuned artifact with `tuning_provenance`.
- Feature recipes are deterministic and leakage-safe; fold-learned transforms still belong in preprocessing, not in the recipe layer.
- Submission uses `candidate.json` as the schema/task source of truth.
- Submission defaults to `config.candidate_id`; `submit --candidate-id <candidate_id>` overrides that selection explicitly.
- Submission metadata includes the selected `model_id`; current candidate artifacts contain exactly one `model_id`.
- Submission validation requires the selected candidate artifact `test_predictions.csv[id_column]` to match `sample_submission.csv[id_column]` exactly in both values and row order.
- Submission requires the current candidate artifact layout under `artifacts/<competition_slug>/candidates/<candidate_id>/`.
- Binary classification supports any two-class labels accepted by scikit-learn.
- For binary `roc_auc` and `log_loss`, prediction artifacts use probabilities aligned to the resolved positive class.
- For binary `accuracy`, prediction artifacts use class labels from the observed binary label set.
- Binary classification requires an explicit positive-class contract. If `positive_label` is omitted, the workflow only auto-resolves the positive class for labels `[0, 1]`, `[False, True]`, or `["No", "Yes"]`; other two-class label pairs fail fast.
- `task_type` and `primary_metric` are explicitly configured for every run.
- Runtime config comes from local repository-root `config.yaml` only; tracked example files are just starting points, and there are no CLI or environment overrides.
- When `experiment.tracking.enabled=true`, MLflow publishing becomes part of stage success; local files are still written first, then uploaded after successful completion.
- Tuning search spaces live in code next to model definitions; there is no YAML search-space DSL.
- The current workflow is CPU-first and optimized for iteration speed over production hardening.
