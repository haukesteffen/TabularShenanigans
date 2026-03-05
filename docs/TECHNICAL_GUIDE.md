# Technical Playbook

Implementation playbook for this repository. Use this file as the technical source of truth.

## Current Phase
Step 5: Baseline modeling with CV.

In scope now:
- `config.yaml`
- `src/tabular_shenanigans/__init__.py`
- `src/tabular_shenanigans/config.py`
- Minimal callable entrypoint (`main.py` or `src/tabular_shenanigans/cli.py`)
- Kaggle data download module (`src/tabular_shenanigans/data.py`)
- EDA module (`src/tabular_shenanigans/eda.py`)
- Preprocessing module (`src/tabular_shenanigans/preprocess.py`)
- CV module (`src/tabular_shenanigans/cv.py`)
- Training module (`src/tabular_shenanigans/train.py`)
- Local data target path: `data/<competition_slug>/`
- EDA report path: `reports/<competition_slug>/`
- Preprocessing artifact path: `artifacts/<competition_slug>/preprocess/`
- Training artifact path: `artifacts/<competition_slug>/train/<run_id>/`
- Training ledger path: `artifacts/<competition_slug>/train/runs.csv`

Out of scope now:
- `src/tabular_shenanigans/pipeline.py`
- Kaggle submission integration
- Plot generation and notebook-first workflows
- Model stacking

## Current Phase Rules (Functionality First)
- Keep implementation simple.
- Absolutely no overengineering.
- One runtime config source only: a single `config.yaml`.
- No config overrides via CLI or environment variables.
- No multi-file config composition.
- Assume Kaggle CLI is available and authenticated.
- Assume secrets are valid and user has joined the configured competition.
- Keep EDA numeric/script-driven (terminal summary + CSV reports), no plots.
- Keep preprocessing output CSV-based for this MVP iteration.
- Focus on working functionality over code quality conventions and polish.
- Unit and integration tests are out of scope in this phase.
- Avoid broad defensive `try/except` blocks; prefer direct failures during development unless minimal handling is required for core flow.
- Keep validation and error messaging minimal: only what is needed to unblock usage and debugging.
- Deterministic behavior is preferred when easy to keep, but iteration speed takes priority in this phase.
- Ship small focused iterations.
- After each iteration, provide a detailed explanation of changes and behavior.

## Future Scalability Guardrails (CPU -> Cloud GPU)
These are design guardrails for upcoming phases. They do not expand current Step 4 scope.

- Stay CPU-first by default; treat GPU support as an optional backend path to add later.
- Keep data and modeling logic behind small internal interfaces so backend swaps are localized.
- Avoid scattering direct pandas/sklearn calls across many modules.
- Prefer columnar/vectorized transformations over row-wise Python loops and heavy `DataFrame.apply` usage.
- Use portable artifacts (`parquet`, `csv`, `json`) for interoperability across local and cloud environments.
- Keep runs stateless and reproducible from config so jobs can move from local to cloud workers.
- Plan dependencies as core CPU requirements plus optional GPU extras, not mandatory RAPIDS install.
- When GPU backend is introduced, require CPU/GPU parity checks with explicit numeric tolerance.

## Build Order (Step 5)
1. Ensure data fetch (Step 2) and EDA (Step 3) run before preprocessing.
2. Run preprocessing and persist CPU-friendly CSV artifacts.
3. Resolve competition `task_type` and `primary_metric` from config/Kaggle metadata.
4. Build deterministic CV splitter:
   - Regression: 7-fold shuffled `KFold`
   - Binary classification: 7-fold shuffled `StratifiedKFold`
5. Train baseline linear model per fold with fold-local preprocessing:
   - Regression: `ElasticNet`
   - Binary classification: `LogisticRegression`
6. Write fold metrics, CV summary, OOF predictions, test predictions, and append run ledger.

Preprocessing implementation details:
   - Numeric: median imputation + `StandardScaler`
   - Categorical: most-frequent imputation + `OneHotEncoder`

## Interfaces (Current Phase)
Input:
- One config file: `config.yaml` (single source of truth)
- Required key: `competition_slug`
- Optional keys for competition metadata:
  - `task_type` (`regression` or `binary`)
  - `primary_metric` (`rmse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`)
- Optional keys for preprocessing:
  - `force_categorical` (list of column names)
  - `force_numeric` (list of column names)
  - `drop_columns` (list of column names)
  - `low_cardinality_int_threshold` (positive integer)
- Optional keys for CV:
  - `cv_n_splits` (integer >= 2, default 7)
  - `cv_shuffle` (boolean, default true)
  - `cv_random_state` (integer, default 42)

Output:
- A validated in-memory config object from Pydantic
- Competition files downloaded under `data/<competition_slug>/`
- EDA summary printed to terminal
- EDA report CSV files under `reports/<competition_slug>/`
- Preprocessed feature/target CSV files under `artifacts/<competition_slug>/preprocess/`
- Training artifacts under `artifacts/<competition_slug>/train/<run_id>/`:
  - `fold_metrics.csv`
  - `cv_summary.csv`
  - `oof_predictions.csv`
  - `test_predictions.csv`
  - `run_manifest.json`
- Append-only training ledger at `artifacts/<competition_slug>/train/runs.csv`

Error contract:
- Missing config file -> hard error
- Schema/type violation -> hard error
- Any attempt to use additional config sources -> hard error
- Kaggle command failure -> hard error (bubble up with minimal wrapping)
- Missing exact competition metadata match for slug -> hard error
- Unknown/unsupported configured `primary_metric` -> hard error
- Partial task/metric inference from Kaggle metadata -> hard error
- Invalid task/metric pairing (for example `binary` + `rmse`) -> hard error
- Missing/invalid competition zip contents -> hard error
- Target inference not exactly one column -> hard error
- Unknown columns in `force_categorical`, `force_numeric`, or `drop_columns` -> hard error
- Any overlap between `force_categorical` and `force_numeric` -> hard error
- No feature columns remaining after `drop_columns` -> hard error
- Preprocessing fit/transform failure -> hard error
- Unsupported task type for CV/model selection -> hard error
- Unsupported metric for chosen task -> hard error
- Any CV/training fit or scoring failure -> hard error
- Fold assignment gaps in OOF generation -> hard error

## Validation And Error Contract
- Validation layer: Pydantic for config only, with minimal scope needed for current functionality.
- Runtime assumptions: Kaggle CLI/auth/competition access are preconfigured by the user.
- Runtime assumptions: Kaggle zip contains `train.csv` and `test.csv`.
- Runtime assumptions: transformed feature matrices are small enough to persist as CSV in this MVP phase.
- Error timing: fail fast; avoid extra preflight checks.
- Error style: simple and direct; detailed/production-grade messaging is deferred.

## Next Phases (Preview Only)
- Expand baseline model bundle beyond linear defaults.
- Kaggle workflow: slug-based storage, fetch-if-missing zip files, compact submission messages.
- Model stacking for stronger CV/LB performance.

## Iteration Reporting Template
For every shipped iteration, provide:
- Changed files
- What behavior changed
- Why the change was made
- How to run it
- Expected outputs and errors
- Current limitations
- Which shortcuts were intentionally taken in this phase (for example: no tests, minimal error handling)
- Next immediate step
