# Technical Playbook

Implementation playbook for this repository. Use this file as the technical source of truth.

## Current Phase
Step 3: Exploratory data analysis.

In scope now:
- `config.yaml`
- `src/tabular_shenanigans/__init__.py`
- `src/tabular_shenanigans/config.py`
- Minimal callable entrypoint (`main.py` or `src/tabular_shenanigans/cli.py`)
- Kaggle data download module (`src/tabular_shenanigans/data.py`)
- EDA module (`src/tabular_shenanigans/eda.py`)
- Local data target path: `data/<competition_slug>/`
- EDA report path: `reports/<competition_slug>/`

Out of scope now:
- `src/tabular_shenanigans/pipeline.py`
- `src/tabular_shenanigans/cv.py`
- `src/tabular_shenanigans/train.py`
- `artifacts/` (except optional placeholder)
- Kaggle submission integration
- Plot generation and notebook-first workflows
- Feature engineering, baseline models, stacking

## Current Phase Rules (Functionality First)
- Keep implementation simple.
- Absolutely no overengineering.
- One runtime config source only: a single `config.yaml`.
- No config overrides via CLI or environment variables.
- No multi-file config composition.
- Assume Kaggle CLI is available and authenticated.
- Assume secrets are valid and user has joined the configured competition.
- Keep EDA numeric/script-driven (terminal summary + CSV reports), no plots.
- Focus on working functionality over code quality conventions and polish.
- Unit and integration tests are out of scope in this phase.
- Avoid broad defensive `try/except` blocks; prefer direct failures during development unless minimal handling is required for core flow.
- Keep validation and error messaging minimal: only what is needed to unblock usage and debugging.
- Deterministic behavior is preferred when easy to keep, but iteration speed takes priority in this phase.
- Ship small focused iterations.
- After each iteration, provide a detailed explanation of changes and behavior.

## Future Scalability Guardrails (CPU -> Cloud GPU)
These are design guardrails for upcoming phases. They do not expand current Step 3 scope.

- Stay CPU-first by default; treat GPU support as an optional backend path to add later.
- Keep data and modeling logic behind small internal interfaces so backend swaps are localized.
- Avoid scattering direct pandas/sklearn calls across many modules.
- Prefer columnar/vectorized transformations over row-wise Python loops and heavy `DataFrame.apply` usage.
- Use portable artifacts (`parquet`, `csv`, `json`) for interoperability across local and cloud environments.
- Keep runs stateless and reproducible from config so jobs can move from local to cloud workers.
- Plan dependencies as core CPU requirements plus optional GPU extras, not mandatory RAPIDS install.
- When GPU backend is introduced, require CPU/GPU parity checks with explicit numeric tolerance.

## Build Order (Step 3)
1. Ensure data fetch (Step 2) runs before analysis.
2. Add EDA function to read `train.csv` and `test.csv` from competition zip.
3. Infer target column from train/test column difference.
4. Write numeric report CSV files to `reports/<competition_slug>/`.
5. Wire entrypoint to call config loader, data fetch, then EDA.

## Interfaces (Current Phase)
Input:
- One config file: `config.yaml` (single source of truth)

Output:
- A validated in-memory config object from Pydantic
- Competition files downloaded under `data/<competition_slug>/`
- EDA summary printed to terminal
- EDA report CSV files under `reports/<competition_slug>/`

Error contract:
- Missing config file -> hard error
- Schema/type violation -> hard error
- Any attempt to use additional config sources -> hard error
- Kaggle command failure -> hard error (bubble up with minimal wrapping)
- Missing/invalid competition zip contents -> hard error
- Target inference not exactly one column -> hard error

## Validation And Error Contract
- Validation layer: Pydantic for config only, with minimal scope needed for current functionality.
- Runtime assumptions: Kaggle CLI/auth/competition access are preconfigured by the user.
- Runtime assumptions: Kaggle zip contains `train.csv` and `test.csv`.
- Error timing: fail fast; avoid extra preflight checks.
- Error style: simple and direct; detailed/production-grade messaging is deferred.

## Next Phases (Preview Only)
- CV strategy: 7-fold shuffled (`KFold` for regression, `StratifiedKFold` for binary classification).
- Baseline modeling: sklearn-native bundle.
- Kaggle workflow: slug-based storage, fetch-if-missing zip files, compact submission messages.
- Run tracking: local append-only CSV with unique `run_id`.

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
