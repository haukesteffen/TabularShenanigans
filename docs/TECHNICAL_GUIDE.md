# Technical Playbook

Implementation playbook for this repository. Use this file as the technical source of truth.

## Current Phase
Step 1: Config only.

In scope now:
- `config.yaml`
- `src/tabular_shenanigans/__init__.py`
- `src/tabular_shenanigans/config.py`
- Minimal callable entrypoint (`main.py` or `src/tabular_shenanigans/cli.py`) to load config, validate, and fail fast

Out of scope now:
- `src/tabular_shenanigans/pipeline.py`
- `src/tabular_shenanigans/data.py`
- `src/tabular_shenanigans/cv.py`
- `src/tabular_shenanigans/train.py`
- `artifacts/` (except optional placeholder)
- Kaggle download/submission integration
- EDA, feature engineering, baseline models, stacking

## Non-Negotiable Rules
- Keep implementation simple.
- Absolutely no overengineering.
- One runtime config source only: a single `config.yaml`.
- No config overrides via CLI or environment variables.
- No multi-file config composition.
- Fail fast with clear error messages on invalid or conflicting inputs.
- Use deterministic behavior by default (fixed random seed in config).
- Ship small focused iterations.
- After each iteration, provide a detailed explanation of changes and behavior.

## Future Scalability Guardrails (CPU -> Cloud GPU)
These are design guardrails for upcoming phases. They do not expand current Step 1 scope.

- Stay CPU-first by default; treat GPU support as an optional backend path to add later.
- Keep data and modeling logic behind small internal interfaces so backend swaps are localized.
- Avoid scattering direct pandas/sklearn calls across many modules.
- Prefer columnar/vectorized transformations over row-wise Python loops and heavy `DataFrame.apply` usage.
- Use portable artifacts (`parquet`, `csv`, `json`) for interoperability across local and cloud environments.
- Keep runs stateless and reproducible from config so jobs can move from local to cloud workers.
- Plan dependencies as core CPU requirements plus optional GPU extras, not mandatory RAPIDS install.
- When GPU backend is introduced, require CPU/GPU parity checks with explicit numeric tolerance.

## Build Order (Step 1)
1. Implement YAML loading.
2. Define Pydantic config models and validation rules.
3. Implement config loading/validation entry function in `config.py`.
4. Wire minimal entrypoint to call loader and surface errors clearly.

## Interfaces (Current Phase)
Input:
- One config file: `config.yaml` (single source of truth)

Output:
- A validated in-memory config object from Pydantic

Error contract:
- Missing config file -> hard error
- Schema/type violation -> hard error
- Any attempt to use additional config sources -> hard error

## Validation And Error Contract
- Validation layer: Pydantic.
- Error timing: startup, before any pipeline side effects.
- Error style: clear, actionable, and specific to the failing field/source.

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
- Next immediate step
