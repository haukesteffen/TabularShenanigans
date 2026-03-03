# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Task scope: regression and binary classification.
- Development style: small, incremental iterations with detailed explanations.
- Scalability direction: CPU-first local workflow now, with a future path to cloud GPU execution (RAPIDS `cudf`/`cuml`) without rewriting the full pipeline.

## Current Development Mode (Functionality First)
- This is an active development process focused on shipping working behavior quickly.
- Engineering polish is intentionally deprioritized in this phase.
- Unit and integration tests are explicitly out of scope for now.
- Avoid broad defensive `try/except` wrapping; let failures surface during development unless handling is required to keep core flow usable.
- Refactoring for style, architecture hardening, and production-grade robustness are deferred to a later stabilization phase.

## Current MVP Status
- Step 1 (configuration pipeline) is complete.
- Step 2 (Kaggle competition data fetch) is complete.
- Step 3 (script-based exploratory data analysis) is complete.
- Current implementation priority is preprocessing for model-ready features (Step 4).

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart (Current Stage)
1. Keep a project `config.yaml` at repository root.
2. Run the current Python entrypoint scripts directly from the repo.
3. Ensure Kaggle CLI access is already configured for your user.
4. Current run behavior: fetch competition zip if missing, generate EDA report CSVs, then write preprocessed train/test feature CSVs under `artifacts/<competition_slug>/preprocess/`.
5. Follow iteration notes and current development-mode rules in [`docs/TECHNICAL_GUIDE.md`](docs/TECHNICAL_GUIDE.md).

## Roadmap
1. Robust config pipeline
2. Kaggle data fetch
3. Exploratory data analysis
4. Preprocessing and feature engineering
5. Baseline models
6. Model stacking for stronger CV/LB performance
