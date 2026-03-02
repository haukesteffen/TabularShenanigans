# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Task scope: regression and binary classification.
- Development style: small, incremental iterations with detailed explanations.
- Scalability direction: CPU-first local workflow now, with a future path to cloud GPU execution (RAPIDS `cudf`/`cuml`) without rewriting the full pipeline.

## Current MVP Status
- Current implementation priority is the configuration pipeline.
- Full Kaggle/data/modeling flow is planned in subsequent MVP steps.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart (Current Stage)
1. Keep a project `config.yaml` at repository root.
2. Run the current Python entrypoint scripts directly from the repo.
3. Follow iteration notes and technical policy in [`docs/TECHNICAL_GUIDE.md`](docs/TECHNICAL_GUIDE.md).

## Roadmap
1. Robust config pipeline
2. Kaggle data fetch
3. Exploratory data analysis
4. Preprocessing and feature engineering
5. Baseline models
6. Model stacking for stronger CV/LB performance
