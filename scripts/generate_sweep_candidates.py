"""Generate the 228-entry candidate YAML for the s6e3 binary classification sweep.

Usage:
    uv run python scripts/generate_sweep_candidates.py
    uv run python scripts/generate_sweep_candidates.py >> config.yaml  # append to existing
"""

import sys
import yaml

FEATURE_RECIPES = ["fr0", "fr1", "fr2", "fr3"]

NUMERIC_OPTIONS = ["median", "standardize", "kbins"]
CATEGORICAL_OPTIONS = ["onehot", "ordinal", "frequency"]

# Models that support all categorical options (onehot/ordinal/frequency)
GPU_MODELS = ["logistic_regression", "random_forest", "lightgbm", "xgboost"]

# CatBoost uses native categorical handling only
CATBOOST_MODELS = ["catboost"]

# CPU-only models
CPU_MODELS = ["extra_trees", "hist_gradient_boosting"]


def build_candidate(recipe, model, numeric, categorical):
    return {
        "candidate_type": "model",
        "feature_recipe_id": recipe,
        "model_family": model,
        "numeric_preprocessor": numeric,
        "categorical_preprocessor": categorical,
        "optimization": {"enabled": False},
    }


def generate_candidates():
    candidates = []
    for recipe in FEATURE_RECIPES:
        for model in GPU_MODELS + CPU_MODELS:
            for numeric in NUMERIC_OPTIONS:
                for categorical in CATEGORICAL_OPTIONS:
                    candidates.append(build_candidate(recipe, model, numeric, categorical))
        for model in CATBOOST_MODELS:
            for numeric in NUMERIC_OPTIONS:
                candidates.append(build_candidate(recipe, model, numeric, "native"))
    return candidates


def main():
    candidates = generate_candidates()
    print(f"# Generated {len(candidates)} candidates", file=sys.stderr)
    print(yaml.dump({"candidates": candidates}, default_flow_style=False, sort_keys=False), end="")


if __name__ == "__main__":
    main()
