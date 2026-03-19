"""Generate the 228-entry candidate YAML for the s6e3 binary classification sweep.

Usage:
    uv run python scripts/generate_sweep_candidates.py
    uv run python scripts/generate_sweep_candidates.py >> config.yaml  # append to existing
"""

import sys
import yaml

from tabular_shenanigans.models import MODEL_REGISTRY
from tabular_shenanigans.runtime_execution import CPU_GPU_BACKEND

FEATURE_RECIPES = ["fr0", "fr1", "fr2", "fr3"]

NUMERIC_OPTIONS = ["median", "standardize", "kbins"]
CATEGORICAL_OPTIONS = ["onehot", "ordinal", "frequency"]

def build_candidate(recipe, model, numeric, categorical):
    return {
        "candidate_type": "model",
        "feature_recipe_id": recipe,
        "model_family": model,
        "numeric_preprocessor": numeric,
        "categorical_preprocessor": categorical,
    }


def get_binary_model_groups():
    gpu_models = []
    cpu_models = []
    native_categorical_models = []

    for model_id, model_definition in MODEL_REGISTRY["binary"].items():
        if model_definition.supports_native_categorical_preprocessing:
            native_categorical_models.append(model_id)
            continue
        gpu_backends = {
            backend
            for gpu_routing_rule in model_definition.gpu_routing_rules
            for backend in gpu_routing_rule.gpu_backends
        }
        has_accelerated_gpu_path = any(backend != CPU_GPU_BACKEND for backend in gpu_backends)
        if has_accelerated_gpu_path:
            gpu_models.append(model_id)
            continue
        cpu_models.append(model_id)

    return gpu_models, cpu_models, native_categorical_models


def generate_candidates():
    gpu_models, cpu_models, native_categorical_models = get_binary_model_groups()
    candidates = []
    for recipe in FEATURE_RECIPES:
        for model in gpu_models + cpu_models:
            for numeric in NUMERIC_OPTIONS:
                for categorical in CATEGORICAL_OPTIONS:
                    candidates.append(build_candidate(recipe, model, numeric, categorical))
        for model in native_categorical_models:
            for numeric in NUMERIC_OPTIONS:
                candidates.append(build_candidate(recipe, model, numeric, "native"))
    return candidates


def main():
    candidates = generate_candidates()
    print(f"# Generated {len(candidates)} candidates", file=sys.stderr)
    print(yaml.dump({"candidates": candidates}, default_flow_style=False, sort_keys=False), end="")


if __name__ == "__main__":
    main()
