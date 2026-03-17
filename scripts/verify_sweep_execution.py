"""Verify s6e3 binary classification sweep execution results in MLflow.

Queries all s6e3 candidate runs and checks that:
- GPU-capable models (logistic_regression, random_forest, lightgbm, xgboost, catboost)
  resolved to GPU (gpu_native)
- CPU-only models (extra_trees, hist_gradient_boosting) resolved to CPU
- Any unexpected GPU→CPU fallbacks are surfaced with their fallback_reason

Usage:
    uv run python scripts/verify_sweep_execution.py
"""

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlflow
import yaml

CONFIG_PATH = REPO_ROOT / "config.yaml"

CPU_ONLY_MODELS = {"extra_trees", "hist_gradient_boosting"}
GPU_MODELS = {"logistic_regression", "random_forest", "lightgbm", "xgboost", "catboost"}


def load_tracking_uri():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["experiment"]["tracking"]["tracking_uri"]


def fetch_candidate_runs(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"ERROR: experiment '{experiment_name}' not found in MLflow")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_kind = 'candidate'",
        max_results=10000,
    )
    return runs


def summarize(runs):
    # model_family -> {compute_target -> count}
    by_model = defaultdict(lambda: defaultdict(int))
    fallbacks = []

    for run in runs:
        params = run.data.params
        tags = run.data.tags
        model = params.get("model_family", "unknown")
        compute = params.get("runtime__resolved_compute_target", "unknown")
        backend = params.get("runtime__acceleration_backend", "unknown")
        fallback_reason = params.get("runtime__fallback_reason", None)

        by_model[model][compute] += 1

        # Flag unexpected CPU fallback: model is GPU-capable but resolved to CPU
        if model in GPU_MODELS and compute == "cpu":
            fallbacks.append(
                {
                    "run_id": run.info.run_id,
                    "model": model,
                    "recipe": params.get("feature_recipe_id", "?"),
                    "numeric": params.get("numeric_preprocessor", "?"),
                    "categorical": params.get("categorical_preprocessor", "?"),
                    "fallback_reason": fallback_reason or "not recorded",
                    "backend": backend,
                }
            )

    return by_model, fallbacks


def print_summary_table(by_model, total_runs):
    print(f"\nTotal candidate runs found: {total_runs}")
    print(f"\n{'Model':<30} {'Compute':<15} {'Count':>6}")
    print("-" * 55)
    for model in sorted(by_model):
        for compute, count in sorted(by_model[model].items()):
            tag = ""
            if model in CPU_ONLY_MODELS and compute == "cpu":
                tag = "  (expected CPU)"
            elif model in GPU_MODELS and compute == "gpu_native":
                tag = "  (expected GPU)"
            elif model in GPU_MODELS and compute == "cpu":
                tag = "  *** UNEXPECTED CPU ***"
            print(f"  {model:<28} {compute:<15} {count:>6}{tag}")
    print()


def assert_routing(by_model):
    errors = []

    for model in CPU_ONLY_MODELS:
        counts = by_model.get(model, {})
        gpu_count = counts.get("gpu_native", 0)
        if gpu_count > 0:
            errors.append(f"CPU-only model '{model}' unexpectedly resolved to GPU ({gpu_count} runs)")

    for model in GPU_MODELS:
        counts = by_model.get(model, {})
        cpu_count = counts.get("cpu", 0)
        if cpu_count > 0:
            errors.append(f"GPU model '{model}' fell back to CPU ({cpu_count} runs)")

    return errors


def main():
    tracking_uri = load_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment_name = "playground-series-s6e3"
    print(f"Querying MLflow at {tracking_uri}")
    print(f"Experiment: {experiment_name}")

    runs = fetch_candidate_runs(client, experiment_name)
    by_model, fallbacks = summarize(runs)
    print_summary_table(by_model, len(runs))

    if fallbacks:
        print(f"Unexpected GPU→CPU fallbacks ({len(fallbacks)}):")
        for f in fallbacks:
            print(
                f"  run={f['run_id'][:8]}  model={f['model']}  recipe={f['recipe']}"
                f"  numeric={f['numeric']}  categorical={f['categorical']}"
                f"  reason={f['fallback_reason']}"
            )
        print()

    errors = assert_routing(by_model)
    if errors:
        print("ASSERTION FAILURES:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("All routing assertions passed.")


if __name__ == "__main__":
    main()
