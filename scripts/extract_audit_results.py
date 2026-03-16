"""
Extract GPU-native vs CPU parity audit results from MLflow.

Usage:
    uv run python scripts/extract_audit_results.py

Reads config.yaml for the tracking URI and competition slug, then queries
MLflow for all audit candidate runs (fr0, optimization disabled) and prints
a comparison table grouped by model_family + preprocessors across backends.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env"), override=False)

from tabular_shenanigans.config import load_config  # noqa: E402

AUDIT_CANDIDATE_MATRIX = [
    ("logistic_regression", "standardize", "frequency"),
    ("logistic_regression", "standardize", "onehot"),
    ("logistic_regression", "standardize", "ordinal"),
    ("random_forest",       "median",      "frequency"),
    ("random_forest",       "median",      "onehot"),
    ("lightgbm",            "median",      "frequency"),
    ("lightgbm",            "median",      "ordinal"),
    ("xgboost",             "median",      "frequency"),
    ("xgboost",             "median",      "onehot"),
    ("catboost",            "median",      "native"),
]


def _load_mlflow(tracking_uri: str):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def _fetch_audit_runs(mlflow, experiment_id: str) -> list[dict]:
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.run_kind = 'candidate' and params.feature_recipe_id = 'fr0'",
        max_results=500,
    )
    results = []
    for run in runs:
        p = run.data.params
        m = run.data.metrics
        results.append({
            "candidate_id": run.data.tags.get("candidate_id", ""),
            "model_family": p.get("model_family", ""),
            "numeric_preprocessor": p.get("numeric_preprocessor", ""),
            "categorical_preprocessor": p.get("categorical_preprocessor", ""),
            "backend": p.get("runtime__acceleration_backend", p.get("runtime__resolved_gpu_backend", "?")),
            "preprocessing_backend": p.get("runtime__preprocessing_backend", "?"),
            "cv_score_mean": m.get("cv_score_mean"),
            "cv_score_std": m.get("cv_score_std"),
            "fit_wall_seconds": m.get("fit_wall_seconds"),
            "cv_preprocess_wall_seconds": m.get("cv_preprocess_wall_seconds"),
            "cv_fit_wall_seconds": m.get("cv_fit_wall_seconds"),
            "cv_predict_wall_seconds": m.get("cv_predict_wall_seconds"),
        })
    return results


def _find_run(runs: list[dict], model_family: str, numeric: str, categorical: str, backend: str) -> dict | None:
    for r in runs:
        if (
            r["model_family"] == model_family
            and r["numeric_preprocessor"] == numeric
            and r["categorical_preprocessor"] == categorical
            and r["backend"] == backend
        ):
            return r
    return None


def _fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _speedup(cpu_val: float | None, gpu_val: float | None) -> str:
    if cpu_val is None or gpu_val is None or gpu_val == 0:
        return "—"
    return f"{cpu_val / gpu_val:.2f}x"


def main() -> None:
    config = load_config()
    mlflow = _load_mlflow(config.experiment.tracking.tracking_uri)
    experiment = mlflow.set_experiment(config.competition.slug)
    runs = _fetch_audit_runs(mlflow, experiment.experiment_id)

    print(f"\nAudit runs found: {len(runs)}")
    print(f"Competition: {config.competition.slug}  |  Metric: {config.competition.primary_metric}\n")

    header = (
        f"{'Model':<22} {'Num':>12} {'Cat':>10} | "
        f"{'GPU score':>10} {'CPU score':>10} {'Δscore':>8} | "
        f"{'GPU fit(s)':>10} {'CPU fit(s)':>10} {'speedup':>8} | "
        f"{'prep_bk':>22}"
    )
    print(header)
    print("-" * len(header))

    for model_family, numeric, categorical in AUDIT_CANDIDATE_MATRIX:
        gpu = _find_run(runs, model_family, numeric, categorical, "gpu_native")
        cpu = _find_run(runs, model_family, numeric, categorical, "cpu")

        score_delta = None
        if gpu and cpu and gpu["cv_score_mean"] is not None and cpu["cv_score_mean"] is not None:
            score_delta = gpu["cv_score_mean"] - cpu["cv_score_mean"]

        prep_backend = (gpu or cpu or {}).get("preprocessing_backend", "?")

        print(
            f"{model_family:<22} {numeric:>12} {categorical:>10} | "
            f"{_fmt((gpu or {}).get('cv_score_mean')):>10} "
            f"{_fmt((cpu or {}).get('cv_score_mean')):>10} "
            f"{_fmt(score_delta, 5):>8} | "
            f"{_fmt((gpu or {}).get('fit_wall_seconds'), 1):>10} "
            f"{_fmt((cpu or {}).get('fit_wall_seconds'), 1):>10} "
            f"{_speedup((cpu or {}).get('fit_wall_seconds'), (gpu or {}).get('fit_wall_seconds')):>8} | "
            f"{prep_backend:>22}"
        )

    print()

    # Summary: any missing runs
    missing = []
    for model_family, numeric, categorical in AUDIT_CANDIDATE_MATRIX:
        for backend in ("gpu_native", "cpu"):
            r = _find_run(runs, model_family, numeric, categorical, backend)
            if r is None:
                missing.append(f"  {backend:<12} {model_family:<22} {numeric:<14} {categorical}")
    if missing:
        print("Missing runs (not yet in MLflow):")
        for m in missing:
            print(m)
    else:
        print("All audit runs present in MLflow.")


if __name__ == "__main__":
    main()
