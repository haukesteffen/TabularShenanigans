"""Caruana greedy ensemble selection on OOF predictions.

Queries all FINISHED model candidate runs from the configured MLflow experiment,
filters to those whose CV config matches the current config.yaml, downloads their
oof_predictions.csv artifacts, then iteratively adds the candidate (with replacement)
that most improves blended OOF score — Caruana-style greedy selection.

Uses score_predictions from cv.py with the configured primary_metric, so the
optimizer works correctly for both regression and binary classification.

OOF data is cached to .oof_cache/ in the repo root after the first download so
reruns skip the network entirely.

Prints an optimization trace and emits a ready-to-paste config.yaml snippet.

Usage:
    uv run python scripts/select_greedy_ensemble.py [--n-rounds 100]
    uv run python scripts/select_greedy_ensemble.py --refresh-cache   # re-download all OOF data
"""

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlflow
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from tabular_shenanigans.cv import is_higher_better, resolve_positive_label, score_predictions


CONFIG_PATH = REPO_ROOT / "config.yaml"
OOF_CACHE_DIR = REPO_ROOT / ".oof_cache"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def fetch_finished_model_runs(client, experiment_id, cv_n_splits, cv_shuffle, cv_random_state):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            "tags.run_kind = 'candidate' "
            "and tags.candidate_type = 'model' "
            "and attributes.status = 'FINISHED'"
        ),
        max_results=10000,
    )

    matching = []
    mismatched = 0
    for run in runs:
        params = run.data.params
        if (
            str(params.get("cv__n_splits")) == str(cv_n_splits)
            and str(params.get("cv__shuffle")) == str(cv_shuffle)
            and str(params.get("cv__random_state")) == str(cv_random_state)
        ):
            matching.append(run)
        else:
            mismatched += 1

    return matching, mismatched


def cache_path_for(run_id):
    return OOF_CACHE_DIR / f"{run_id}.npz"


def load_from_cache(run_id):
    p = cache_path_for(run_id)
    if not p.exists():
        return None, None
    data = np.load(p, allow_pickle=True)
    return data["y_pred"], data["y_true"]


def save_to_cache(run_id, y_pred, y_true):
    OOF_CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache_path_for(run_id), y_pred=y_pred, y_true=y_true)


def download_oof(client, run_id, dst_dir):
    path = client.download_artifacts(run_id, "candidate/oof_predictions.csv", dst_dir)
    df = pd.read_csv(path)
    df = df.sort_values("row_idx").reset_index(drop=True)
    return df["y_pred"].to_numpy(dtype=float), df["y_true"].to_numpy()


def load_oof(client, run_id, tmp_dir, refresh_cache):
    if not refresh_cache:
        y_pred, y_true = load_from_cache(run_id)
        if y_pred is not None:
            return y_pred, y_true, "cache"

    y_pred, y_true = download_oof(client, run_id, tmp_dir)
    save_to_cache(run_id, y_pred, y_true)
    return y_pred, y_true, "download"


def build_fast_scorer(task_type, primary_metric, y_true_series, positive_label):
    """Return a fast (y_pred: np.ndarray) -> float scorer.

    Precomputes y_true in the right form once so the inner greedy loop avoids
    repeated pd.Series operations and resolve_binary_label_pair calls.
    """
    import math
    from sklearn.metrics import (
        accuracy_score, log_loss, mean_absolute_error,
        mean_squared_error, mean_squared_log_error, roc_auc_score,
    )

    if task_type == "regression":
        y = y_true_series.to_numpy()
        if primary_metric == "rmse":
            return lambda p: float(math.sqrt(mean_squared_error(y, p)))
        if primary_metric == "mse":
            return lambda p: float(mean_squared_error(y, p))
        if primary_metric == "rmsle":
            return lambda p: float(math.sqrt(mean_squared_log_error(y, np.clip(p, 0, None))))
        if primary_metric == "mae":
            return lambda p: float(mean_absolute_error(y, p))
        raise ValueError(f"Unsupported regression metric: {primary_metric}")

    if task_type == "binary":
        y_binary = (y_true_series == positive_label).astype(int).to_numpy()
        neg_label, pos_label = (
            (1, 0) if positive_label == 0 else (0, positive_label)  # fallback for non-int
        )
        if primary_metric == "roc_auc":
            return lambda p: float(roc_auc_score(y_binary, p))
        if primary_metric == "log_loss":
            return lambda p: float(log_loss(y_binary, p))
        if primary_metric == "accuracy":
            actual = y_true_series.to_numpy()
            return lambda p: float(accuracy_score(actual, np.where(p >= 0.5, positive_label, neg_label)))
        raise ValueError(f"Unsupported binary metric: {primary_metric}")

    raise ValueError(f"Unsupported task_type: {task_type}")


def greedy_select(oof_matrix, y_true_series, task_type, primary_metric, positive_label, n_rounds, higher_better):
    """Caruana greedy ensemble selection.

    oof_matrix: shape (n_candidates, n_rows)
    Iteratively adds the candidate (with replacement) that most improves blended OOF score.
    Returns weights (sums to 1) and final score.
    """
    n = oof_matrix.shape[0]
    counts = np.zeros(n, dtype=int)
    fast_score = build_fast_scorer(task_type, primary_metric, y_true_series, positive_label)

    # Seed: pick the best single candidate (n individual score_predictions calls — one-time cost)
    single_scores = [
        score_predictions(task_type, primary_metric, y_true_series, oof_matrix[i], positive_label)
        for i in range(n)
    ]
    best_seed = int(np.argmax(single_scores) if higher_better else np.argmin(single_scores))
    counts[best_seed] = 1
    current_score = single_scores[best_seed]

    print(f"\nOptimization trace ({n_rounds} rounds, metric={primary_metric}):")
    print(f"  seed: best single = index {best_seed}, {primary_metric}={current_score:.6f}")

    bar = tqdm(range(n_rounds), desc="Greedy select", unit="round", file=sys.stdout)
    rounds_done = 0
    for round_num in bar:
        # Maintain running sum to avoid recomputing full weighted average per candidate.
        # current_sum = oof_matrix.T @ counts (unnormalized).
        # Trying candidate i: blend = (current_sum + oof_matrix[i]) / (total + 1)
        total = int(counts.sum())
        current_sum = oof_matrix.T @ counts  # shape (n_rows,)

        best_gain = -np.inf if higher_better else np.inf
        best_i = None

        for i in range(n):
            trial_blend = (current_sum + oof_matrix[i]) / (total + 1)
            trial_score = fast_score(trial_blend)
            gain = trial_score - current_score if higher_better else current_score - trial_score
            if gain > best_gain:
                best_gain = gain
                best_i = i

        counts[best_i] += 1
        current_score = fast_score((current_sum + oof_matrix[best_i]) / (total + 1))
        bar.set_postfix({primary_metric: f"{current_score:.6f}"})
        tqdm.write(f"  round {round_num + 1:3d}: added index {best_i:3d}, gain={best_gain:+.2e}, {primary_metric}={current_score:.6f}")
        rounds_done += 1

        # Early stop: no candidate improved the ensemble this round.
        if best_gain <= 0:
            bar.close()
            print(f"  early stop: no improvement in round {round_num + 1}.")
            break

    print(f"  done after {rounds_done} rounds. Final {primary_metric} = {current_score:.6f}")
    weights = counts / counts.sum()
    return weights, current_score


def print_component_table(candidate_ids, cv_scores, weights, primary_metric):
    nonzero = [(cid, cv, w) for cid, cv, w in zip(candidate_ids, cv_scores, weights) if w > 0]
    nonzero.sort(key=lambda x: -x[2])
    print(f"\nComponents ({len(nonzero)} of {len(candidate_ids)} have non-zero weight):")
    print(f"  {'rank':<6} {'candidate_id':<50} {f'cv_{primary_metric}':>14}  {'weight':>8}")
    for rank, (cid, cv, w) in enumerate(nonzero, 1):
        print(f"  {rank:<6} {cid:<50} {cv:>14.4f}  {w:>8.4f}")


def emit_config_snippet(candidate_ids, weights):
    nonzero = [(cid, w) for cid, w in zip(candidate_ids, weights) if w > 0]
    nonzero.sort(key=lambda x: -x[1])
    ids = [cid for cid, _ in nonzero]
    ws = [round(float(w), 6) for _, w in nonzero]

    print("\n--- paste into config.yaml candidates: ---")
    print("- candidate_type: blend")
    print("  base_candidate_ids:")
    for cid in ids:
        print(f"    - {cid}")
    print("  weights:")
    for w in ws:
        print(f"    - {w}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rounds", type=int, default=100, help="Number of greedy selection rounds (default: 100)")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-download all OOF data even if cached")
    args = parser.parse_args()

    cfg = load_config()
    tracking_uri = cfg["experiment"]["tracking"]["tracking_uri"]
    competition_slug = cfg["competition"]["slug"]
    task_type = cfg["competition"]["task_type"]
    primary_metric = cfg["competition"]["primary_metric"]
    configured_positive_label = cfg["competition"].get("positive_label")
    cv_cfg = cfg["competition"]["cv"]
    cv_n_splits = cv_cfg["n_splits"]
    cv_shuffle = cv_cfg["shuffle"]
    cv_random_state = cv_cfg["random_state"]
    higher = is_higher_better(primary_metric)

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(competition_slug)
    if experiment is None:
        print(f"ERROR: experiment '{competition_slug}' not found in MLflow at {tracking_uri}")
        sys.exit(1)

    print(f"MLflow: {tracking_uri}")
    print(f"Experiment: {competition_slug}")
    print(f"Task: {task_type}, metric: {primary_metric} ({'higher' if higher else 'lower'} is better)")
    print(f"CV filter: n_splits={cv_n_splits}, shuffle={cv_shuffle}, random_state={cv_random_state}")
    print(f"OOF cache: {OOF_CACHE_DIR}")

    runs, mismatched = fetch_finished_model_runs(
        client, experiment.experiment_id, cv_n_splits, cv_shuffle, cv_random_state
    )
    total = len(runs) + mismatched
    print(f"\nFound {total} FINISHED model candidates. {len(runs)} match current CV config ({mismatched} skipped).")

    if not runs:
        print("No matching candidates. Exiting.")
        sys.exit(1)

    runs.sort(key=lambda r: float(r.data.metrics.get("cv_score_mean", 0.0)), reverse=True)

    print(f"Loading OOF predictions for {len(runs)} candidates...")
    candidate_ids = []
    cv_scores = []
    oof_arrays = []
    y_true_ref = None
    n_cached = 0
    n_downloaded = 0

    with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-oof-") as tmp:
        for run in tqdm(runs, desc="Loading OOF", file=sys.stdout):
            cid = run.data.tags.get("candidate_id", run.info.run_id)
            cv_score = float(run.data.metrics.get("cv_score_mean", 0.0))
            try:
                y_pred, y_true, source = load_oof(client, run.info.run_id, tmp, args.refresh_cache)
            except Exception as e:
                tqdm.write(f"  SKIP {cid}: {e}")
                continue

            if source == "cache":
                n_cached += 1
            else:
                n_downloaded += 1

            if y_true_ref is None:
                y_true_ref = y_true
            elif not np.array_equal(y_true, y_true_ref):
                tqdm.write(f"  SKIP {cid}: y_true mismatch")
                continue

            candidate_ids.append(cid)
            cv_scores.append(cv_score)
            oof_arrays.append(y_pred)

    print(f"  {len(candidate_ids)} candidates loaded (cached={n_cached}, downloaded={n_downloaded}).")

    if not candidate_ids:
        print("No usable candidates after loading. Exiting.")
        sys.exit(1)

    y_true_series = pd.Series(y_true_ref)

    # Resolve positive_label for binary tasks (handles None config by inferring from safe label pairs)
    if task_type == "binary":
        _, positive_label, _ = resolve_positive_label(y_true_series, configured_positive_label)
    else:
        positive_label = None

    oof_matrix = np.vstack(oof_arrays)  # (n_candidates, n_rows)

    weights, final_score = greedy_select(
        oof_matrix, y_true_series, task_type, primary_metric, positive_label,
        n_rounds=args.n_rounds, higher_better=higher,
    )

    print_component_table(candidate_ids, cv_scores, weights, primary_metric)
    emit_config_snippet(candidate_ids, weights)


if __name__ == "__main__":
    main()
