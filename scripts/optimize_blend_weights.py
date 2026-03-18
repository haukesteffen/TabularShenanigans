"""Find optimal ensemble blend weights via hill climbing on OOF predictions.

Queries all FINISHED model candidate runs from the configured MLflow experiment,
filters to those whose CV config matches the current config.yaml, downloads only
their oof_predictions.csv artifacts, then hill-climbs to find weights that
maximize blended OOF AUC.

OOF data is cached to .oof_cache/ in the repo root after the first download so
reruns skip the network entirely.

Prints an optimization trace and emits a ready-to-paste config.yaml snippet.

Usage:
    uv run python scripts/optimize_blend_weights.py [--step-size 0.05] [--threshold 1e-5]
    uv run python scripts/optimize_blend_weights.py --refresh-cache   # re-download all OOF data
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
from sklearn.metrics import roc_auc_score


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
    data = np.load(p)
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


def hill_climb(oof_matrix, y_true, step_size=0.05, threshold=1e-5):
    """Hill climb blend weights to maximize OOF AUC.

    oof_matrix: shape (n_candidates, n_rows)
    Returns weights array (sums to 1, zeros for excluded candidates).
    """
    n = oof_matrix.shape[0]
    weights = np.zeros(n, dtype=float)
    weights[0] = 1.0  # start with the best candidate (pre-sorted by cv_score_mean)

    def score(w):
        blend = np.average(oof_matrix, axis=0, weights=w)
        return roc_auc_score(y_true, blend)

    best_score = score(weights)
    print(f"\nOptimization trace:")
    print(f"  start (best single candidate): OOF AUC = {best_score:.6f}")

    round_num = 0
    while step_size >= 1e-6:
        score_before_round = best_score
        accepted = 0
        for i in range(n):
            for delta in (+step_size, -step_size):
                trial = weights.copy()
                trial[i] += delta
                trial = np.clip(trial, 0.0, None)
                total = trial.sum()
                if total == 0:
                    continue
                trial /= total
                new_score = score(trial)
                if new_score - best_score > threshold:
                    weights = trial
                    best_score = new_score
                    accepted += 1

        round_num += 1
        round_gain = best_score - score_before_round
        print(f"  round {round_num}: accepted {accepted}, gain={round_gain:.2e}, AUC={best_score:.6f}, step={step_size:.2e}")
        # halve step_size if no improvement or the round's total gain was negligible
        if accepted == 0 or round_gain < threshold:
            step_size /= 2

    print(f"  converged after {round_num} rounds. Final OOF AUC = {best_score:.6f}")
    return weights, best_score


def print_component_table(candidate_ids, cv_scores, weights):
    nonzero = [(cid, cv, w) for cid, cv, w in zip(candidate_ids, cv_scores, weights) if w > 0]
    nonzero.sort(key=lambda x: -x[2])
    print(f"\nComponents ({len(nonzero)} of {len(candidate_ids)} have non-zero weight):")
    print(f"  {'rank':<6} {'candidate_id':<50} {'cv_auc':>8}  {'weight':>8}")
    for rank, (cid, cv, w) in enumerate(nonzero, 1):
        print(f"  {rank:<6} {cid:<50} {cv:>8.4f}  {w:>8.4f}")


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
    sys.stdout.reconfigure(line_buffering=True)  # flush on every newline even when piped
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--refresh-cache", action="store_true", help="Re-download all OOF data even if cached")
    args = parser.parse_args()

    cfg = load_config()
    tracking_uri = cfg["experiment"]["tracking"]["tracking_uri"]
    competition_slug = cfg["competition"]["slug"]
    cv_cfg = cfg["competition"]["cv"]
    cv_n_splits = cv_cfg["n_splits"]
    cv_shuffle = cv_cfg["shuffle"]
    cv_random_state = cv_cfg["random_state"]

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(competition_slug)
    if experiment is None:
        print(f"ERROR: experiment '{competition_slug}' not found in MLflow at {tracking_uri}")
        sys.exit(1)

    print(f"MLflow: {tracking_uri}")
    print(f"Experiment: {competition_slug}")
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

    # Sort by cv_score_mean descending so weights[0] starts with the best candidate
    runs.sort(key=lambda r: float(r.data.metrics.get("cv_score_mean", 0.0)), reverse=True)

    print(f"Loading OOF predictions for {len(runs)} candidates...")
    candidate_ids = []
    cv_scores = []
    oof_arrays = []
    y_true_ref = None
    n_cached = 0
    n_downloaded = 0

    with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-oof-") as tmp:
        for i, run in enumerate(runs):
            cid = run.data.tags.get("candidate_id", run.info.run_id)
            cv_score = float(run.data.metrics.get("cv_score_mean", 0.0))
            try:
                y_pred, y_true, source = load_oof(client, run.info.run_id, tmp, args.refresh_cache)
            except Exception as e:
                print(f"  SKIP {cid}: {e}")
                continue

            if source == "cache":
                n_cached += 1
            else:
                n_downloaded += 1

            if y_true_ref is None:
                y_true_ref = y_true
            elif not np.array_equal(y_true, y_true_ref):
                print(f"  SKIP {cid}: y_true mismatch")
                continue

            candidate_ids.append(cid)
            cv_scores.append(cv_score)
            oof_arrays.append(y_pred)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(runs)} loaded (cached={n_cached}, downloaded={n_downloaded})...")

    print(f"  {len(candidate_ids)} candidates loaded (cached={n_cached}, downloaded={n_downloaded}).")

    if not candidate_ids:
        print("No usable candidates after loading. Exiting.")
        sys.exit(1)

    oof_matrix = np.vstack(oof_arrays)  # (n_candidates, n_rows)

    weights, final_auc = hill_climb(
        oof_matrix, y_true_ref, step_size=args.step_size, threshold=args.threshold
    )

    print_component_table(candidate_ids, cv_scores, weights)
    emit_config_snippet(candidate_ids, weights)


if __name__ == "__main__":
    main()
