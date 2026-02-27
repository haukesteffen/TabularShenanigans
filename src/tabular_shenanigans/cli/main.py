from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from tabular_shenanigans.core.config import (
    PROJECT_ROOT,
    list_competitions,
    list_profiles,
    load_runtime_config,
    write_yaml_file,
)
from tabular_shenanigans.core.device import detect_device
from tabular_shenanigans.core.logging import configure_logging
from tabular_shenanigans.core.manifest import write_manifest
from tabular_shenanigans.core.metrics import metric_direction, resolve_metric
from tabular_shenanigans.core.runs import (
    build_run_id,
    get_latest_run_id,
    list_run_ids,
    prepare_new_run,
    promote_run_to_latest,
    resolve_existing_run,
)
from tabular_shenanigans.data.kaggle_api import fetch_competition_data
from tabular_shenanigans.models.inference import generate_predictions
from tabular_shenanigans.models.catalog import baseline_specs, meta_learner_specs, tune_space
from tabular_shenanigans.models.stacking import build_stack_artifacts
from tabular_shenanigans.models.tuning import sample_params
from tabular_shenanigans.models.training import train_baseline
from tabular_shenanigans.submission.formatting import make_submission
from tabular_shenanigans.submission.kaggle_submit import (
    list_submissions,
    submit_to_kaggle,
    write_submission_meta,
)
from tabular_shenanigans.submission.validate import validate_submission_file

app = typer.Typer(help="Reusable tabular Kaggle workflow CLI")


def _runtime_dirs(cfg: dict) -> tuple[Path, Path, Path]:
    paths = cfg.get("paths", {})
    data_dir = PROJECT_ROOT / paths.get("data_dir", "data")
    artifacts_dir = PROJECT_ROOT / paths.get("artifacts_dir", "artifacts")
    reports_dir = PROJECT_ROOT / paths.get("reports_dir", "reports")
    return data_dir, artifacts_dir, reports_dir


def _collect_run_summaries(artifacts_dir: Path, competition: str, limit: int | None = None) -> list[dict]:
    run_ids = list_run_ids(artifacts_dir, competition)
    if limit is not None:
        run_ids = run_ids[:limit]

    rows: list[dict] = []
    for run_id in run_ids:
        run_dir = artifacts_dir / competition / "runs" / run_id
        manifest_path = run_dir / "run_manifest.json"
        row = {
            "run_id": run_id,
            "model": "-",
            "metric": "-",
            "metric_direction": "-",
            "score": None,
            "metrics": {},
            "timestamp_utc": "-",
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            results = manifest.get("results", {})
            row["model"] = str(results.get("model_family", "-"))
            row["metric"] = str(results.get("metric", "-"))
            row["metric_direction"] = str(results.get("metric_direction", "-"))
            metrics_dict = results.get("metrics", {})
            if isinstance(metrics_dict, dict):
                row["metrics"] = {
                    str(k).lower(): float(v)
                    for k, v in metrics_dict.items()
                    if isinstance(v, (int, float))
                }
            score = results.get("cv_score")
            try:
                row["score"] = float(score) if score is not None else None
            except (TypeError, ValueError):
                row["score"] = None
            row["timestamp_utc"] = str(manifest.get("timestamp_utc", "-"))
        rows.append(row)
    return rows


def _score_is_better(candidate: float, incumbent: float, direction: str) -> bool:
    if direction == "maximize":
        return candidate > incumbent
    return candidate < incumbent


def _run_tuning_trials(
    *,
    cfg: dict,
    competition: str,
    profile: str,
    data_dir: Path,
    artifacts_dir: Path,
    model_family: str,
    base_params: dict,
    search_space: dict,
    total_trials: int,
    sampler_seed: int,
    run_tag: str,
) -> tuple[dict, list[dict]]:
    import random

    rng = random.Random(sampler_seed)
    trial_rows: list[dict] = []
    best_trial: dict | None = None

    for idx in range(1, total_trials + 1):
        sampled_params = sample_params(search_space, rng)
        merged_params = {**base_params, **sampled_params}

        trial_cfg = deepcopy(cfg)
        trial_cfg.setdefault("training", {})
        trial_cfg["training"]["model_family"] = model_family
        trial_cfg["training"]["model_params"] = merged_params

        trial_run_id = f"{build_run_id(run_tag)}_t{idx:03d}"
        trial_run_dir = artifacts_dir / competition / "runs" / trial_run_id
        if trial_run_dir.exists():
            trial_run_id = f"{trial_run_id}_{rng.randint(1000, 9999)}"
            trial_run_dir = artifacts_dir / competition / "runs" / trial_run_id
        trial_run_dir.mkdir(parents=True, exist_ok=False)

        resolved_config_path = trial_run_dir / "resolved_config.yaml"
        write_yaml_file(resolved_config_path, trial_cfg)

        results = train_baseline(
            runtime_cfg=trial_cfg,
            output_dir=trial_run_dir,
            data_dir=data_dir,
            competition=competition,
        )
        write_manifest(
            trial_run_dir / "run_manifest.json",
            {
                "competition": competition,
                "profile": profile,
                "run_id": trial_run_id,
                "run_dir": str(trial_run_dir),
                "latest_dir": str(artifacts_dir / competition / "latest"),
                "resolved_config_path": str(resolved_config_path),
                "device": detect_device(trial_cfg),
                "results": results,
                "tuning": {
                    "trial_index": idx,
                    "sampled_params": sampled_params,
                    "merged_model_params": merged_params,
                    "sampler_seed": sampler_seed,
                    "total_trials": total_trials,
                    "model_family": model_family,
                },
            },
        )

        row = {
            "trial_index": idx,
            "run_id": trial_run_id,
            "score": float(results.get("cv_score", 0.0)),
            "metric": str(results.get("metric", "-")),
            "metric_direction": str(results.get("metric_direction", "maximize")),
            "model_family": model_family,
            "sampled_params": sampled_params,
        }
        trial_rows.append(row)

        if best_trial is None:
            best_trial = row
        else:
            direction = row["metric_direction"]
            if _score_is_better(float(row["score"]), float(best_trial["score"]), direction):
                best_trial = row

        typer.echo(
            f"trial={idx}/{total_trials} model={model_family} run_id={trial_run_id} "
            f"score={row['score']:.10g} metric={row['metric']}"
        )

    if best_trial is None:
        raise RuntimeError("No tuning trials were executed.")
    return best_trial, trial_rows


def _detect_task_type(cfg: dict, data_dir: Path, competition: str) -> str:
    task_type = str(cfg.get("schema", {}).get("task_type", "auto")).lower()
    if task_type in {"classification", "regression"}:
        return task_type
    y_path = data_dir / competition / "processed" / "y_train.csv"
    if not y_path.exists():
        return "classification"
    target_col = str(
        cfg.get("schema", {}).get("target")
        or cfg.get("submission", {}).get("target_column", "target")
    )
    y_df = pd.read_csv(y_path)
    if target_col not in y_df.columns:
        return "classification"
    y = y_df[target_col]
    if y.dtype == "object" or y.nunique(dropna=False) <= 20:
        return "classification"
    return "regression"


def _train_single_run(
    *,
    cfg: dict,
    competition: str,
    profile: str,
    data_dir: Path,
    artifacts_dir: Path,
    run_id: str,
) -> dict:
    run_dir = artifacts_dir / competition / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config_path = run_dir / "resolved_config.yaml"
    write_yaml_file(resolved_config_path, cfg)
    results = train_baseline(
        runtime_cfg=cfg,
        output_dir=run_dir,
        data_dir=data_dir,
        competition=competition,
    )
    write_manifest(
        run_dir / "run_manifest.json",
        {
            "competition": competition,
            "profile": profile,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "latest_dir": str(artifacts_dir / competition / "latest"),
            "resolved_config_path": str(resolved_config_path),
            "device": detect_device(cfg),
            "results": results,
        },
    )
    return {"run_id": run_id, "run_dir": run_dir, "results": results}


@app.command("validate-env")
def validate_env(
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    competition: str = typer.Option("titanic", help="Competition config name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    device = detect_device(cfg)
    typer.echo(f"competition={competition}")
    typer.echo(f"profile={profile}")
    typer.echo(f"device={device}")


@app.command("fetch-data")
def fetch_data(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    force: bool = typer.Option(False, help="Force re-download from Kaggle"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, _, _ = _runtime_dirs(cfg)
    competition_data_dir = data_dir / competition / "raw"
    competition_slug = cfg.get("competition", {}).get("kaggle_slug", competition)

    out = fetch_competition_data(
        competition_slug=competition_slug,
        output_dir=competition_data_dir,
        force=force,
    )
    typer.echo(f"Data prepared at: {out}")


@app.command("prepare-data")
def prepare_data(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, _, _ = _runtime_dirs(cfg)
    from tabular_shenanigans.data.prepare import prepare_competition_data

    _, _, _, _, target_col, id_col, task_type = prepare_competition_data(
        runtime_cfg=cfg,
        competition=competition,
        data_dir=data_dir,
    )
    processed_dir = data_dir / competition / "processed"
    report_path = processed_dir / "prepare_report.json"
    typer.echo(f"Prepared data at: {processed_dir}")
    typer.echo(f"prepare_report={report_path}")
    typer.echo(f"task_type={task_type}")
    typer.echo(f"target_column={target_col}")
    typer.echo(f"id_column={id_col}")


@app.command("train")
def train(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    run_id: str | None = typer.Option(
        None,
        help="Optional run id override. If omitted, a new run id is generated.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)

    model_family = str(cfg.get("training", {}).get("model_family", "sklearn"))
    if run_id:
        current_run_id = run_id
        run_dir = artifacts_dir / competition / "runs" / current_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        current_run_id, run_dir = prepare_new_run(
            artifacts_dir=artifacts_dir,
            competition=competition,
            model_family=model_family,
        )

    resolved_config_path = run_dir / "resolved_config.yaml"
    write_yaml_file(resolved_config_path, cfg)

    results = train_baseline(
        runtime_cfg=cfg,
        output_dir=run_dir,
        data_dir=data_dir,
        competition=competition,
    )
    latest_path = artifacts_dir / competition / "latest"

    write_manifest(
        run_dir / "run_manifest.json",
        {
            "competition": competition,
            "profile": profile,
            "run_id": current_run_id,
            "run_dir": str(run_dir),
            "latest_dir": str(latest_path),
            "resolved_config_path": str(resolved_config_path),
            "device": detect_device(cfg),
            "results": results,
        },
    )
    latest_path = promote_run_to_latest(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=current_run_id,
        run_path=run_dir,
    )
    typer.echo(f"Run id: {current_run_id}")
    typer.echo(f"Training artifacts: {run_dir}")


@app.command("baseline")
def baseline(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    promote_best: bool = typer.Option(True, "--promote-best/--no-promote-best"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)
    from tabular_shenanigans.data.prepare import prepare_competition_data

    prepare_competition_data(runtime_cfg=cfg, competition=competition, data_dir=data_dir)
    task_type = _detect_task_type(cfg, data_dir, competition)
    specs = baseline_specs(task_type)
    if not specs:
        raise ValueError(f"No baseline model specs available for task_type={task_type}")

    best: dict | None = None
    for spec in specs:
        run_cfg = deepcopy(cfg)
        run_cfg.setdefault("training", {})
        run_cfg["training"]["model_family"] = spec["model_family"]
        run_cfg["training"]["model_params"] = dict(spec.get("model_params", {}))
        run_id = build_run_id(f"baseline_{spec['name']}_{spec['model_family']}")
        out = _train_single_run(
            cfg=run_cfg,
            competition=competition,
            profile=profile,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            run_id=run_id,
        )
        generate_predictions(
            run_dir=out["run_dir"],
            processed_test_path=data_dir / competition / "processed" / "X_test.csv",
            pred_path=out["run_dir"] / "predictions.csv",
        )
        score = float(out["results"]["cv_score"])
        direction = str(out["results"]["metric_direction"])
        typer.echo(
            f"baseline model={spec['model_family']} run_id={run_id} "
            f"score={score:.10g} metric={out['results']['metric']}"
        )
        candidate = {"run_id": run_id, "score": score, "direction": direction}
        if best is None or _score_is_better(score, float(best["score"]), direction):
            best = candidate

    if best and promote_best:
        latest_path = promote_run_to_latest(
            artifacts_dir=artifacts_dir,
            competition=competition,
            run_id=str(best["run_id"]),
            run_path=artifacts_dir / competition / "runs" / str(best["run_id"]),
        )
        typer.echo(f"Best baseline promoted: run_id={best['run_id']}")
        typer.echo(f"Latest updated: {latest_path}")


@app.command("tune-stage")
def tune_stage(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    n_trials: int = typer.Option(10, help="Trials per model family."),
    promote_best: bool = typer.Option(True, "--promote-best/--no-promote-best"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("optuna is required for tune-stage. Install with `uv pip install optuna`.") from exc

    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)
    from tabular_shenanigans.data.prepare import prepare_competition_data

    prepare_competition_data(runtime_cfg=cfg, competition=competition, data_dir=data_dir)
    task_type = _detect_task_type(cfg, data_dir, competition)
    specs = baseline_specs(task_type)
    metric_name, metric_direction = resolve_metric(cfg, task_type)
    direction_optuna = "maximize" if metric_direction == "maximize" else "minimize"

    all_best: list[dict] = []
    tuning_dir = artifacts_dir / competition / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        family = spec["model_family"]
        name = spec["name"]
        base_params = dict(spec.get("model_params", {}))
        search_space = tune_space(family)
        if not search_space:
            typer.echo(f"Skipping {family}: no tune search space.")
            continue

        typer.echo(f"Tuning model={family} trials={n_trials}")
        run_ids: list[str] = []
        trial_rows: list[dict] = []

        def objective(trial: "optuna.Trial") -> float:
            sampled = {}
            for key, spec_def in search_space.items():
                t = str(spec_def.get("type", "")).lower()
                if t == "int":
                    sampled[key] = trial.suggest_int(
                        key,
                        int(spec_def["low"]),
                        int(spec_def["high"]),
                        step=int(spec_def.get("step", 1)),
                    )
                elif t == "float":
                    sampled[key] = trial.suggest_float(
                        key,
                        float(spec_def["low"]),
                        float(spec_def["high"]),
                        log=bool(spec_def.get("log", False)),
                    )
                elif t == "categorical":
                    sampled[key] = trial.suggest_categorical(key, list(spec_def["choices"]))
                else:
                    sampled[key] = sample_params({key: spec_def}, random.Random(trial.number))[key]

            merged = {**base_params, **sampled}
            trial_cfg = deepcopy(cfg)
            trial_cfg.setdefault("training", {})
            trial_cfg["training"]["model_family"] = family
            trial_cfg["training"]["model_params"] = merged

            run_id = f"{build_run_id(f'tune_stage_{name}_{family}')}_t{trial.number:03d}"
            out = _train_single_run(
                cfg=trial_cfg,
                competition=competition,
                profile=profile,
                data_dir=data_dir,
                artifacts_dir=artifacts_dir,
                run_id=run_id,
            )
            generate_predictions(
                run_dir=out["run_dir"],
                processed_test_path=data_dir / competition / "processed" / "X_test.csv",
                pred_path=out["run_dir"] / "predictions.csv",
            )
            score = float(out["results"]["cv_score"])
            run_ids.append(run_id)
            trial_rows.append(
                {"trial": trial.number, "run_id": run_id, "score": score, "sampled_params": sampled}
            )
            trial.set_user_attr("run_id", run_id)
            return score

        study = optuna.create_study(direction=direction_optuna)
        study.optimize(objective, n_trials=n_trials)
        best_run_id = str(study.best_trial.user_attrs.get("run_id"))
        best_score = float(study.best_value)
        all_best.append({"name": name, "model_family": family, "run_id": best_run_id, "score": best_score})
        typer.echo(f"Best tuned model={family} run_id={best_run_id} score={best_score:.10g}")

        family_summary = {
            "competition": competition,
            "profile": profile,
            "stage": "tune",
            "model_family": family,
            "name": name,
            "metric": metric_name,
            "metric_direction": metric_direction,
            "best_run_id": best_run_id,
            "best_score": best_score,
            "trials": trial_rows,
        }
        out_path = tuning_dir / (
            f"stage_tune_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{name}_{family}.json"
        )
        out_path.write_text(json.dumps(family_summary, indent=2), encoding="utf-8")

    if all_best and promote_best:
        best = all_best[0]
        for candidate in all_best[1:]:
            if _score_is_better(float(candidate["score"]), float(best["score"]), metric_direction):
                best = candidate
        latest_path = promote_run_to_latest(
            artifacts_dir=artifacts_dir,
            competition=competition,
            run_id=str(best["run_id"]),
            run_path=artifacts_dir / competition / "runs" / str(best["run_id"]),
        )
        typer.echo(f"Best tuned run promoted: run_id={best['run_id']}")
        typer.echo(f"Latest updated: {latest_path}")


@app.command("stack-stage")
def stack_stage(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    n_trials: int = typer.Option(10, help="Trials for tunable meta learners."),
    promote_best: bool = typer.Option(True, "--promote-best/--no-promote-best"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("optuna is required for stack-stage. Install with `uv pip install optuna`.") from exc

    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)
    task_type = _detect_task_type(cfg, data_dir, competition)
    metric_name, metric_direction = resolve_metric(cfg, task_type)

    rows = _collect_run_summaries(artifacts_dir, competition, limit=None)
    model_families = {spec["model_family"] for spec in baseline_specs(task_type)}
    family_best: dict[str, dict] = {}
    for row in rows:
        if row["model"] not in model_families or row["score"] is None:
            continue
        fam = str(row["model"])
        if fam not in family_best or _score_is_better(
            float(row["score"]), float(family_best[fam]["score"]), metric_direction
        ):
            family_best[fam] = row
    base_run_ids = [str(v["run_id"]) for v in family_best.values()]
    if len(base_run_ids) < 2:
        raise ValueError("Need at least two tuned base runs to execute stack-stage.")

    processed_test_path = data_dir / competition / "processed" / "X_test.csv"
    for base_run_id in base_run_ids:
        base_run_dir = artifacts_dir / competition / "runs" / base_run_id
        pred_path = base_run_dir / "predictions.csv"
        if not pred_path.exists():
            generate_predictions(run_dir=base_run_dir, processed_test_path=processed_test_path, pred_path=pred_path)

    best_stack: dict | None = None
    for meta in meta_learner_specs(task_type):
        method = str(meta["method"])
        name = str(meta["name"])
        search_space = dict(meta.get("search_space", {}))

        if method == "mean" or not search_space:
            run_id = build_run_id(f"stack_stage_{name}")
            run_dir = artifacts_dir / competition / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            write_yaml_file(run_dir / "resolved_config.yaml", cfg)
            info = build_stack_artifacts(
                runtime_cfg=cfg,
                artifacts_dir=artifacts_dir,
                data_dir=data_dir,
                competition=competition,
                run_ids=base_run_ids,
                output_dir=run_dir,
                method=method,
            )
            write_manifest(
                run_dir / "run_manifest.json",
                {
                    "competition": competition,
                    "profile": profile,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "latest_dir": str(artifacts_dir / competition / "latest"),
                    "resolved_config_path": str(run_dir / "resolved_config.yaml"),
                    "device": detect_device(cfg),
                    "results": {
                        "status": "ok",
                        "task_type": info["task_type"],
                        "metric": info["metric"],
                        "metric_direction": info["metric_direction"],
                        "cv_score": info["cv_score"],
                        "model_family": f"stack_{method}",
                        "n_folds": 0,
                        "metrics": {info["metric"]: info["cv_score"]},
                        "fold_metrics": {},
                    },
                    "stack": info,
                },
            )
            candidate = {"run_id": run_id, "score": float(info["cv_score"]), "method": method}
        else:
            direction_optuna = "maximize" if metric_direction == "maximize" else "minimize"

            def objective(trial: "optuna.Trial") -> float:
                meta_params = {}
                for key, spec_def in search_space.items():
                    if spec_def.get("type") == "float":
                        meta_params[key] = trial.suggest_float(
                            key,
                            float(spec_def["low"]),
                            float(spec_def["high"]),
                            log=bool(spec_def.get("log", False)),
                        )
                run_id = f"{build_run_id(f'stack_stage_{name}')}_t{trial.number:03d}"
                run_dir = artifacts_dir / competition / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=False)
                write_yaml_file(run_dir / "resolved_config.yaml", cfg)
                info = build_stack_artifacts(
                    runtime_cfg=cfg,
                    artifacts_dir=artifacts_dir,
                    data_dir=data_dir,
                    competition=competition,
                    run_ids=base_run_ids,
                    output_dir=run_dir,
                    method=method,
                    meta_params=meta_params,
                )
                write_manifest(
                    run_dir / "run_manifest.json",
                    {
                        "competition": competition,
                        "profile": profile,
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "latest_dir": str(artifacts_dir / competition / "latest"),
                        "resolved_config_path": str(run_dir / "resolved_config.yaml"),
                        "device": detect_device(cfg),
                        "results": {
                            "status": "ok",
                            "task_type": info["task_type"],
                            "metric": info["metric"],
                            "metric_direction": info["metric_direction"],
                            "cv_score": info["cv_score"],
                            "model_family": f"stack_{method}",
                            "n_folds": 0,
                            "metrics": {info["metric"]: info["cv_score"]},
                            "fold_metrics": {},
                        },
                        "stack": info,
                        "meta_params": meta_params,
                    },
                )
                trial.set_user_attr("run_id", run_id)
                return float(info["cv_score"])

            study = optuna.create_study(direction=direction_optuna)
            study.optimize(objective, n_trials=n_trials)
            candidate = {
                "run_id": str(study.best_trial.user_attrs["run_id"]),
                "score": float(study.best_value),
                "method": method,
            }

        typer.echo(
            f"stack meta={name} method={method} run_id={candidate['run_id']} "
            f"score={candidate['score']:.10g}"
        )
        if best_stack is None or _score_is_better(
            float(candidate["score"]), float(best_stack["score"]), metric_direction
        ):
            best_stack = candidate

    if best_stack and promote_best:
        latest_path = promote_run_to_latest(
            artifacts_dir=artifacts_dir,
            competition=competition,
            run_id=str(best_stack["run_id"]),
            run_path=artifacts_dir / competition / "runs" / str(best_stack["run_id"]),
        )
        typer.echo(f"Best stacked run promoted: run_id={best_stack['run_id']}")
        typer.echo(f"Latest updated: {latest_path}")


@app.command("tune")
def tune(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    n_trials: int | None = typer.Option(
        None,
        help="Override number of tuning trials. Defaults to training.tune.n_trials.",
    ),
    seed: int | None = typer.Option(
        None,
        help="Override tuning sampler seed. Defaults to global seed.",
    ),
    promote_best: bool = typer.Option(
        True,
        "--promote-best/--no-promote-best",
        help="Promote best trial run as latest.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)

    training_cfg = cfg.get("training", {})
    tune_cfg = training_cfg.get("tune", {})
    ensemble_cfg = training_cfg.get("ensemble", {})
    learners_cfg = ensemble_cfg.get("learners", [])

    sampler_seed = int(seed if seed is not None else cfg.get("seed", 42))

    learner_summaries: list[dict] = []
    all_trial_rows: list[dict] = []
    best_run_ids: list[str] = []

    if isinstance(learners_cfg, list) and learners_cfg:
        for learner_index, learner in enumerate(learners_cfg, start=1):
            if not isinstance(learner, dict):
                raise ValueError("Each training.ensemble.learners entry must be a mapping.")

            learner_name = str(learner.get("name", f"learner_{learner_index}"))
            learner_family = str(learner.get("model_family", "")).lower()
            if not learner_family:
                raise ValueError("Each learner must define model_family.")

            learner_base_params = learner.get("model_params", {})
            learner_tune_cfg = learner.get("tune", {})
            search_space = learner_tune_cfg.get("search_space", tune_cfg.get("search_space", {}))
            if not isinstance(search_space, dict) or not search_space:
                raise ValueError(
                    f"Missing tune.search_space for learner '{learner_name}'."
                )

            total_trials = int(
                n_trials
                if n_trials is not None
                else learner_tune_cfg.get("n_trials", tune_cfg.get("n_trials", 10))
            )
            if total_trials <= 0:
                raise ValueError("n_trials must be > 0.")

            typer.echo(
                f"Tuning learner '{learner_name}' model_family={learner_family} "
                f"trials={total_trials}"
            )
            best_trial, trials = _run_tuning_trials(
                cfg=cfg,
                competition=competition,
                profile=profile,
                data_dir=data_dir,
                artifacts_dir=artifacts_dir,
                model_family=learner_family,
                base_params=learner_base_params,
                search_space=search_space,
                total_trials=total_trials,
                sampler_seed=sampler_seed + learner_index,
                run_tag=f"tune_{learner_name}_{learner_family}",
            )
            learner_summaries.append(
                {
                    "name": learner_name,
                    "model_family": learner_family,
                    "best_trial": best_trial,
                }
            )
            all_trial_rows.extend([{**row, "learner": learner_name} for row in trials])
            best_run_ids.append(str(best_trial["run_id"]))
            typer.echo(
                f"Best learner trial: name={learner_name} run_id={best_trial['run_id']} "
                f"score={best_trial['score']:.10g}"
            )
    else:
        search_space = tune_cfg.get("search_space", {})
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError(
                "Missing training.tune.search_space in config. "
                "Define parameter ranges before running tune."
            )
        total_trials = int(n_trials if n_trials is not None else tune_cfg.get("n_trials", 10))
        if total_trials <= 0:
            raise ValueError("n_trials must be > 0.")

        base_model_family = str(training_cfg.get("model_family", "sklearn")).lower()
        base_params = training_cfg.get("model_params", {})
        best_trial, trials = _run_tuning_trials(
            cfg=cfg,
            competition=competition,
            profile=profile,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            model_family=base_model_family,
            base_params=base_params,
            search_space=search_space,
            total_trials=total_trials,
            sampler_seed=sampler_seed,
            run_tag=f"tune_{base_model_family}",
        )
        learner_summaries.append(
            {"name": base_model_family, "model_family": base_model_family, "best_trial": best_trial}
        )
        all_trial_rows.extend(trials)
        best_run_ids.append(str(best_trial["run_id"]))
        typer.echo(
            f"Best trial: run_id={best_trial['run_id']} score={best_trial['score']:.10g}"
        )

    # Optional ensemble auto-stack after tuning learner winners.
    stack_cfg = ensemble_cfg.get("stack", {})
    stack_enabled = bool(stack_cfg.get("enabled", False))
    if stack_enabled and len(best_run_ids) >= 2:
        stack_method = str(stack_cfg.get("method", "linear")).lower()
        stack_run_id, stack_run_dir = prepare_new_run(
            artifacts_dir=artifacts_dir,
            competition=competition,
            model_family=f"stack_{stack_method}_tuned",
        )
        stack_config_path = stack_run_dir / "resolved_config.yaml"
        write_yaml_file(stack_config_path, cfg)

        processed_test_path = data_dir / competition / "processed" / "X_test.csv"
        for base_run_id in best_run_ids:
            base_run_dir = artifacts_dir / competition / "runs" / base_run_id
            pred_path = base_run_dir / "predictions.csv"
            if not pred_path.exists():
                generate_predictions(
                    run_dir=base_run_dir,
                    processed_test_path=processed_test_path,
                    pred_path=pred_path,
                )

        stack_info = build_stack_artifacts(
            runtime_cfg=cfg,
            artifacts_dir=artifacts_dir,
            data_dir=data_dir,
            competition=competition,
            run_ids=best_run_ids,
            output_dir=stack_run_dir,
            method=stack_method,
        )
        write_manifest(
            stack_run_dir / "run_manifest.json",
            {
                "competition": competition,
                "profile": profile,
                "run_id": stack_run_id,
                "run_dir": str(stack_run_dir),
                "latest_dir": str(artifacts_dir / competition / "latest"),
                "resolved_config_path": str(stack_config_path),
                "device": detect_device(cfg),
                "results": {
                    "status": "ok",
                    "task_type": stack_info.get("task_type", "unknown"),
                    "metric": "stack",
                    "metric_direction": "maximize",
                    "cv_score": None,
                    "model_family": f"stack_{stack_method}",
                    "n_folds": 0,
                    "target_column": cfg.get("schema", {}).get("target"),
                    "id_column": cfg.get("schema", {}).get("id_column"),
                    "metrics": {},
                    "fold_metrics": {},
                },
                "stack": stack_info,
                "tuning": {"base_best_run_ids": best_run_ids},
            },
        )
        promote_stack = bool(stack_cfg.get("promote", True))
        if promote_stack and promote_best:
            latest_path = promote_run_to_latest(
                artifacts_dir=artifacts_dir,
                competition=competition,
                run_id=stack_run_id,
                run_path=stack_run_dir,
            )
            typer.echo(f"Auto-stacked best learners into run_id={stack_run_id}")
            typer.echo(f"Latest updated: {latest_path}")

    best_trial = None
    for summary_item in learner_summaries:
        candidate = summary_item["best_trial"]
        if best_trial is None:
            best_trial = candidate
        elif _score_is_better(
            float(candidate["score"]),
            float(best_trial["score"]),
            str(candidate["metric_direction"]),
        ):
            best_trial = candidate

    if best_trial is None:
        raise RuntimeError("No tuning trials were executed.")

    summary = {
        "competition": competition,
        "profile": profile,
        "sampler_seed": sampler_seed,
        "learners": learner_summaries,
        "best_trial_overall": best_trial,
        "best_run_ids": best_run_ids,
        "trials": all_trial_rows,
    }
    tuning_dir = artifacts_dir / competition / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tuning_dir / f"tune_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if promote_best:
        best_run_id = str(best_trial["run_id"])
        best_run_path = artifacts_dir / competition / "runs" / best_run_id
        latest_path = promote_run_to_latest(
            artifacts_dir=artifacts_dir,
            competition=competition,
            run_id=best_run_id,
            run_path=best_run_path,
        )
        typer.echo(f"Best trial promoted: run_id={best_run_id}")
        typer.echo(f"Latest updated: {latest_path}")

    typer.echo(f"Tuning summary: {summary_path}")


@app.command("predict")
def predict(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    run_id: str | None = typer.Option(
        None,
        help="Run id to use. Defaults to latest tracked run.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)

    current_run_id, run_dir = resolve_existing_run(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=run_id,
    )
    pred_path = run_dir / "predictions.csv"
    processed_test_path = data_dir / competition / "processed" / "X_test.csv"
    out = generate_predictions(
        run_dir=run_dir,
        processed_test_path=processed_test_path,
        pred_path=pred_path,
    )
    latest_path = promote_run_to_latest(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=current_run_id,
        run_path=run_dir,
    )
    typer.echo(f"Run id: {current_run_id}")
    typer.echo(f"Predictions written to: {out}")
    typer.echo(f"Latest updated: {latest_path}")


@app.command("stack")
def stack(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    run_id: list[str] = typer.Option(
        ...,
        "--run-id",
        help="Base run id to include in stack. Repeat for multiple runs.",
    ),
    method: str = typer.Option(
        "linear",
        help="Stacking method: linear or mean.",
    ),
    output_run_id: str | None = typer.Option(
        None,
        help="Optional output run id for stacked artifacts.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)

    if output_run_id:
        current_run_id = output_run_id
        run_dir = artifacts_dir / competition / "runs" / current_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        current_run_id, run_dir = prepare_new_run(
            artifacts_dir=artifacts_dir,
            competition=competition,
            model_family=f"stack_{method}",
        )

    resolved_config_path = run_dir / "resolved_config.yaml"
    write_yaml_file(resolved_config_path, cfg)

    processed_test_path = data_dir / competition / "processed" / "X_test.csv"
    for base_run_id in run_id:
        base_run_dir = artifacts_dir / competition / "runs" / base_run_id
        pred_path = base_run_dir / "predictions.csv"
        if not pred_path.exists():
            generate_predictions(
                run_dir=base_run_dir,
                processed_test_path=processed_test_path,
                pred_path=pred_path,
            )

    stack_info = build_stack_artifacts(
        runtime_cfg=cfg,
        artifacts_dir=artifacts_dir,
        data_dir=data_dir,
        competition=competition,
        run_ids=list(run_id),
        output_dir=run_dir,
        method=method,
    )

    latest_path = artifacts_dir / competition / "latest"
    write_manifest(
        run_dir / "run_manifest.json",
        {
            "competition": competition,
            "profile": profile,
            "run_id": current_run_id,
            "run_dir": str(run_dir),
            "latest_dir": str(latest_path),
            "resolved_config_path": str(resolved_config_path),
            "device": detect_device(cfg),
            "results": {
                "status": "ok",
                "task_type": stack_info.get("task_type", "unknown"),
                "metric": "stack",
                "metric_direction": "maximize",
                "cv_score": None,
                "model_family": f"stack_{method}",
                "n_folds": 0,
                "target_column": cfg.get("schema", {}).get("target"),
                "id_column": cfg.get("schema", {}).get("id_column"),
                "metrics": {},
                "fold_metrics": {},
            },
            "stack": stack_info,
        },
    )
    latest_path = promote_run_to_latest(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=current_run_id,
        run_path=run_dir,
    )
    typer.echo(f"Run id: {current_run_id}")
    typer.echo(f"Stacked artifacts: {run_dir}")
    typer.echo(f"Latest updated: {latest_path}")


@app.command("make-submission")
def make_submission_cmd(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    run_id: str | None = typer.Option(
        None,
        help="Run id to use. Defaults to latest tracked run.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)

    submission_cfg = cfg.get("submission", {})
    id_col = submission_cfg.get("id_column", "id")
    target_col = submission_cfg.get("target_column", "target")
    prediction_type = str(submission_cfg.get("prediction_type", "raw"))
    threshold = float(submission_cfg.get("classification_threshold", 0.5))
    positive_label = submission_cfg.get("positive_label", 1)
    negative_label = submission_cfg.get("negative_label", 0)

    current_run_id, run_dir = resolve_existing_run(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=run_id,
    )

    ids_path = data_dir / competition / "processed" / "test_ids.csv"
    preds_path = run_dir / "predictions.csv"
    output_path = run_dir / "submission.csv"

    if not ids_path.exists():
        raise FileNotFoundError(f"Missing {ids_path}. Run train first to prepare processed IDs.")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}. Run predict first.")

    out = make_submission(
        ids_path=ids_path,
        preds_path=preds_path,
        output_path=output_path,
        id_col=id_col,
        target_col=target_col,
        prediction_type=prediction_type,
        classification_threshold=threshold,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    validate_submission_file(
        submission_path=output_path,
        ids_path=ids_path,
        id_col=id_col,
        target_col=target_col,
        prediction_type=prediction_type,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    latest_path = promote_run_to_latest(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=current_run_id,
        run_path=run_dir,
    )
    typer.echo(f"Run id: {current_run_id}")
    typer.echo(f"Submission written to: {out}")
    typer.echo(f"Latest updated: {latest_path}")


@app.command("submit")
def submit(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    message: str = typer.Option(..., help="Kaggle submission message"),
    run_id: str | None = typer.Option(
        None,
        help="Run id to use. Defaults to latest tracked run.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)
    competition_slug = cfg.get("competition", {}).get("kaggle_slug", competition)
    submission_cfg = cfg.get("submission", {})
    id_col = submission_cfg.get("id_column", "id")
    target_col = submission_cfg.get("target_column", "target")
    prediction_type = str(submission_cfg.get("prediction_type", "raw"))
    positive_label = submission_cfg.get("positive_label", 1)
    negative_label = submission_cfg.get("negative_label", 0)

    current_run_id, run_dir = resolve_existing_run(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=run_id,
    )
    submission_path = run_dir / "submission.csv"
    if not submission_path.exists():
        raise FileNotFoundError(f"Missing {submission_path}. Run make-submission first.")
    ids_path = data_dir / competition / "processed" / "test_ids.csv"
    validate_submission_file(
        submission_path=submission_path,
        ids_path=ids_path,
        id_col=id_col,
        target_col=target_col,
        prediction_type=prediction_type,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    if not yes:
        confirmed = typer.confirm(
            f"Submit {submission_path} to Kaggle competition '{competition_slug}'?"
        )
        if not confirmed:
            typer.echo("Submission cancelled.")
            raise typer.Exit()

    meta = submit_to_kaggle(
        competition_slug=competition_slug,
        submission_path=submission_path,
        message=message,
    )
    write_submission_meta(run_dir / "submission_meta.json", meta)
    latest_path = promote_run_to_latest(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=current_run_id,
        run_path=run_dir,
    )
    typer.echo(meta["stdout"])
    typer.echo(f"Run id: {current_run_id}")
    typer.echo(f"Submission metadata written to: {run_dir / 'submission_meta.json'}")
    typer.echo(f"Latest updated: {latest_path}")


@app.command("submissions")
def submissions(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    competition_slug = cfg.get("competition", {}).get("kaggle_slug", competition)
    typer.echo(list_submissions(competition_slug))


@app.command("runs")
def runs(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    limit: int = typer.Option(20, help="Max number of runs to display"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    _, artifacts_dir, _ = _runtime_dirs(cfg)

    latest_id = get_latest_run_id(artifacts_dir, competition)
    rows = _collect_run_summaries(artifacts_dir, competition, limit=limit)
    if not rows:
        typer.echo("No runs found.")
        return

    typer.echo("latest run_id                     model      metric      score      timestamp_utc")
    for row in rows:
        score = "-" if row["score"] is None else f"{row['score']:.10g}"
        marker = "*" if latest_id == row["run_id"] else " "
        typer.echo(
            f"{marker:>6} {row['run_id']:<25} {row['model']:<10} "
            f"{row['metric']:<11} {score:<10} {row['timestamp_utc']}"
        )


@app.command("compare-runs")
def compare_runs(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    metric: str | None = typer.Option(
        None,
        help="Metric to compare by. Defaults to evaluation.metric from config.",
    ),
    limit: int = typer.Option(20, help="Max number of ranked runs to display"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    _, artifacts_dir, _ = _runtime_dirs(cfg)

    eval_cfg = cfg.get("evaluation", {})
    configured_metric = str(eval_cfg.get("metric", "accuracy")).lower()
    selected_metric = str(metric or configured_metric).lower()
    if metric is None:
        selected_direction = str(eval_cfg.get("direction", metric_direction(selected_metric))).lower()
    else:
        selected_direction = metric_direction(selected_metric)
    if selected_direction not in {"maximize", "minimize"}:
        raise ValueError("evaluation.direction must be 'maximize' or 'minimize'.")

    rows = _collect_run_summaries(artifacts_dir, competition, limit=None)
    metric_rows = [
        {
            **row,
            "selected_score": (
                row["metrics"].get(selected_metric)
                if selected_metric in row["metrics"]
                else (
                    row["score"]
                    if row["metric"].lower() == selected_metric and row["score"] is not None
                    else None
                )
            ),
        }
        for row in rows
    ]
    metric_rows = [row for row in metric_rows if row["selected_score"] is not None]
    if not metric_rows:
        available_metrics = sorted({str(row["metric"]).lower() for row in rows if row["metric"] != "-"})
        typer.echo(f"No runs found for metric '{selected_metric}'.")
        if available_metrics:
            typer.echo(f"Available metrics in stored runs: {', '.join(available_metrics)}")
        return

    reverse = selected_direction == "maximize"
    metric_rows.sort(key=lambda r: float(r["selected_score"]), reverse=reverse)
    metric_rows = metric_rows[:limit]
    best_score = float(metric_rows[0]["selected_score"])
    latest_id = get_latest_run_id(artifacts_dir, competition)

    typer.echo(
        f"Comparing runs for metric='{selected_metric}' "
        f"(direction={selected_direction}, best={best_score:.10g})"
    )
    typer.echo("rank latest run_id                     score       delta       model      timestamp_utc")

    for idx, row in enumerate(metric_rows, start=1):
        score = float(row["selected_score"])
        delta = score - best_score
        marker = "*" if latest_id == row["run_id"] else " "
        typer.echo(
            f"{idx:<4} {marker:^6} {row['run_id']:<25} {score:<11.10g} "
            f"{delta:<11.10g} {row['model']:<10} {row['timestamp_utc']}"
        )


@app.command("check")
def check(
    competition: str | None = typer.Option(
        None,
        help="Optional competition filter for config checks.",
    ),
    profile: str | None = typer.Option(
        None,
        help="Optional profile filter for config checks.",
    ),
    run_tests: bool = typer.Option(True, "--run-tests/--skip-tests", help="Run unit tests."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)

    competitions = [competition] if competition else list_competitions()
    profiles = [profile] if profile else list_profiles()
    if not competitions:
        raise ValueError("No competitions found in configs/competitions.")
    if not profiles:
        raise ValueError("No profiles found in configs/profiles.")

    typer.echo("Validating config matrix:")
    checked = 0
    for comp in competitions:
        for prof in profiles:
            _ = load_runtime_config(competition=comp, profile=prof)
            typer.echo(f"- ok: competition={comp}, profile={prof}")
            checked += 1
    typer.echo(f"Validated {checked} config combination(s).")

    if run_tests:
        typer.echo("Running unit tests...")
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": "src"},
            text=True,
        )
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)
        typer.echo("Unit tests passed.")


@app.command("best-run")
def best_run(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    metric: str | None = typer.Option(
        None,
        help="Metric to rank by. Defaults to evaluation.metric from config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    _, artifacts_dir, _ = _runtime_dirs(cfg)

    eval_cfg = cfg.get("evaluation", {})
    configured_metric = str(eval_cfg.get("metric", "accuracy")).lower()
    selected_metric = str(metric or configured_metric).lower()
    if metric is None:
        selected_direction = str(eval_cfg.get("direction", metric_direction(selected_metric))).lower()
    else:
        selected_direction = metric_direction(selected_metric)
    if selected_direction not in {"maximize", "minimize"}:
        raise ValueError("evaluation.direction must be 'maximize' or 'minimize'.")

    rows = _collect_run_summaries(artifacts_dir, competition, limit=None)
    metric_rows = [
        {
            **row,
            "selected_score": (
                row["metrics"].get(selected_metric)
                if selected_metric in row["metrics"]
                else (
                    row["score"]
                    if row["metric"].lower() == selected_metric and row["score"] is not None
                    else None
                )
            ),
        }
        for row in rows
    ]
    metric_rows = [row for row in metric_rows if row["selected_score"] is not None]
    if not metric_rows:
        typer.echo(f"No runs found for metric '{selected_metric}'.")
        return

    reverse = selected_direction == "maximize"
    metric_rows.sort(key=lambda r: float(r["selected_score"]), reverse=reverse)
    best = metric_rows[0]
    latest_id = get_latest_run_id(artifacts_dir, competition)
    marker = "*" if latest_id == best["run_id"] else " "

    typer.echo(
        f"Best run for metric='{selected_metric}' (direction={selected_direction}):"
    )
    typer.echo(f"latest={marker.strip() or '-'}")
    typer.echo(f"run_id={best['run_id']}")
    typer.echo(f"score={best['selected_score']:.10g}")
    typer.echo(f"model={best['model']}")
    typer.echo(f"timestamp_utc={best['timestamp_utc']}")


@app.command("promote-run")
def promote_run(
    competition: str = typer.Option(..., help="Competition config name"),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    run_id: str = typer.Option(..., help="Run id to promote as latest."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    cfg = load_runtime_config(competition=competition, profile=profile)
    _, artifacts_dir, _ = _runtime_dirs(cfg)

    resolved_run_id, run_path = resolve_existing_run(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=run_id,
    )
    latest_path = promote_run_to_latest(
        artifacts_dir=artifacts_dir,
        competition=competition,
        run_id=resolved_run_id,
        run_path=run_path,
    )
    typer.echo(f"Promoted run_id={resolved_run_id}")
    typer.echo(f"Latest updated: {latest_path}")


@app.command("clean")
def clean(
    competition: str | None = typer.Option(
        None,
        help="Competition config name to clean. Required unless --all-competitions is set.",
    ),
    all_competitions: bool = typer.Option(
        False,
        "--all-competitions",
        help="Clean all competitions found under data/ and artifacts/.",
    ),
    scope: list[str] = typer.Option(
        ["artifacts"],
        "--scope",
        "-s",
        help="What to clean: artifacts, processed, raw, all. Repeat for multiple scopes.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt."),
    profile: str = typer.Option("local_arm64", help="Runtime profile name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    if competition is not None:
        cfg = load_runtime_config(competition=competition, profile=profile)
    else:
        cfg = {}
    data_dir, artifacts_dir, _ = _runtime_dirs(cfg)

    if all_competitions and competition is not None:
        raise ValueError("Use either --competition or --all-competitions, not both.")
    if not all_competitions and competition is None:
        raise ValueError("Provide --competition or set --all-competitions.")

    selected_scopes = {s.lower() for s in scope}
    allowed_scopes = {"artifacts", "processed", "raw", "all"}
    unknown_scopes = sorted(selected_scopes - allowed_scopes)
    if unknown_scopes:
        raise ValueError(f"Unknown scope values: {unknown_scopes}")
    if "all" in selected_scopes:
        selected_scopes = {"artifacts", "processed", "raw"}

    if all_competitions:
        competitions = sorted(
            {
                *(p.name for p in artifacts_dir.glob("*") if p.is_dir()),
                *(p.name for p in data_dir.glob("*") if p.is_dir()),
            }
        )
    else:
        competitions = [competition] if competition is not None else []

    targets: list[Path] = []
    for comp in competitions:
        if "artifacts" in selected_scopes:
            targets.append(artifacts_dir / comp)
        if "processed" in selected_scopes:
            targets.append(data_dir / comp / "processed")
        if "raw" in selected_scopes:
            targets.append(data_dir / comp / "raw")

    existing_targets = [p for p in targets if p.exists()]
    if not existing_targets:
        typer.echo("Nothing to clean.")
        return

    typer.echo("Cleanup targets:")
    for path in existing_targets:
        typer.echo(f"- {path}")

    if dry_run:
        typer.echo("Dry-run only. No files were deleted.")
        return

    if not yes:
        confirmed = typer.confirm(
            f"Delete {len(existing_targets)} path(s) for scope={sorted(selected_scopes)}?"
        )
        if not confirmed:
            typer.echo("Cleanup cancelled.")
            raise typer.Exit()

    for path in existing_targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    typer.echo(f"Deleted {len(existing_targets)} path(s).")


if __name__ == "__main__":
    app()
