import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import optuna
import pandas as pd

from tabular_shenanigans.competition import ensure_prepared_competition_context
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better, resolve_positive_label
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.models import build_tuning_space, get_model_definition
from tabular_shenanigans.preprocess import prepare_feature_frames
from tabular_shenanigans.train import (
    TrainingModelSpec,
    _build_target_summary,
    _evaluate_model_spec,
    _json_ready,
    run_training,
)


@dataclass(frozen=True)
class TuningResult:
    study_dir: Path
    candidate_dir: Path


def _make_study_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _build_study_config_snapshot(
    config: AppConfig,
    tuning_model_spec: TrainingModelSpec,
    positive_label: object | None,
    id_column: str,
    label_column: str,
) -> dict[str, object]:
    return {
        "competition": {
            **config.competition.model_dump(mode="python"),
            "primary_metric": config.primary_metric,
            "positive_label": positive_label,
            "id_column": id_column,
            "label_column": label_column,
        },
        "experiment": config.experiment.model_dump(mode="python"),
        "resolved_model_id": tuning_model_spec.model_id,
    }


def _build_trials_df(study: optuna.Study, metric_name: str) -> pd.DataFrame:
    param_names = sorted({param_name for trial in study.trials for param_name in trial.params})
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "metric_name": metric_name,
            "metric_value": trial.value,
            "metric_std": trial.user_attrs.get("metric_std"),
            "started_at_utc": trial.datetime_start.isoformat() if trial.datetime_start is not None else "",
            "completed_at_utc": trial.datetime_complete.isoformat() if trial.datetime_complete is not None else "",
            "duration_seconds": trial.duration.total_seconds() if trial.duration is not None else None,
            "params_json": json.dumps(_json_ready(trial.params), sort_keys=True),
            "model_params_json": json.dumps(_json_ready(trial.user_attrs.get("model_params")), sort_keys=True),
        }
        for param_name in param_names:
            row[f"param_{param_name}"] = trial.params.get(param_name)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_study_manifest(
    config: AppConfig,
    study: optuna.Study,
    study_id: str,
    study_config_snapshot: dict[str, object],
    tuning_model_spec: TrainingModelSpec,
    model_name: str,
    preprocessing_scheme_id: str,
    target_summary: dict[str, object],
    candidate_dir: Path | None = None,
) -> dict[str, object]:
    best_trial = study.best_trial
    return {
        "study_id": study_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": config.competition_slug,
        "task_type": config.task_type,
        "primary_metric": config.primary_metric,
        "optimization_direction": study.direction.name.lower(),
        "config_snapshot": study_config_snapshot,
        "model_id": tuning_model_spec.model_id,
        "model_name": model_name,
        "preprocessing_scheme_id": preprocessing_scheme_id,
        "trial_count": len(study.trials),
        "completed_trial_count": len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_model_params": best_trial.user_attrs.get("model_params"),
        "target_summary": target_summary,
        "train_candidate_id": candidate_dir.name if candidate_dir is not None else None,
        "train_candidate_dir": str(candidate_dir) if candidate_dir is not None else None,
    }


def _write_tuning_artifacts(
    study_dir: Path,
    study_manifest: dict[str, object],
    trials_df: pd.DataFrame,
) -> None:
    best_params = study_manifest["best_params"]
    study_summary = pd.DataFrame(
        [
            {
                "study_id": study_manifest["study_id"],
                "competition_slug": study_manifest["competition_slug"],
                "task_type": study_manifest["task_type"],
                "primary_metric": study_manifest["primary_metric"],
                "optimization_direction": study_manifest["optimization_direction"],
                "model_id": study_manifest["model_id"],
                "model_name": study_manifest["model_name"],
                "preprocessing_scheme_id": study_manifest["preprocessing_scheme_id"],
                "trial_count": study_manifest["trial_count"],
                "completed_trial_count": study_manifest["completed_trial_count"],
                "best_trial_number": study_manifest["best_trial_number"],
                "best_value": study_manifest["best_value"],
                "train_candidate_id": study_manifest["train_candidate_id"],
            }
        ]
    )
    study_summary.to_csv(study_dir / "study_summary.csv", index=False)
    trials_df.to_csv(study_dir / "trials.csv", index=False)
    (study_dir / "best_params.json").write_text(
        json.dumps(_json_ready(best_params), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (study_dir / "study_manifest.json").write_text(
        json.dumps(_json_ready(study_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_tuning(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> TuningResult:
    if config.tuning is None or not config.tuning.enabled:
        raise ValueError("The tune stage requires tuning.enabled=true in config.yaml.")
    if config.tuning.model_id is None:
        raise ValueError("The tune stage requires tuning.model_id.")

    task_type = config.task_type
    primary_metric = config.primary_metric
    tuning_model_spec = TrainingModelSpec(model_id=config.tuning.model_id)
    model_definition = get_model_definition(task_type, tuning_model_spec.model_id)

    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, x_test_raw, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
    )

    positive_label = config.positive_label
    observed_label_pair = None
    negative_label = None
    if task_type == "binary":
        negative_label, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    prepared_context = ensure_prepared_competition_context(
        config=config,
        dataset_context=dataset_context,
        expected_feature_columns=x_train_raw.columns.tolist(),
    )
    split_indices = prepared_context.split_indices
    target_summary = _build_target_summary(
        task_type=task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )

    study_id = _make_study_id()
    study_dir = Path("artifacts") / config.competition_slug / "tune" / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    study_config_snapshot = _build_study_config_snapshot(
        config=config,
        tuning_model_spec=tuning_model_spec,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
    )

    direction = "maximize" if is_higher_better(primary_metric) else "minimize"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=config.tuning.random_state)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_id)

    def objective(trial: optuna.Trial) -> float:
        parameter_overrides = build_tuning_space(task_type, tuning_model_spec.model_id, trial)
        evaluation_artifacts = _evaluate_model_spec(
            task_type=task_type,
            primary_metric=primary_metric,
            model_spec=TrainingModelSpec(
                model_id=tuning_model_spec.model_id,
                parameter_overrides=parameter_overrides,
            ),
            x_train_raw=x_train_raw,
            x_test_raw=x_test_raw,
            y_train=y_train,
            split_indices=split_indices,
            force_categorical=config.force_categorical,
            force_numeric=config.force_numeric,
            low_cardinality_int_threshold=config.low_cardinality_int_threshold,
            cv_random_state=config.cv_random_state,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        metric_mean = evaluation_artifacts.model_result.cv_summary.metric_mean
        metric_std = evaluation_artifacts.model_result.cv_summary.metric_std
        trial.set_user_attr("metric_std", metric_std)
        trial.set_user_attr("model_params", _json_ready(evaluation_artifacts.model_result.model_params))
        print(
            f"Trial {trial.number}: {primary_metric}={metric_mean:.6f} "
            f"(std={metric_std:.6f}) params={parameter_overrides}"
        )
        return metric_mean

    study.optimize(
        objective,
        n_trials=config.tuning.n_trials,
        timeout=config.tuning.timeout_seconds,
        gc_after_trial=True,
    )

    trials_df = _build_trials_df(study=study, metric_name=primary_metric)
    study_manifest = _build_study_manifest(
        config=config,
        study=study,
        study_id=study_id,
        study_config_snapshot=study_config_snapshot,
        tuning_model_spec=tuning_model_spec,
        model_name=model_definition.model_name,
        preprocessing_scheme_id=model_definition.preprocessing_scheme_id,
        target_summary=target_summary,
    )
    _write_tuning_artifacts(study_dir=study_dir, study_manifest=study_manifest, trials_df=trials_df)

    best_parameter_overrides = dict(study.best_trial.params)
    tuning_provenance = {
        "study_id": study_id,
        "study_dir": str(study_dir),
        "trial_number": study.best_trial.number,
        "base_model_id": tuning_model_spec.model_id,
        "parameter_overrides": best_parameter_overrides,
    }
    candidate_dir = run_training(
        config=config,
        dataset_context=dataset_context,
        model_spec=TrainingModelSpec(
            model_id=tuning_model_spec.model_id,
            parameter_overrides=best_parameter_overrides,
        ),
        tuning_provenance=tuning_provenance,
    )

    updated_study_manifest = _build_study_manifest(
        config=config,
        study=study,
        study_id=study_id,
        study_config_snapshot=study_config_snapshot,
        tuning_model_spec=tuning_model_spec,
        model_name=model_definition.model_name,
        preprocessing_scheme_id=model_definition.preprocessing_scheme_id,
        target_summary=target_summary,
        candidate_dir=candidate_dir,
    )
    _write_tuning_artifacts(study_dir=study_dir, study_manifest=updated_study_manifest, trials_df=trials_df)
    print(
        f"Tuning complete: best_trial={study.best_trial.number}, "
        f"best_{primary_metric}={study.best_value:.6f}, candidate={candidate_dir.name}"
    )
    return TuningResult(study_dir=study_dir, candidate_dir=candidate_dir)
