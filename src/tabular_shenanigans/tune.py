import json
from dataclasses import dataclass
from datetime import datetime, timezone

import optuna
import pandas as pd

from tabular_shenanigans.competition import ensure_prepared_competition_context
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better, resolve_positive_label
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.feature_recipes import apply_feature_recipe
from tabular_shenanigans.models import build_tuning_space, get_model_definition
from tabular_shenanigans.preprocess import prepare_feature_frames
from tabular_shenanigans.train import (
    TrainingModelSpec,
    _build_target_summary,
    _evaluate_model_spec,
    _json_ready,
)


@dataclass(frozen=True)
class OptimizationResult:
    best_model_spec: TrainingModelSpec
    tuning_provenance: dict[str, object]
    optimization_summary: dict[str, object]
    trials_df: pd.DataFrame
    best_trial_number: int
    best_value: float


def _make_study_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _build_optimization_config_snapshot(
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


def _build_optimization_summary(
    config: AppConfig,
    study: optuna.Study,
    study_id: str,
    optimization_config_snapshot: dict[str, object],
    tuning_model_spec: TrainingModelSpec,
    model_name: str,
    preprocessing_scheme_id: str,
    target_summary: dict[str, object],
) -> dict[str, object]:
    best_trial = study.best_trial
    return {
        "artifact_type": "candidate_optimization",
        "study_id": study_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": config.competition_slug,
        "candidate_id": config.candidate_id,
        "task_type": config.task_type,
        "primary_metric": config.primary_metric,
        "optimization_method": config.experiment.candidate.optimization.method,
        "optimization_direction": study.direction.name.lower(),
        "config_snapshot": optimization_config_snapshot,
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
    }


def run_optimization(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> OptimizationResult:
    optimization = config.experiment.candidate.optimization
    if not optimization.enabled:
        raise ValueError("Optimization requires experiment.candidate.optimization.enabled=true in config.yaml.")

    task_type = config.task_type
    primary_metric = config.primary_metric
    tuning_model_spec = TrainingModelSpec(model_id=config.resolved_model_id)
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
    x_train_features, x_test_features = apply_feature_recipe(
        recipe_id=config.feature_recipe_id,
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
    )
    target_summary = _build_target_summary(
        task_type=task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )

    study_id = _make_study_id()
    optimization_config_snapshot = _build_optimization_config_snapshot(
        config=config,
        tuning_model_spec=tuning_model_spec,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
    )

    direction = "maximize" if is_higher_better(primary_metric) else "minimize"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=optimization.random_state)
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
            x_train_raw=x_train_features,
            x_test_raw=x_test_features,
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
        n_trials=optimization.n_trials,
        timeout=optimization.timeout_seconds,
        gc_after_trial=True,
    )

    best_parameter_overrides = dict(study.best_trial.params)
    tuning_provenance = {
        "study_id": study_id,
        "optimization_method": optimization.method,
        "trial_number": study.best_trial.number,
        "base_model_id": tuning_model_spec.model_id,
        "parameter_overrides": best_parameter_overrides,
        "best_value": study.best_trial.value,
    }
    optimization_summary = _build_optimization_summary(
        config=config,
        study=study,
        study_id=study_id,
        optimization_config_snapshot=optimization_config_snapshot,
        tuning_model_spec=tuning_model_spec,
        model_name=model_definition.model_name,
        preprocessing_scheme_id=model_definition.preprocessing_scheme_id,
        target_summary=target_summary,
    )
    return OptimizationResult(
        best_model_spec=TrainingModelSpec(
            model_id=tuning_model_spec.model_id,
            parameter_overrides=best_parameter_overrides,
        ),
        tuning_provenance=tuning_provenance,
        optimization_summary=optimization_summary,
        trials_df=_build_trials_df(study=study, metric_name=primary_metric),
        best_trial_number=study.best_trial.number,
        best_value=study.best_trial.value,
    )
