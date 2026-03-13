import argparse
from pathlib import Path

from tabular_shenanigans.competition import prepare_competition
from tabular_shenanigans.config import AppConfig, load_config
from tabular_shenanigans.data import fetch_competition_data, load_competition_dataset_context
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.submit import run_submission, run_submission_refresh
from tabular_shenanigans.train import run_training_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TabularShenanigans workflow or one explicit stage.",
    )
    subparsers = parser.add_subparsers(dest="stage")

    subparsers.add_parser("fetch", help="Download competition data if it is missing.")
    subparsers.add_parser(
        "prepare",
        help="Run EDA reports and materialize the in-memory competition context used by train.",
    )
    subparsers.add_parser("eda", help="Run EDA reports only.")
    subparsers.add_parser("train", help="Train the current candidate into MLflow.")
    subparsers.add_parser(
        "refresh-submissions",
        help="Fetch Kaggle submission outcomes for candidate runs tracked in MLflow.",
    )

    submit_parser = subparsers.add_parser("submit", help="Prepare or submit from an MLflow candidate run.")
    submit_parser.add_argument(
        "--candidate-id",
        help=(
            "Optional candidate_id resolved in the current competition MLflow experiment. "
            "Defaults to the derived candidate_id for the current config."
        ),
    )

    return parser


def _print_resolved_setup(config: AppConfig) -> None:
    competition = config.competition
    candidate = config.experiment.candidate
    if config.is_blend_candidate:
        blend_weights = candidate.weights
        weight_summary = "equal-weight"
        if blend_weights is not None:
            weight_summary = ",".join(str(weight) for weight in blend_weights)
        print(
            "Resolved competition setup: "
            f"task_type={competition.task_type}, primary_metric={competition.primary_metric}, "
            f"candidate_id={config.resolved_candidate_id}, candidate_type=blend, "
            f"base_candidates={candidate.base_candidate_ids}, weights={weight_summary}"
        )
        return

    print(
        "Resolved competition setup: "
        f"task_type={competition.task_type}, primary_metric={competition.primary_metric}, "
        f"candidate_id={config.resolved_candidate_id}, candidate_type=model, "
        f"feature_recipe={candidate.feature_recipe_id}, model_family={candidate.model_family}, "
        f"numeric_preprocessor={candidate.numeric_preprocessor}, "
        f"categorical_preprocessor={candidate.categorical_preprocessor}"
    )


def _ensure_data_ready(config: AppConfig) -> Path:
    data_dir = fetch_competition_data(config.competition.slug)
    print(f"Data ready: {data_dir}")
    return data_dir


def _load_shared_dataset_context(config: AppConfig):
    competition = config.competition
    return load_competition_dataset_context(
        competition_slug=competition.slug,
        configured_id_column=competition.id_column,
        configured_label_column=competition.label_column,
    )


def _prepare_competition_stage(
    config: AppConfig,
    dataset_context=None,
):
    if dataset_context is None:
        _ensure_data_ready(config)
        dataset_context = _load_shared_dataset_context(config)
    prepared_context = prepare_competition(config=config, dataset_context=dataset_context)
    print(
        "Competition context ready in memory: "
        f"train_rows={prepared_context.manifest['train_rows']}, "
        f"test_rows={prepared_context.manifest['test_rows']}, "
        f"folds={config.competition.cv.n_splits}"
    )
    if prepared_context.report_dir is not None:
        print(f"EDA reports ready: {prepared_context.report_dir}")
    return dataset_context, prepared_context


def _run_eda_stage(config: AppConfig) -> None:
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    report_dir = run_eda(config=config, dataset_context=dataset_context)
    print(f"EDA reports ready: {report_dir}")


def _run_train_stage(
    config: AppConfig,
    dataset_context=None,
):
    if dataset_context is None:
        _ensure_data_ready(config)
        dataset_context = _load_shared_dataset_context(config)
    candidate_run = run_training_workflow(config=config, dataset_context=dataset_context)
    print(
        "Candidate run ready: "
        f"candidate_id={candidate_run.candidate_id}, mlflow_run_id={candidate_run.run_id}"
    )
    return candidate_run


def _run_submit_stage(
    config: AppConfig,
    candidate_id: str | None = None,
):
    resolved_candidate_id = candidate_id or config.resolved_candidate_id
    print(f"Using candidate_id: {resolved_candidate_id}")
    submission_result = run_submission(
        config=config,
        candidate_id=resolved_candidate_id,
    )
    print(
        "Submission stage complete: "
        f"candidate_id={submission_result.candidate_id}, "
        f"mlflow_run_id={submission_result.candidate_run_id}, "
        f"status={submission_result.submission_status}"
    )
    if submission_result.submission_event_id is not None:
        print(f"Submission event recorded: {submission_result.submission_event_id}")
    if submission_result.submission_artifact_path is not None:
        print(f"Submission artifact path: {submission_result.submission_artifact_path}")
    if submission_result.submission_refresh_result is not None:
        print(
            "Submission score refresh: "
            f"matched_events={submission_result.submission_refresh_result.matched_submission_event_count}, "
            f"updated_candidates={submission_result.submission_refresh_result.updated_candidate_count}, "
            f"appended_observations={submission_result.submission_refresh_result.appended_observation_count}"
        )
    return submission_result


def _run_submission_refresh_stage(config: AppConfig):
    refresh_result = run_submission_refresh(config)
    print(
        "Submission scores refreshed: "
        f"tracked_candidates={refresh_result.tracked_candidate_count}, "
        f"tracked_events={refresh_result.tracked_submission_event_count}, "
        f"matched_events={refresh_result.matched_submission_event_count}, "
        f"updated_candidates={refresh_result.updated_candidate_count}, "
        f"appended_observations={refresh_result.appended_observation_count}, "
        f"scanned_remote_submissions={refresh_result.scanned_remote_submission_count}"
    )
    return refresh_result


def _run_full_pipeline(config: AppConfig) -> None:
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    _prepare_competition_stage(config=config, dataset_context=dataset_context)
    _run_train_stage(config=config, dataset_context=dataset_context)
    _run_submit_stage(config=config)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    _print_resolved_setup(config)

    if args.stage is None:
        _run_full_pipeline(config=config)
        return

    if args.stage == "fetch":
        _ensure_data_ready(config)
        return

    if args.stage == "prepare":
        _prepare_competition_stage(config)
        return

    if args.stage == "eda":
        _run_eda_stage(config)
        return

    if args.stage == "train":
        _run_train_stage(config=config)
        return

    if args.stage == "refresh-submissions":
        _run_submission_refresh_stage(config)
        return

    if args.stage == "submit":
        _run_submit_stage(
            config=config,
            candidate_id=args.candidate_id or config.resolved_candidate_id,
        )
        return

    raise ValueError(f"Unsupported stage: {args.stage}")
