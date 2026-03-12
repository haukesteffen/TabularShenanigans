import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.competition import prepare_competition
from tabular_shenanigans.config import AppConfig, load_config
from tabular_shenanigans.data import fetch_competition_data, load_competition_dataset_context
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.submit import run_submission, run_submission_refresh
from tabular_shenanigans.tracking import (
    build_train_tracking_tags,
    log_prepare_stage_outputs,
    log_submission_refresh_outputs,
    log_submit_outputs,
    log_train_outputs,
    make_pipeline_invocation_id,
    run_stage_with_tracking,
)
from tabular_shenanigans.train import run_training_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TabularShenanigans workflow or one explicit stage.",
    )
    subparsers = parser.add_subparsers(dest="stage")

    subparsers.add_parser("fetch", help="Download competition data if it is missing.")
    subparsers.add_parser("prepare", help="Persist EDA reports, competition metadata, and frozen folds.")
    subparsers.add_parser("eda", help="Run EDA reports only.")
    subparsers.add_parser("train", help="Train the current candidate, with optional optimization.")
    subparsers.add_parser(
        "refresh-submissions",
        help="Fetch Kaggle submission outcomes for locally tracked submission events.",
    )

    submit_parser = subparsers.add_parser("submit", help="Prepare or submit from a candidate artifact.")
    submit_parser.add_argument(
        "--candidate-id",
        help=(
            "Optional candidate_id resolved under artifacts/<competition_slug>/candidates/<candidate_id>. "
            "Defaults to config.experiment.candidate.candidate_id."
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
            f"candidate_id={candidate.candidate_id}, candidate_type=blend, "
            f"base_candidates={candidate.base_candidate_ids}, weights={weight_summary}"
        )
        return

    print(
        "Resolved competition setup: "
        f"task_type={competition.task_type}, primary_metric={competition.primary_metric}, "
        f"candidate_id={candidate.candidate_id}, candidate_type=model, "
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


def _prepare_competition_stage(config: AppConfig):
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    prepared_context = prepare_competition(config=config, dataset_context=dataset_context)
    print(f"Competition context ready: {prepared_context.manifest_path}")
    print(f"Frozen folds ready: {prepared_context.folds_path}")
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
    candidate_dir = run_training_workflow(config=config, dataset_context=dataset_context)
    print(f"Candidate artifacts ready: {candidate_dir}")
    return candidate_dir


def _run_submit_stage(
    config: AppConfig,
    candidate_id: str | None = None,
):
    resolved_candidate_id = candidate_id or config.experiment.candidate.candidate_id
    print(f"Using candidate_id: {resolved_candidate_id}")
    submission_result = run_submission(
        config=config,
        candidate_id=resolved_candidate_id,
    )
    print(f"Submission file ready: {submission_result.submission_path} ({submission_result.submission_status})")
    if submission_result.submission_event is not None:
        print(
            "Submission event recorded: "
            f"{submission_result.submission_event.submission_event_id}"
        )
    if submission_result.submission_refresh_result is not None:
        print(
            "Submission score refresh: "
            f"matched_events={submission_result.submission_refresh_result.matched_submission_event_count}, "
            f"appended_observations={submission_result.submission_refresh_result.appended_observation_count}"
        )
    return submission_result


def _run_submission_refresh_stage(config: AppConfig):
    refresh_result = run_submission_refresh(config)
    print(
        "Submission scores refreshed: "
        f"tracked_events={refresh_result.tracked_submission_event_count}, "
        f"matched_events={refresh_result.matched_submission_event_count}, "
        f"appended_observations={refresh_result.appended_observation_count}, "
        f"scanned_remote_submissions={refresh_result.scanned_remote_submission_count}"
    )
    return refresh_result


def _run_full_pipeline(
    config: AppConfig,
    pipeline_invocation_id: str,
) -> None:
    dataset_context, _ = run_stage_with_tracking(
        config=config,
        stage="prepare",
        pipeline_invocation_id=pipeline_invocation_id,
        stage_fn=lambda: _prepare_competition_stage(config),
        result_logger=log_prepare_stage_outputs,
    )
    run_stage_with_tracking(
        config=config,
        stage="train",
        pipeline_invocation_id=pipeline_invocation_id,
        stage_fn=lambda: _run_train_stage(config=config, dataset_context=dataset_context),
        extra_tags=build_train_tracking_tags(config),
        result_logger=log_train_outputs,
    )
    run_stage_with_tracking(
        config=config,
        stage="submit",
        pipeline_invocation_id=pipeline_invocation_id,
        stage_fn=lambda: _run_submit_stage(config=config),
        extra_tags={"candidate_id": config.experiment.candidate.candidate_id},
        result_logger=log_submit_outputs,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    pipeline_invocation_id = make_pipeline_invocation_id()
    _print_resolved_setup(config)

    if args.stage is None:
        _run_full_pipeline(config=config, pipeline_invocation_id=pipeline_invocation_id)
        return

    if args.stage == "fetch":
        _ensure_data_ready(config)
        return

    if args.stage == "prepare":
        run_stage_with_tracking(
            config=config,
            stage="prepare",
            pipeline_invocation_id=pipeline_invocation_id,
            stage_fn=lambda: _prepare_competition_stage(config),
            result_logger=log_prepare_stage_outputs,
        )
        return

    if args.stage == "eda":
        _run_eda_stage(config)
        return

    if args.stage == "train":
        run_stage_with_tracking(
            config=config,
            stage="train",
            pipeline_invocation_id=pipeline_invocation_id,
            stage_fn=lambda: _run_train_stage(config=config),
            extra_tags=build_train_tracking_tags(config),
            result_logger=log_train_outputs,
        )
        return

    if args.stage == "refresh-submissions":
        run_stage_with_tracking(
            config=config,
            stage="refresh-submissions",
            pipeline_invocation_id=pipeline_invocation_id,
            stage_fn=lambda: _run_submission_refresh_stage(config),
            result_logger=log_submission_refresh_outputs,
        )
        return

    if args.stage == "submit":
        resolved_candidate_id = args.candidate_id or config.experiment.candidate.candidate_id
        run_stage_with_tracking(
            config=config,
            stage="submit",
            pipeline_invocation_id=pipeline_invocation_id,
            stage_fn=lambda: _run_submit_stage(
                config=config,
                candidate_id=resolved_candidate_id,
            ),
            extra_tags={"candidate_id": resolved_candidate_id},
            result_logger=log_submit_outputs,
        )
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main(sys.argv[1:])
