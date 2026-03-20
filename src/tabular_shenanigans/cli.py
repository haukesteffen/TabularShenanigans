import argparse
from pathlib import Path

from dotenv import load_dotenv

from tabular_shenanigans.competition import prepare_competition
from tabular_shenanigans.config import AppConfig, load_config
from tabular_shenanigans.data import fetch_competition_data, load_competition_dataset_context
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.submit import run_submission, run_submission_refresh
from tabular_shenanigans.training_orchestration import (
    TrainingBatchSummary,
    print_training_batch_summary,
    run_training_batch,
)


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

    train_parser = subparsers.add_parser("train", help="Train configured candidates into MLflow.")
    train_selector_group = train_parser.add_mutually_exclusive_group()
    train_selector_group.add_argument(
        "--candidate-id",
        help="Optional configured candidate_id from config.yaml. Omit to train all configured candidates.",
    )
    train_selector_group.add_argument(
        "--index",
        type=int,
        help="Optional 1-based configured candidate index from config.yaml.",
    )
    train_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configured candidates that already exist in MLflow.",
    )

    subparsers.add_parser(
        "refresh-submissions",
        help="Fetch Kaggle submission outcomes for candidate runs tracked in MLflow.",
    )

    submit_parser = subparsers.add_parser("submit", help="Validate or submit a candidate to Kaggle.")
    submit_selector_group = submit_parser.add_mutually_exclusive_group()
    submit_selector_group.add_argument(
        "--candidate-id",
        help="Existing candidate_id in the current competition MLflow experiment.",
    )
    submit_selector_group.add_argument(
        "--index",
        type=int,
        help="1-based configured candidate index from config.yaml. Defaults to 1 when no selector is given.",
    )
    submit_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute a real Kaggle submission. Without this flag, submit performs dry-run validation only.",
    )
    submit_parser.add_argument(
        "--message-prefix",
        help="Optional prefix for the Kaggle submission description. Only used with --execute.",
    )

    return parser


def _print_candidate_setup(candidate_config: AppConfig, candidate_index: int) -> None:
    competition = candidate_config.competition
    candidate = candidate_config.experiment.candidate
    prefix = f"  [{candidate_index}] "
    if candidate_config.is_blend_candidate:
        blend_weights = candidate.weights
        weight_summary = "equal-weight"
        if blend_weights is not None:
            weight_summary = ",".join(str(weight) for weight in blend_weights)
        print(
            f"{prefix}candidate_id={candidate_config.resolved_candidate_id}, "
            f"candidate_type=blend, "
            f"task_type={competition.task_type}, "
            f"primary_metric={competition.primary_metric}, "
            f"base_candidates={candidate.base_candidate_ids}, "
            f"weights={weight_summary}"
        )
        return

    print(
        f"{prefix}candidate_id={candidate_config.resolved_candidate_id}, "
        f"candidate_type=model, "
        f"task_type={competition.task_type}, "
        f"primary_metric={competition.primary_metric}, "
        f"representation={candidate.representation_id}, "
        f"model_family={candidate.model_family}"
    )


def _print_resolved_setup(config: AppConfig) -> None:
    competition = config.competition
    print(
        "Resolved competition setup: "
        f"slug={competition.slug}, "
        f"task_type={competition.task_type}, "
        f"primary_metric={competition.primary_metric}, "
        f"configured_candidates={config.candidate_count}"
    )
    for candidate_index in range(config.candidate_count):
        _print_candidate_setup(
            candidate_config=config.with_candidate_index(candidate_index),
            candidate_index=candidate_index + 1,
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
    candidate_id: str | None = None,
    index: int | None = None,
    skip_existing: bool = False,
) -> TrainingBatchSummary:
    if dataset_context is None:
        _ensure_data_ready(config)
        dataset_context = _load_shared_dataset_context(config)
    batch_summary = run_training_batch(
        config=config,
        dataset_context=dataset_context,
        candidate_id=candidate_id,
        index=index,
        skip_existing=skip_existing,
    )
    print_training_batch_summary(batch_summary)
    if batch_summary.failed_count > 0:
        raise RuntimeError(
            f"{batch_summary.failed_count} candidate(s) failed during train. See the batch summary above."
        )
    return batch_summary


def _resolve_submit_candidate_id(
    config: AppConfig,
    candidate_id: str | None = None,
    index: int | None = None,
) -> str:
    if candidate_id is not None:
        return candidate_id

    effective_index = index if index is not None else 1
    selected_candidate_indices = config.resolve_candidate_indices(
        index=effective_index,
        require_explicit=True,
    )
    return config.with_candidate_index(selected_candidate_indices[0]).resolved_candidate_id


def _run_submit_stage(
    config: AppConfig,
    candidate_id: str | None = None,
    index: int | None = None,
    execute: bool = False,
    message_prefix: str | None = None,
):
    resolved_candidate_id = _resolve_submit_candidate_id(
        config=config,
        candidate_id=candidate_id,
        index=index,
    )
    print(f"Using candidate_id: {resolved_candidate_id}")
    submission_result = run_submission(
        config=config,
        candidate_id=resolved_candidate_id,
        execute=execute,
        message_prefix=message_prefix,
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
    print(
        "Default pipeline stops after train. "
        "Run `uv run python main.py submit --candidate-id <candidate_id>` or "
        "`uv run python main.py submit --index <n>` explicitly to submit."
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(dotenv_path=Path(".env"), override=False)
    config = load_config()
    if config.experiment.legacy_candidate_contract_used:
        print(
            "Config deprecation: experiment.candidate is deprecated. "
            "Migrate config.yaml to experiment.candidates."
        )
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
        _run_train_stage(
            config=config,
            candidate_id=args.candidate_id,
            index=args.index,
            skip_existing=args.skip_existing,
        )
        return

    if args.stage == "refresh-submissions":
        _run_submission_refresh_stage(config)
        return

    if args.stage == "submit":
        _run_submit_stage(
            config=config,
            candidate_id=args.candidate_id,
            index=args.index,
            execute=args.execute,
            message_prefix=args.message_prefix,
        )
        return

    raise ValueError(f"Unsupported stage: {args.stage}")
