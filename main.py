import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.competition import prepare_competition
from tabular_shenanigans.config import AppConfig, load_config
from tabular_shenanigans.data import fetch_competition_data, load_competition_dataset_context
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.preprocess import run_preprocess
from tabular_shenanigans.submit import run_submission
from tabular_shenanigans.tune import run_tuning
from tabular_shenanigans.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TabularShenanigans workflow or one explicit stage.",
    )
    subparsers = parser.add_subparsers(dest="stage")

    subparsers.add_parser("fetch", help="Download competition data if it is missing.")
    subparsers.add_parser("prepare", help="Persist EDA reports, competition metadata, and frozen folds.")
    subparsers.add_parser("eda", help="Run EDA reports only.")
    subparsers.add_parser("preprocess", help="Write preprocessing diagnostics only.")
    subparsers.add_parser("train", help="Run training only.")
    subparsers.add_parser("tune", help="Run Optuna tuning and retrain the best trial.")

    submit_parser = subparsers.add_parser("submit", help="Prepare or submit from an existing run artifact.")
    run_selection = submit_parser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument(
        "--run-dir",
        type=Path,
        help="Explicit run directory such as artifacts/<competition_slug>/train/<run_id>.",
    )
    run_selection.add_argument(
        "--run-id",
        help="Run identifier resolved under artifacts/<competition_slug>/train/<run_id> using config.competition_slug.",
    )
    submit_parser.add_argument(
        "--model-id",
        help="Optional model_id to submit from a multi-model run; defaults to the run manifest best_model_id.",
    )

    return parser


def _print_resolved_setup(config: AppConfig) -> None:
    print(
        "Resolved competition setup: "
        f"task_type={config.task_type}, primary_metric={config.primary_metric}, "
        f"candidate_id={config.candidate_id}, model_family={config.model_family}, "
        f"preprocessor={config.preprocessor}, model_id={config.resolved_model_id}"
    )


def _ensure_data_ready(config: AppConfig) -> Path:
    data_dir = fetch_competition_data(config.competition_slug)
    print(f"Data ready: {data_dir}")
    return data_dir


def _load_shared_dataset_context(config: AppConfig):
    return load_competition_dataset_context(
        competition_slug=config.competition_slug,
        configured_id_column=config.id_column,
        configured_label_column=config.label_column,
    )


def _resolve_run_dir(config: AppConfig, args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    return Path("artifacts") / config.competition_slug / "train" / str(args.run_id)


def _prepare_competition_stage(config: AppConfig):
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    prepared_context = prepare_competition(config=config, dataset_context=dataset_context)
    print(f"Competition context ready: {prepared_context.manifest_path}")
    print(f"Frozen folds ready: {prepared_context.folds_path}")
    print(f"EDA reports ready: {prepared_context.report_dir}")
    return dataset_context, prepared_context


def _run_full_pipeline(config: AppConfig) -> None:
    dataset_context, _ = _prepare_competition_stage(config)
    train_dir = run_training(config=config, dataset_context=dataset_context)
    print(f"Training artifacts ready: {train_dir}")
    submission_path, submission_status = run_submission(config=config, run_dir=train_dir)
    print(f"Submission file ready: {submission_path} ({submission_status})")


def _run_eda_stage(config: AppConfig) -> None:
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    report_dir = run_eda(config=config, dataset_context=dataset_context)
    print(f"EDA reports ready: {report_dir}")


def _run_preprocess_stage(config: AppConfig) -> None:
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    report_dir = run_preprocess(config=config, dataset_context=dataset_context)
    print(f"Preprocess reports ready: {report_dir}")


def _run_train_stage(config: AppConfig) -> None:
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    train_dir = run_training(config=config, dataset_context=dataset_context)
    print(f"Training artifacts ready: {train_dir}")


def _run_tune_stage(config: AppConfig) -> None:
    _ensure_data_ready(config)
    dataset_context = _load_shared_dataset_context(config)
    tuning_result = run_tuning(config=config, dataset_context=dataset_context)
    print(f"Tuning artifacts ready: {tuning_result.study_dir}")
    print(f"Best-trial training artifacts ready: {tuning_result.train_dir}")


def _run_submit_stage(config: AppConfig, args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(config, args)
    print(f"Using run_dir: {run_dir}")
    submission_path, submission_status = run_submission(
        config=config,
        run_dir=run_dir,
        model_id=args.model_id,
    )
    print(f"Submission file ready: {submission_path} ({submission_status})")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    _print_resolved_setup(config)

    if args.stage is None:
        _run_full_pipeline(config)
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

    if args.stage == "preprocess":
        _run_preprocess_stage(config)
        return

    if args.stage == "train":
        _run_train_stage(config)
        return

    if args.stage == "tune":
        _run_tune_stage(config)
        return

    if args.stage == "submit":
        _run_submit_stage(config, args)
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main(sys.argv[1:])
