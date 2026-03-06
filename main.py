import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.config import load_config
from tabular_shenanigans.data import fetch_competition_data
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.preprocess import run_preprocessing
from tabular_shenanigans.submit import run_submission
from tabular_shenanigans.train import run_training


def main() -> None:
    config = load_config()
    print(
        "Resolved competition setup: "
        f"task_type={config.task_type}, primary_metric={config.primary_metric}"
    )
    data_dir = fetch_competition_data(config.competition_slug)
    print(f"Data ready: {data_dir}")
    report_dir = run_eda(
        competition_slug=config.competition_slug,
        id_column=config.id_column,
        label_column=config.label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
        low_cardinality_int_threshold=config.low_cardinality_int_threshold,
    )
    print(f"EDA reports ready: {report_dir}")
    artifact_dir = run_preprocessing(
        competition_slug=config.competition_slug,
        id_column=config.id_column,
        label_column=config.label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
        low_cardinality_int_threshold=config.low_cardinality_int_threshold,
    )
    print(f"Preprocessing artifacts ready: {artifact_dir}")
    train_dir = run_training(
        competition_slug=config.competition_slug,
        task_type=config.task_type,
        primary_metric=config.primary_metric,
        id_column=config.id_column,
        label_column=config.label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
        low_cardinality_int_threshold=config.low_cardinality_int_threshold,
        cv_n_splits=config.cv_n_splits,
        cv_shuffle=config.cv_shuffle,
        cv_random_state=config.cv_random_state,
    )
    print(f"Training artifacts ready: {train_dir}")
    submission_path, submission_status = run_submission(
        competition_slug=config.competition_slug,
        run_dir=train_dir,
        submit_enabled=config.submit_enabled,
        submit_message_prefix=config.submit_message_prefix,
        id_column=config.id_column,
        label_column=config.label_column,
    )
    print(f"Submission file ready: {submission_path} ({submission_status})")


if __name__ == "__main__":
    main()
