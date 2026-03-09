import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.config import load_config
from tabular_shenanigans.data import fetch_competition_data, load_competition_dataset_context
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.submit import run_submission
from tabular_shenanigans.train import run_training


def main() -> None:
    config = load_config()
    print(
        "Resolved competition setup: "
        f"task_type={config.task_type}, primary_metric={config.primary_metric}, model_ids={config.model_ids}"
    )
    data_dir = fetch_competition_data(config.competition_slug)
    print(f"Data ready: {data_dir}")
    dataset_context = load_competition_dataset_context(
        competition_slug=config.competition_slug,
        configured_id_column=config.id_column,
        configured_label_column=config.label_column,
    )
    report_dir = run_eda(config=config, dataset_context=dataset_context)
    print(f"EDA reports ready: {report_dir}")
    train_dir = run_training(config=config, dataset_context=dataset_context)
    print(f"Training artifacts ready: {train_dir}")
    submission_path, submission_status = run_submission(config=config, run_dir=train_dir)
    print(f"Submission file ready: {submission_path} ({submission_status})")


if __name__ == "__main__":
    main()
