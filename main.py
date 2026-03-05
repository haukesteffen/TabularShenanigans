import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.config import load_config
from tabular_shenanigans.data import fetch_competition_data, resolve_task_type_and_primary_metric
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.preprocess import run_preprocessing


def main() -> None:
    config = load_config()
    task_type, primary_metric, provenance = resolve_task_type_and_primary_metric(
        competition_slug=config.competition_slug,
        configured_task_type=config.task_type,
        configured_primary_metric=config.primary_metric,
    )
    print(
        "Resolved competition setup: "
        f"task_type={task_type}, primary_metric={primary_metric}, source={provenance}"
    )
    data_dir = fetch_competition_data(config.competition_slug)
    print(f"Data ready: {data_dir}")
    report_dir = run_eda(config.competition_slug)
    print(f"EDA reports ready: {report_dir}")
    artifact_dir = run_preprocessing(
        competition_slug=config.competition_slug,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
        low_cardinality_int_threshold=config.low_cardinality_int_threshold,
    )
    print(f"Preprocessing artifacts ready: {artifact_dir}")


if __name__ == "__main__":
    main()
