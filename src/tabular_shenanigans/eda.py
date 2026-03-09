from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.preprocess import prepare_feature_frames, summarize_feature_types


def _column_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "null_count": [int(df[col].isna().sum()) for col in df.columns],
            "null_pct": [float(df[col].isna().mean()) for col in df.columns],
            "nunique": [int(df[col].nunique(dropna=False)) for col in df.columns],
        }
    )
    return summary


def _missingness_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, df in (("train", train_df), ("test", test_df)):
        for column in df.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "null_count": int(df[column].isna().sum()),
                    "null_pct": float(df[column].isna().mean()),
                }
            )
    return pd.DataFrame(rows, columns=["dataset", "column", "null_count", "null_pct"]).sort_values(
        ["dataset", "null_pct", "column"],
        ascending=[True, False, True],
    )


def _categorical_cardinality_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, df in (("train", train_df), ("test", test_df)):
        categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
        for column in categorical_columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "nunique": int(df[column].nunique(dropna=False)),
                    "null_count": int(df[column].isna().sum()),
                    "null_pct": float(df[column].isna().mean()),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["dataset", "column", "nunique", "null_count", "null_pct"],
    ).sort_values(["dataset", "nunique", "column"], ascending=[True, False, True])


def _target_summary(train_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    target_series = train_df[target_column]

    if pd.api.types.is_numeric_dtype(target_series):
        stats = target_series.describe().to_dict()
        return pd.DataFrame([{"metric": key, "value": float(value)} for key, value in stats.items()])

    value_counts = target_series.value_counts(dropna=False, normalize=False).to_dict()
    total = int(target_series.shape[0])
    return pd.DataFrame(
        [
            {"metric": f"value_{key}_count", "value": int(value)}
            for key, value in value_counts.items()
        ]
        + [
            {"metric": f"value_{key}_ratio", "value": float(value / total)}
            for key, value in value_counts.items()
        ]
    )


def run_eda(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> Path:
    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, _, _ = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
    )

    report_dir = Path("reports") / config.competition_slug
    report_dir.mkdir(parents=True, exist_ok=True)

    _column_summary(train_df).to_csv(report_dir / "columns_train.csv", index=False)
    _column_summary(test_df).to_csv(report_dir / "columns_test.csv", index=False)
    _missingness_summary(train_df, test_df).to_csv(report_dir / "missingness_summary.csv", index=False)
    _categorical_cardinality_summary(train_df, test_df).to_csv(
        report_dir / "categorical_cardinality_summary.csv",
        index=False,
    )
    _target_summary(train_df, label_column).to_csv(report_dir / "target_summary.csv", index=False)
    feature_type_counts = summarize_feature_types(
        x_train_raw=x_train_raw,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        low_cardinality_int_threshold=config.low_cardinality_int_threshold,
    )
    feature_type_counts.to_csv(report_dir / "feature_type_counts.csv", index=False)

    run_summary = pd.DataFrame(
        [
            {"metric": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"metric": "train_rows", "value": int(train_df.shape[0])},
            {"metric": "train_cols", "value": int(train_df.shape[1])},
            {"metric": "test_rows", "value": int(test_df.shape[0])},
            {"metric": "test_cols", "value": int(test_df.shape[1])},
            {"metric": "id_column", "value": id_column},
            {"metric": "label_column", "value": label_column},
            {"metric": "train_missing_pct", "value": float(train_df.isna().mean().mean())},
            {"metric": "test_missing_pct", "value": float(test_df.isna().mean().mean())},
            {"metric": "train_duplicate_rows", "value": int(train_df.duplicated().sum())},
            {"metric": "test_duplicate_rows", "value": int(test_df.duplicated().sum())},
            {"metric": "model_feature_count", "value": int(x_train_raw.shape[1])},
        ]
    )
    run_summary.to_csv(report_dir / "run_summary.csv", index=False)

    print(f"Train shape: {train_df.shape[0]} rows x {train_df.shape[1]} cols")
    print(f"Test shape: {test_df.shape[0]} rows x {test_df.shape[1]} cols")
    print(f"ID column: {id_column}")
    print(f"Label column: {label_column}")
    print(f"Train missing pct: {train_df.isna().mean().mean():.6f}")
    print(f"Test missing pct: {test_df.isna().mean().mean():.6f}")
    print(f"Train duplicate rows: {int(train_df.duplicated().sum())}")
    print(f"Test duplicate rows: {int(test_df.duplicated().sum())}")
    print(f"Modeled feature count: {int(x_train_raw.shape[1])}")

    return report_dir
