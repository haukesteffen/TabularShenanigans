from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tabular_shenanigans.data import find_competition_zip, infer_target_column, read_csv_from_zip


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


def run_eda(competition_slug: str) -> Path:
    zip_path = find_competition_zip(competition_slug)
    train_df = read_csv_from_zip(zip_path, "train.csv")
    test_df = read_csv_from_zip(zip_path, "test.csv")
    target_column = infer_target_column(train_df, test_df)

    report_dir = Path("reports") / competition_slug
    report_dir.mkdir(parents=True, exist_ok=True)

    _column_summary(train_df).to_csv(report_dir / "columns_train.csv", index=False)
    _column_summary(test_df).to_csv(report_dir / "columns_test.csv", index=False)
    _target_summary(train_df, target_column).to_csv(report_dir / "target_summary.csv", index=False)

    run_summary = pd.DataFrame(
        [
            {"metric": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"metric": "train_rows", "value": int(train_df.shape[0])},
            {"metric": "train_cols", "value": int(train_df.shape[1])},
            {"metric": "test_rows", "value": int(test_df.shape[0])},
            {"metric": "test_cols", "value": int(test_df.shape[1])},
            {"metric": "target_column", "value": target_column},
            {"metric": "train_missing_pct", "value": float(train_df.isna().mean().mean())},
            {"metric": "test_missing_pct", "value": float(test_df.isna().mean().mean())},
            {"metric": "train_duplicate_rows", "value": int(train_df.duplicated().sum())},
            {"metric": "test_duplicate_rows", "value": int(test_df.duplicated().sum())},
        ]
    )
    run_summary.to_csv(report_dir / "run_summary.csv", index=False)

    print(f"Train shape: {train_df.shape[0]} rows x {train_df.shape[1]} cols")
    print(f"Test shape: {test_df.shape[0]} rows x {test_df.shape[1]} cols")
    print(f"Target column: {target_column}")
    print(f"Train missing pct: {train_df.isna().mean().mean():.6f}")
    print(f"Test missing pct: {test_df.isna().mean().mean():.6f}")
    print(f"Train duplicate rows: {int(train_df.duplicated().sum())}")
    print(f"Test duplicate rows: {int(test_df.duplicated().sum())}")

    return report_dir
