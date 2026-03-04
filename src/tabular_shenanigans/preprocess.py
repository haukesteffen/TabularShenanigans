from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabular_shenanigans.data import find_competition_zip, infer_target_column, read_csv_from_zip


def _validate_column_names(config_name: str, columns: list[str], available_columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in available_columns]
    if missing_columns:
        raise ValueError(f"{config_name} has unknown columns: {missing_columns}")


def _resolve_feature_types(
    x_train_raw: pd.DataFrame,
    force_categorical: list[str],
    force_numeric: list[str],
    low_cardinality_int_threshold: int | None,
) -> tuple[list[str], list[str]]:
    all_columns = x_train_raw.columns.tolist()
    numeric_columns = set(x_train_raw.select_dtypes(include=["number"]).columns.tolist())

    if low_cardinality_int_threshold is not None:
        for column in all_columns:
            if column not in numeric_columns:
                continue
            column_series = x_train_raw[column]
            if pd.api.types.is_integer_dtype(column_series):
                unique_count = int(column_series.nunique(dropna=True))
                if unique_count <= low_cardinality_int_threshold:
                    numeric_columns.remove(column)

    numeric_columns.difference_update(force_categorical)
    numeric_columns.update(force_numeric)

    ordered_numeric_columns = [column for column in all_columns if column in numeric_columns]
    categorical_columns = [column for column in all_columns if column not in numeric_columns]
    return ordered_numeric_columns, categorical_columns


def run_preprocessing(
    competition_slug: str,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    drop_columns: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> Path:
    zip_path = find_competition_zip(competition_slug)
    train_df = read_csv_from_zip(zip_path, "train.csv")
    test_df = read_csv_from_zip(zip_path, "test.csv")
    target_column = infer_target_column(train_df, test_df)

    force_categorical = force_categorical or []
    force_numeric = force_numeric or []
    drop_columns = drop_columns or []

    x_train_raw = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    x_test_raw = test_df

    _validate_column_names("drop_columns", drop_columns, x_train_raw.columns.tolist())
    _validate_column_names("force_categorical", force_categorical, x_train_raw.columns.tolist())
    _validate_column_names("force_numeric", force_numeric, x_train_raw.columns.tolist())

    overlap = sorted(set(force_categorical).intersection(force_numeric))
    if overlap:
        raise ValueError(f"Columns cannot be both forced categorical and forced numeric: {overlap}")

    if drop_columns:
        x_train_raw = x_train_raw.drop(columns=drop_columns)
        x_test_raw = x_test_raw.drop(columns=drop_columns)

    numeric_columns, categorical_columns = _resolve_feature_types(
        x_train_raw=x_train_raw,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        low_cardinality_int_threshold=low_cardinality_int_threshold,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No features remain after applying drop_columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    x_train_processed = preprocessor.fit_transform(x_train_raw)
    x_test_processed = preprocessor.transform(x_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    artifact_dir = Path("artifacts") / competition_slug / "preprocess"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    x_train_df = pd.DataFrame(x_train_processed, columns=feature_names)
    x_test_df = pd.DataFrame(x_test_processed, columns=feature_names)

    x_train_df.to_csv(artifact_dir / "X_train_processed.csv", index=False)
    x_test_df.to_csv(artifact_dir / "X_test_processed.csv", index=False)
    y_train.to_frame(name=target_column).to_csv(artifact_dir / "y_train.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"metric": "target_column", "value": target_column},
            {"metric": "train_rows", "value": int(x_train_df.shape[0])},
            {"metric": "train_cols", "value": int(x_train_df.shape[1])},
            {"metric": "test_rows", "value": int(x_test_df.shape[0])},
            {"metric": "test_cols", "value": int(x_test_df.shape[1])},
            {"metric": "numeric_feature_count", "value": len(numeric_columns)},
            {"metric": "categorical_feature_count", "value": len(categorical_columns)},
            {"metric": "dropped_feature_count", "value": len(drop_columns)},
        ]
    )
    summary_df.to_csv(artifact_dir / "preprocess_summary.csv", index=False)

    print(f"Preprocessed train shape: {x_train_df.shape[0]} rows x {x_train_df.shape[1]} cols")
    print(f"Preprocessed test shape: {x_test_df.shape[0]} rows x {x_test_df.shape[1]} cols")

    return artifact_dir
