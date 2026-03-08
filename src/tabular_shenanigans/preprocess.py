from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabular_shenanigans.data import find_competition_zip, read_csv_from_zip, resolve_id_and_label_columns


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


def prepare_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    label_column: str,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    force_categorical = force_categorical or []
    force_numeric = force_numeric or []
    drop_columns = drop_columns or []

    available_feature_columns = train_df.drop(columns=[label_column]).columns.tolist()
    _validate_column_names("drop_columns", drop_columns, available_feature_columns)

    excluded_feature_columns = [id_column]
    excluded_feature_columns.extend(
        column for column in drop_columns if column != id_column
    )

    x_train_raw = train_df.drop(columns=[label_column, *excluded_feature_columns])
    y_train = train_df[label_column]
    x_test_raw = test_df.drop(columns=excluded_feature_columns)

    _validate_column_names("force_categorical", force_categorical, x_train_raw.columns.tolist())
    _validate_column_names("force_numeric", force_numeric, x_train_raw.columns.tolist())

    overlap = sorted(set(force_categorical).intersection(force_numeric))
    if overlap:
        raise ValueError(f"Columns cannot be both forced categorical and forced numeric: {overlap}")

    return x_train_raw, x_test_raw, y_train


def build_preprocessor(
    x_train_raw: pd.DataFrame,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    force_categorical = force_categorical or []
    force_numeric = force_numeric or []

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
            (
                "coerce_object",
                FunctionTransformer(
                    lambda frame: frame.astype(object),
                    feature_names_out="one-to-one",
                ),
            ),
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
        raise ValueError("No modeled features remain after excluding id_column and applying drop_columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_columns, categorical_columns


def summarize_feature_types(
    x_train_raw: pd.DataFrame,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> pd.DataFrame:
    force_categorical = force_categorical or []
    force_numeric = force_numeric or []

    numeric_columns, categorical_columns = _resolve_feature_types(
        x_train_raw=x_train_raw,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        low_cardinality_int_threshold=low_cardinality_int_threshold,
    )

    return pd.DataFrame(
        [
            {"feature_type": "numeric", "feature_count": len(numeric_columns)},
            {"feature_type": "categorical", "feature_count": len(categorical_columns)},
            {"feature_type": "total", "feature_count": int(x_train_raw.shape[1])},
        ]
    )


def run_preprocessing(
    competition_slug: str,
    id_column: str | None = None,
    label_column: str | None = None,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    drop_columns: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> Path:
    zip_path = find_competition_zip(competition_slug)
    train_df = read_csv_from_zip(zip_path, "train.csv")
    test_df = read_csv_from_zip(zip_path, "test.csv")
    sample_submission_df = read_csv_from_zip(zip_path, "sample_submission.csv")
    id_column, label_column = resolve_id_and_label_columns(
        train_df=train_df,
        test_df=test_df,
        sample_submission_df=sample_submission_df,
        configured_id_column=id_column,
        configured_label_column=label_column,
    )

    force_categorical = force_categorical or []
    force_numeric = force_numeric or []
    drop_columns = drop_columns or []

    x_train_raw, x_test_raw, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        drop_columns=drop_columns,
    )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(
        x_train_raw=x_train_raw,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        low_cardinality_int_threshold=low_cardinality_int_threshold,
    )

    x_train_processed = preprocessor.fit_transform(x_train_raw)
    x_test_processed = preprocessor.transform(x_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    artifact_dir = Path("artifacts") / competition_slug / "preprocess"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    x_train_df = pd.DataFrame(x_train_processed, columns=feature_names)
    x_test_df = pd.DataFrame(x_test_processed, columns=feature_names)

    x_train_df.to_csv(artifact_dir / "X_train_processed.csv", index=False)
    x_test_df.to_csv(artifact_dir / "X_test_processed.csv", index=False)
    y_train.to_frame(name=label_column).to_csv(artifact_dir / "y_train.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"metric": "id_column", "value": id_column},
            {"metric": "label_column", "value": label_column},
            {"metric": "train_rows", "value": int(x_train_df.shape[0])},
            {"metric": "train_cols", "value": int(x_train_df.shape[1])},
            {"metric": "test_rows", "value": int(x_test_df.shape[0])},
            {"metric": "test_cols", "value": int(x_test_df.shape[1])},
            {"metric": "numeric_feature_count", "value": len(numeric_columns)},
            {"metric": "categorical_feature_count", "value": len(categorical_columns)},
            {"metric": "model_feature_count", "value": int(x_train_raw.shape[1])},
            {"metric": "config_drop_column_count", "value": len(drop_columns)},
            {"metric": "excluded_id_column", "value": id_column},
        ]
    )
    summary_df.to_csv(artifact_dir / "preprocess_summary.csv", index=False)

    print(f"Preprocessed train shape: {x_train_df.shape[0]} rows x {x_train_df.shape[1]} cols")
    print(f"Preprocessed test shape: {x_test_df.shape[0]} rows x {x_test_df.shape[1]} cols")

    return artifact_dir
