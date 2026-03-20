from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResolvedFeatureSchema:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


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


def resolve_feature_types(
    x_train_raw: pd.DataFrame,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> tuple[list[str], list[str]]:
    return _resolve_feature_types(
        x_train_raw=x_train_raw,
        force_categorical=force_categorical or [],
        force_numeric=force_numeric or [],
        low_cardinality_int_threshold=low_cardinality_int_threshold,
    )


def resolve_feature_schema(
    x_train_raw: pd.DataFrame,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> ResolvedFeatureSchema:
    numeric_columns, categorical_columns = resolve_feature_types(
        x_train_raw=x_train_raw,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        low_cardinality_int_threshold=low_cardinality_int_threshold,
    )
    return ResolvedFeatureSchema(
        feature_columns=x_train_raw.columns.tolist(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
