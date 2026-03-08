from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

PreprocessorBuilder = Callable[[list[str], list[str]], ColumnTransformer]


@dataclass(frozen=True)
class PreprocessingDefinition:
    scheme_id: str
    scheme_name: str
    builder: PreprocessorBuilder


def _validate_column_names(config_name: str, columns: list[str], available_columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in available_columns]
    if missing_columns:
        raise ValueError(f"{config_name} has unknown columns: {missing_columns}")


def _coerce_object(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object)


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


def _build_linear_onehot_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("coerce_object", FunctionTransformer(_coerce_object, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_ordinal_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("coerce_object", FunctionTransformer(_coerce_object, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


PREPROCESSING_REGISTRY = {
    "onehot": PreprocessingDefinition(
        scheme_id="onehot",
        scheme_name="OneHotLinear",
        builder=_build_linear_onehot_preprocessor,
    ),
    "ordinal": PreprocessingDefinition(
        scheme_id="ordinal",
        scheme_name="OrdinalTree",
        builder=_build_ordinal_preprocessor,
    ),
}


def get_preprocessing_definition(scheme_id: str) -> PreprocessingDefinition:
    try:
        return PREPROCESSING_REGISTRY[scheme_id]
    except KeyError as exc:
        supported_scheme_ids = sorted(PREPROCESSING_REGISTRY)
        raise ValueError(
            f"Unsupported preprocessing scheme '{scheme_id}'. Supported values: {supported_scheme_ids}"
        ) from exc


def build_preprocessor(
    scheme_id: str,
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

    if not numeric_columns and not categorical_columns:
        raise ValueError("No modeled features remain after excluding id_column and applying drop_columns.")

    preprocessing_definition = get_preprocessing_definition(scheme_id)
    preprocessor = preprocessing_definition.builder(numeric_columns, categorical_columns)
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
