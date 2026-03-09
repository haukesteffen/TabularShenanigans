from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.models import get_model_definition

PreprocessorBuilder = Callable[[list[str], list[str], list[str]], object]


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


class NativeFramePreprocessor:
    CATEGORICAL_MISSING_VALUE = "__missing__"

    def __init__(
        self,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.numeric_fill_values: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "NativeFramePreprocessor":
        if self.numeric_columns:
            self.numeric_fill_values = frame.loc[:, self.numeric_columns].median()
        else:
            self.numeric_fill_values = pd.Series(dtype=float)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed_frame = frame.loc[:, self.feature_columns].copy()

        if self.numeric_columns:
            transformed_frame.loc[:, self.numeric_columns] = transformed_frame.loc[:, self.numeric_columns].fillna(
                self.numeric_fill_values
            )

        if self.categorical_columns:
            transformed_frame = transformed_frame.astype(
                {column: object for column in self.categorical_columns},
                copy=False,
            )
            categorical_frame = transformed_frame.loc[:, self.categorical_columns].astype(object)
            categorical_frame = categorical_frame.where(
                categorical_frame.notna(),
                self.CATEGORICAL_MISSING_VALUE,
            )
            categorical_frame = categorical_frame.astype(str)
            for column in self.categorical_columns:
                transformed_frame.loc[:, column] = categorical_frame.loc[:, column]

        return transformed_frame

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


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
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    del feature_columns
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
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    del feature_columns
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


def _build_native_preprocessor(
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> NativeFramePreprocessor:
    return NativeFramePreprocessor(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


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
    "native": PreprocessingDefinition(
        scheme_id="native",
        scheme_name="NativeFrame",
        builder=_build_native_preprocessor,
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
    preprocessor = preprocessing_definition.builder(
        x_train_raw.columns.tolist(),
        numeric_columns,
        categorical_columns,
    )
    return preprocessor, numeric_columns, categorical_columns


def summarize_feature_types(
    x_train_raw: pd.DataFrame,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> pd.DataFrame:
    force_categorical = force_categorical or []
    force_numeric = force_numeric or []

    numeric_columns, categorical_columns = resolve_feature_types(
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


def _transformed_column_count(transformed_values: object) -> int:
    if isinstance(transformed_values, pd.DataFrame):
        return int(transformed_values.shape[1])
    return int(np.asarray(transformed_values).shape[1])


def _transformed_output_kind(transformed_values: object) -> str:
    if isinstance(transformed_values, pd.DataFrame):
        return "dataframe"
    if isinstance(transformed_values, np.ndarray):
        return "ndarray"
    return type(transformed_values).__name__


def run_preprocess(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> Path:
    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, x_test_raw, _ = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
    )
    numeric_columns, categorical_columns = resolve_feature_types(
        x_train_raw=x_train_raw,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        low_cardinality_int_threshold=config.low_cardinality_int_threshold,
    )

    report_dir = Path("reports") / config.competition_slug
    report_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, object]] = []
    forced_categorical = set(config.force_categorical)
    forced_numeric = set(config.force_numeric)
    numeric_column_set = set(numeric_columns)
    for column in x_train_raw.columns:
        forced_feature_type = ""
        if column in forced_categorical:
            forced_feature_type = "categorical"
        elif column in forced_numeric:
            forced_feature_type = "numeric"
        inferred_feature_type = "numeric" if column in numeric_column_set else "categorical"
        feature_rows.append(
            {
                "feature_name": column,
                "train_dtype": str(x_train_raw[column].dtype),
                "test_dtype": str(x_test_raw[column].dtype),
                "train_null_pct": float(x_train_raw[column].isna().mean()),
                "test_null_pct": float(x_test_raw[column].isna().mean()),
                "train_nunique": int(x_train_raw[column].nunique(dropna=False)),
                "forced_feature_type": forced_feature_type,
                "inferred_feature_type": inferred_feature_type,
            }
        )
    feature_details_df = pd.DataFrame(feature_rows)
    feature_details_df.to_csv(report_dir / "preprocess_features.csv", index=False)

    model_rows: list[dict[str, object]] = []
    for model_id in config.model_ids:
        model_definition = get_model_definition(config.task_type, model_id)
        preprocessing_definition = get_preprocessing_definition(model_definition.preprocessing_scheme_id)
        preprocessor, model_numeric_columns, model_categorical_columns = build_preprocessor(
            scheme_id=model_definition.preprocessing_scheme_id,
            x_train_raw=x_train_raw,
            force_categorical=config.force_categorical,
            force_numeric=config.force_numeric,
            low_cardinality_int_threshold=config.low_cardinality_int_threshold,
        )
        if model_definition.model_name.startswith("LGBM") and hasattr(preprocessor, "set_output"):
            preprocessor.set_output(transform="pandas")
        x_train_processed = preprocessor.fit_transform(x_train_raw)
        x_test_processed = preprocessor.transform(x_test_raw)
        model_rows.append(
            {
                "model_id": model_definition.model_id,
                "model_name": model_definition.model_name,
                "preprocessing_scheme_id": preprocessing_definition.scheme_id,
                "preprocessing_scheme_name": preprocessing_definition.scheme_name,
                "numeric_feature_count": len(model_numeric_columns),
                "categorical_feature_count": len(model_categorical_columns),
                "processed_train_rows": int(x_train_raw.shape[0]),
                "processed_train_cols": _transformed_column_count(x_train_processed),
                "processed_test_rows": int(x_test_raw.shape[0]),
                "processed_test_cols": _transformed_column_count(x_test_processed),
                "output_kind": _transformed_output_kind(x_train_processed),
            }
        )
    model_preprocess_df = pd.DataFrame(model_rows)
    model_preprocess_df.to_csv(report_dir / "preprocess_models.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"metric": "id_column", "value": id_column},
            {"metric": "label_column", "value": label_column},
            {"metric": "model_count", "value": len(config.model_ids)},
            {"metric": "modeled_feature_count", "value": int(x_train_raw.shape[1])},
            {"metric": "numeric_feature_count", "value": len(numeric_columns)},
            {"metric": "categorical_feature_count", "value": len(categorical_columns)},
        ]
    )
    summary_df.to_csv(report_dir / "preprocess_summary.csv", index=False)

    print(f"Preprocess feature count: {int(x_train_raw.shape[1])}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Model-specific preprocess summaries: {len(model_rows)}")

    return report_dir
