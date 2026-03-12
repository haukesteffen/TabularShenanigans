from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

NumericPreprocessorBuilder = Callable[[], object]
CategoricalPreprocessorBuilder = Callable[[], object]

LEGACY_PREPROCESSOR_MAPPING = {
    "onehot": ("standardize", "onehot"),
    "ordinal": ("median", "ordinal"),
    "frequency": ("median", "frequency"),
    "native": ("median", "native"),
}
NUMERIC_PREPROCESSOR_IDS = tuple(["median", "standardize", "kbins"])
CATEGORICAL_PREPROCESSOR_IDS = tuple(["onehot", "ordinal", "frequency", "native"])


@dataclass(frozen=True)
class NumericPreprocessingDefinition:
    scheme_id: str
    scheme_name: str
    builder: NumericPreprocessorBuilder


@dataclass(frozen=True)
class CategoricalPreprocessingDefinition:
    scheme_id: str
    scheme_name: str
    compose_mode: str
    builder: CategoricalPreprocessorBuilder | None = None


@dataclass(frozen=True)
class ResolvedFeatureSchema:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


def _validate_column_names(config_name: str, columns: list[str], available_columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in available_columns]
    if missing_columns:
        raise ValueError(f"{config_name} has unknown columns: {missing_columns}")


def _coerce_object(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object)


def _normalize_categorical_series(series: pd.Series, missing_value: str) -> pd.Series:
    return series.astype(object).where(series.notna(), missing_value).astype(str)


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


def resolve_legacy_preprocessor_selection(preprocessor_id: str) -> tuple[str, str]:
    try:
        return LEGACY_PREPROCESSOR_MAPPING[preprocessor_id]
    except KeyError as exc:
        supported_preprocessor_ids = sorted(LEGACY_PREPROCESSOR_MAPPING)
        raise ValueError(
            f"Unsupported preprocessing scheme '{preprocessor_id}'. Supported values: {supported_preprocessor_ids}"
        ) from exc


def build_preprocessing_scheme_id(
    numeric_preprocessor_id: str,
    categorical_preprocessor_id: str,
) -> str:
    return f"num_{numeric_preprocessor_id}__cat_{categorical_preprocessor_id}"


def categorical_preprocessor_uses_native_columns(categorical_preprocessor_id: str) -> bool:
    return categorical_preprocessor_id == "native"


def _ensure_dense_array(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def _resolve_transformed_numeric_columns(
    numeric_preprocessor: object,
    numeric_columns: list[str],
) -> list[str]:
    if not numeric_columns:
        return []
    if hasattr(numeric_preprocessor, "get_feature_names_out"):
        feature_names = numeric_preprocessor.get_feature_names_out(numeric_columns)
        return [str(feature_name) for feature_name in feature_names]
    return list(numeric_columns)


class NativeFramePreprocessor:
    CATEGORICAL_MISSING_VALUE = "__missing__"

    def __init__(
        self,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        numeric_preprocessor: object,
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.numeric_preprocessor = numeric_preprocessor
        self.transformed_numeric_columns: list[str] = list(numeric_columns)

    def fit(self, frame: pd.DataFrame) -> "NativeFramePreprocessor":
        if self.numeric_columns:
            self.numeric_preprocessor.fit(frame.loc[:, self.numeric_columns])
            self.transformed_numeric_columns = _resolve_transformed_numeric_columns(
                self.numeric_preprocessor,
                self.numeric_columns,
            )
        return self

    def _transform_numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_columns:
            return pd.DataFrame(index=frame.index)
        transformed_values = self.numeric_preprocessor.transform(frame.loc[:, self.numeric_columns])
        return pd.DataFrame(
            _ensure_dense_array(transformed_values),
            index=frame.index,
            columns=self.transformed_numeric_columns,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed_frames: list[pd.DataFrame] = [self._transform_numeric_frame(frame)]

        if self.categorical_columns:
            categorical_frame = frame.loc[:, self.categorical_columns].astype(object)
            categorical_frame = categorical_frame.where(
                categorical_frame.notna(),
                self.CATEGORICAL_MISSING_VALUE,
            )
            transformed_frames.append(categorical_frame.astype(str))

        populated_frames = [transformed_frame for transformed_frame in transformed_frames if not transformed_frame.empty]
        if not populated_frames:
            return pd.DataFrame(index=frame.index)
        return pd.concat(populated_frames, axis=1)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


class FrequencyFramePreprocessor:
    CATEGORICAL_MISSING_VALUE = "__missing__"

    def __init__(
        self,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        numeric_preprocessor: object,
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.numeric_preprocessor = numeric_preprocessor
        self.transformed_numeric_columns: list[str] = list(numeric_columns)
        self.category_frequencies: dict[str, dict[str, float]] = {}

    def fit(self, frame: pd.DataFrame) -> "FrequencyFramePreprocessor":
        if self.numeric_columns:
            self.numeric_preprocessor.fit(frame.loc[:, self.numeric_columns])
            self.transformed_numeric_columns = _resolve_transformed_numeric_columns(
                self.numeric_preprocessor,
                self.numeric_columns,
            )

        for column in self.categorical_columns:
            normalized_series = _normalize_categorical_series(
                frame[column],
                missing_value=self.CATEGORICAL_MISSING_VALUE,
            )
            value_frequencies = normalized_series.value_counts(normalize=True, dropna=False)
            self.category_frequencies[column] = {
                category_value: float(frequency)
                for category_value, frequency in value_frequencies.items()
            }

        return self

    def _transform_numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_columns:
            return pd.DataFrame(index=frame.index)
        transformed_values = self.numeric_preprocessor.transform(frame.loc[:, self.numeric_columns])
        return pd.DataFrame(
            _ensure_dense_array(transformed_values),
            index=frame.index,
            columns=self.transformed_numeric_columns,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed_frames: list[pd.DataFrame] = [self._transform_numeric_frame(frame)]

        if self.categorical_columns:
            encoded_categorical_columns: dict[str, pd.Series] = {}
            for column in self.categorical_columns:
                normalized_series = _normalize_categorical_series(
                    frame[column],
                    missing_value=self.CATEGORICAL_MISSING_VALUE,
                )
                encoded_categorical_columns[column] = normalized_series.map(
                    self.category_frequencies[column]
                ).fillna(0.0)
            transformed_frames.append(pd.DataFrame(encoded_categorical_columns, index=frame.index).astype(float))

        populated_frames = [transformed_frame for transformed_frame in transformed_frames if not transformed_frame.empty]
        if not populated_frames:
            return pd.DataFrame(index=frame.index)
        return pd.concat(populated_frames, axis=1)

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
    excluded_feature_columns.extend(column for column in drop_columns if column != id_column)

    x_train_raw = train_df.drop(columns=[label_column, *excluded_feature_columns])
    y_train = train_df[label_column]
    x_test_raw = test_df.drop(columns=excluded_feature_columns)

    _validate_column_names("force_categorical", force_categorical, x_train_raw.columns.tolist())
    _validate_column_names("force_numeric", force_numeric, x_train_raw.columns.tolist())

    overlap = sorted(set(force_categorical).intersection(force_numeric))
    if overlap:
        raise ValueError(f"Columns cannot be both forced categorical and forced numeric: {overlap}")

    return x_train_raw, x_test_raw, y_train


def _build_numeric_median_preprocessor() -> object:
    return SimpleImputer(strategy="median")


def _build_numeric_standardize_preprocessor() -> object:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _build_numeric_kbins_preprocessor() -> object:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "kbins",
                KBinsDiscretizer(
                    n_bins=5,
                    encode="onehot-dense",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
            ),
        ]
    )


def _build_categorical_onehot_preprocessor() -> object:
    return Pipeline(
        steps=[
            ("coerce_object", FunctionTransformer(_coerce_object, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def _build_categorical_ordinal_preprocessor() -> object:
    return Pipeline(
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


NUMERIC_PREPROCESSING_REGISTRY = {
    "median": NumericPreprocessingDefinition(
        scheme_id="median",
        scheme_name="MedianImpute",
        builder=_build_numeric_median_preprocessor,
    ),
    "standardize": NumericPreprocessingDefinition(
        scheme_id="standardize",
        scheme_name="MedianImputeStandardize",
        builder=_build_numeric_standardize_preprocessor,
    ),
    "kbins": NumericPreprocessingDefinition(
        scheme_id="kbins",
        scheme_name="MedianImputeKBins",
        builder=_build_numeric_kbins_preprocessor,
    ),
}

CATEGORICAL_PREPROCESSING_REGISTRY = {
    "onehot": CategoricalPreprocessingDefinition(
        scheme_id="onehot",
        scheme_name="OneHotCategorical",
        compose_mode="column_transformer",
        builder=_build_categorical_onehot_preprocessor,
    ),
    "ordinal": CategoricalPreprocessingDefinition(
        scheme_id="ordinal",
        scheme_name="OrdinalCategorical",
        compose_mode="column_transformer",
        builder=_build_categorical_ordinal_preprocessor,
    ),
    "frequency": CategoricalPreprocessingDefinition(
        scheme_id="frequency",
        scheme_name="FrequencyCategorical",
        compose_mode="frequency_frame",
    ),
    "native": CategoricalPreprocessingDefinition(
        scheme_id="native",
        scheme_name="NativeCategorical",
        compose_mode="native_frame",
    ),
}


def get_numeric_preprocessing_definition(scheme_id: str) -> NumericPreprocessingDefinition:
    try:
        return NUMERIC_PREPROCESSING_REGISTRY[scheme_id]
    except KeyError as exc:
        supported_scheme_ids = sorted(NUMERIC_PREPROCESSING_REGISTRY)
        raise ValueError(
            f"Unsupported numeric_preprocessor '{scheme_id}'. Supported values: {supported_scheme_ids}"
        ) from exc


def get_categorical_preprocessing_definition(scheme_id: str) -> CategoricalPreprocessingDefinition:
    try:
        return CATEGORICAL_PREPROCESSING_REGISTRY[scheme_id]
    except KeyError as exc:
        supported_scheme_ids = sorted(CATEGORICAL_PREPROCESSING_REGISTRY)
        raise ValueError(
            f"Unsupported categorical_preprocessor '{scheme_id}'. Supported values: {supported_scheme_ids}"
        ) from exc


def _build_column_transformer_preprocessor(
    feature_schema: ResolvedFeatureSchema,
    numeric_preprocessor: object,
    categorical_preprocessor: object,
) -> ColumnTransformer:
    transformers = []
    if feature_schema.numeric_columns:
        transformers.append(("num", numeric_preprocessor, feature_schema.numeric_columns))
    if feature_schema.categorical_columns:
        transformers.append(("cat", categorical_preprocessor, feature_schema.categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_preprocessor(
    x_train_raw: pd.DataFrame,
    numeric_preprocessor_id: str,
    categorical_preprocessor_id: str,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
) -> tuple[object, list[str], list[str]]:
    feature_schema = resolve_feature_schema(
        x_train_raw=x_train_raw,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        low_cardinality_int_threshold=low_cardinality_int_threshold,
    )
    preprocessor = build_preprocessor_from_schema(
        feature_schema=feature_schema,
        numeric_preprocessor_id=numeric_preprocessor_id,
        categorical_preprocessor_id=categorical_preprocessor_id,
    )
    return preprocessor, feature_schema.numeric_columns, feature_schema.categorical_columns


def build_preprocessor_from_schema(
    feature_schema: ResolvedFeatureSchema,
    numeric_preprocessor_id: str,
    categorical_preprocessor_id: str,
) -> object:
    if not feature_schema.numeric_columns and not feature_schema.categorical_columns:
        raise ValueError("No modeled features remain after excluding id_column and applying drop_columns.")

    numeric_definition = get_numeric_preprocessing_definition(numeric_preprocessor_id)
    categorical_definition = get_categorical_preprocessing_definition(categorical_preprocessor_id)
    numeric_preprocessor = numeric_definition.builder()

    if categorical_definition.compose_mode == "column_transformer":
        if categorical_definition.builder is None:
            raise RuntimeError("column_transformer categorical preprocessing requires a builder.")
        return _build_column_transformer_preprocessor(
            feature_schema=feature_schema,
            numeric_preprocessor=numeric_preprocessor,
            categorical_preprocessor=categorical_definition.builder(),
        )

    if categorical_definition.compose_mode == "frequency_frame":
        return FrequencyFramePreprocessor(
            feature_columns=feature_schema.feature_columns,
            numeric_columns=feature_schema.numeric_columns,
            categorical_columns=feature_schema.categorical_columns,
            numeric_preprocessor=numeric_preprocessor,
        )

    if categorical_definition.compose_mode == "native_frame":
        return NativeFramePreprocessor(
            feature_columns=feature_schema.feature_columns,
            numeric_columns=feature_schema.numeric_columns,
            categorical_columns=feature_schema.categorical_columns,
            numeric_preprocessor=numeric_preprocessor,
        )

    raise ValueError(
        "Unsupported categorical preprocessing compose mode: "
        f"{categorical_definition.compose_mode!r}"
    )


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
