from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer as SklearnKBinsDiscretizer

from tabular_shenanigans.preprocess import ResolvedFeatureSchema


def _import_gpu_preprocessing_modules():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "Explicit cuML preprocessing requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    try:
        import cudf
    except ImportError as exc:
        raise RuntimeError(
            "Explicit cuML preprocessing requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    try:
        from cuml.preprocessing import KBinsDiscretizer, OneHotEncoder, SimpleImputer, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Explicit cuML preprocessing requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    return cp, cudf, SimpleImputer, StandardScaler, OneHotEncoder, KBinsDiscretizer

def _to_cupy_dense(values: object):
    cp, _, _, _, _, _ = _import_gpu_preprocessing_modules()
    if type(values).__module__.startswith("cupy"):
        return values
    if hasattr(values, "to_cupy"):
        return values.to_cupy()
    if hasattr(values, "values"):
        return cp.asarray(values.values)
    return cp.asarray(values)


@dataclass(frozen=True)
class _ResolvedNumericGpuTransformers:
    imputer: object
    post_imputer_transformer: object | None


def build_gpu_kbins_discretizer(kbins_discretizer_class: type[object]) -> object:
    try:
        return kbins_discretizer_class(
            n_bins=5,
            encode="onehot-dense",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
    except TypeError as exc:
        if "quantile_method" not in str(exc):
            raise
        return kbins_discretizer_class(
            n_bins=5,
            encode="onehot-dense",
            strategy="quantile",
        )


def fit_kbins_transformer(imputed_numeric: object, kbins_discretizer_class: type[object]) -> tuple[object, object]:
    post_imputer_transformer = build_gpu_kbins_discretizer(kbins_discretizer_class)
    try:
        transformed_numeric = post_imputer_transformer.fit_transform(imputed_numeric)
    except TypeError as exc:
        if "Implicit conversion to a NumPy array is not allowed" not in str(exc):
            raise
        cpu_kbins = SklearnKBinsDiscretizer(
            n_bins=5,
            encode="onehot-dense",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
        cpu_input = imputed_numeric.to_pandas() if hasattr(imputed_numeric, "to_pandas") else imputed_numeric
        transformed_numeric = cpu_kbins.fit_transform(cpu_input)
        return cpu_kbins, transformed_numeric
    return post_imputer_transformer, transformed_numeric


def transform_kbins_values(post_imputer_transformer: object, imputed_numeric: object) -> object:
    if type(post_imputer_transformer).__module__.startswith("sklearn"):
        cpu_input = imputed_numeric.to_pandas() if hasattr(imputed_numeric, "to_pandas") else imputed_numeric
        return post_imputer_transformer.transform(cpu_input)
    return post_imputer_transformer.transform(imputed_numeric)


class GpuCumlDensePreprocessor:
    CATEGORICAL_MISSING_VALUE = "__missing__"

    def __init__(
        self,
        feature_schema: ResolvedFeatureSchema,
        numeric_preprocessor_id: str,
        categorical_preprocessor_id: str,
    ) -> None:
        self.feature_schema = feature_schema
        self.numeric_preprocessor_id = numeric_preprocessor_id
        self.categorical_preprocessor_id = categorical_preprocessor_id
        self.numeric_transformers: _ResolvedNumericGpuTransformers | None = None
        self.categorical_fill_values: dict[str, object] | None = None
        self.categorical_encoder: object | None = None

    def _fit_numeric(self, frame: pd.DataFrame) -> object | None:
        if not self.feature_schema.numeric_columns:
            return None
        _, cudf, SimpleImputer, StandardScaler, _, KBinsDiscretizer = _import_gpu_preprocessing_modules()
        numeric_frame = cudf.from_pandas(frame.loc[:, self.feature_schema.numeric_columns]).astype("float64")
        numeric_imputer = SimpleImputer(strategy="median")
        imputed_numeric = numeric_imputer.fit_transform(numeric_frame)

        post_imputer_transformer = None
        transformed_numeric = imputed_numeric
        if self.numeric_preprocessor_id == "standardize":
            post_imputer_transformer = StandardScaler()
            transformed_numeric = post_imputer_transformer.fit_transform(imputed_numeric)
        elif self.numeric_preprocessor_id == "kbins":
            post_imputer_transformer, transformed_numeric = fit_kbins_transformer(
                imputed_numeric,
                KBinsDiscretizer,
            )

        self.numeric_transformers = _ResolvedNumericGpuTransformers(
            imputer=numeric_imputer,
            post_imputer_transformer=post_imputer_transformer,
        )
        return _to_cupy_dense(transformed_numeric)

    def _fit_categorical(self, frame: pd.DataFrame) -> object | None:
        if not self.feature_schema.categorical_columns:
            return None
        _, cudf, _, _, OneHotEncoder, _ = _import_gpu_preprocessing_modules()
        categorical_fill_values: dict[str, object] = {}
        filled_categorical_frame = frame.loc[:, self.feature_schema.categorical_columns].astype(object).copy()
        for column in self.feature_schema.categorical_columns:
            non_null_values = filled_categorical_frame[column].dropna()
            fill_value = self.CATEGORICAL_MISSING_VALUE
            if not non_null_values.empty:
                fill_value = non_null_values.mode(dropna=True).iloc[0]
            categorical_fill_values[column] = fill_value
            filled_categorical_frame[column] = filled_categorical_frame[column].where(
                filled_categorical_frame[column].notna(),
                fill_value,
            )

        categorical_frame = cudf.from_pandas(filled_categorical_frame)
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformed_categorical = categorical_encoder.fit_transform(categorical_frame)
        self.categorical_fill_values = categorical_fill_values
        self.categorical_encoder = categorical_encoder
        return _to_cupy_dense(transformed_categorical)

    def fit(self, frame: pd.DataFrame) -> "GpuCumlDensePreprocessor":
        self._fit_numeric(frame)
        self._fit_categorical(frame)
        return self

    def _transform_numeric(self, frame: pd.DataFrame) -> object | None:
        if not self.feature_schema.numeric_columns:
            return None
        if self.numeric_transformers is None:
            raise ValueError("Numeric GPU preprocessing must be fit before transform.")
        _, cudf, _, _, _, _ = _import_gpu_preprocessing_modules()
        numeric_frame = cudf.from_pandas(frame.loc[:, self.feature_schema.numeric_columns]).astype("float64")
        transformed_numeric = self.numeric_transformers.imputer.transform(numeric_frame)
        if self.numeric_transformers.post_imputer_transformer is not None:
            if self.numeric_preprocessor_id == "kbins":
                transformed_numeric = transform_kbins_values(
                    self.numeric_transformers.post_imputer_transformer,
                    transformed_numeric,
                )
            else:
                transformed_numeric = self.numeric_transformers.post_imputer_transformer.transform(
                    transformed_numeric
                )
        return _to_cupy_dense(transformed_numeric)

    def _transform_categorical(self, frame: pd.DataFrame) -> object | None:
        if not self.feature_schema.categorical_columns:
            return None
        if self.categorical_fill_values is None or self.categorical_encoder is None:
            raise ValueError("Categorical GPU preprocessing must be fit before transform.")
        _, cudf, _, _, _, _ = _import_gpu_preprocessing_modules()
        filled_categorical_frame = frame.loc[:, self.feature_schema.categorical_columns].astype(object).copy()
        for column, fill_value in self.categorical_fill_values.items():
            filled_categorical_frame[column] = filled_categorical_frame[column].where(
                filled_categorical_frame[column].notna(),
                fill_value,
            )
        categorical_frame = cudf.from_pandas(filled_categorical_frame)
        transformed_categorical = self.categorical_encoder.transform(categorical_frame)
        return _to_cupy_dense(transformed_categorical)

    def transform(self, frame: pd.DataFrame):
        cp, _, _, _, _, _ = _import_gpu_preprocessing_modules()
        transformed_parts = [
            values
            for values in (
                self._transform_numeric(frame),
                self._transform_categorical(frame),
            )
            if values is not None
        ]
        if not transformed_parts:
            return cp.empty((len(frame), 0), dtype=cp.float64)
        if len(transformed_parts) == 1:
            return transformed_parts[0]
        return cp.concatenate(transformed_parts, axis=1)

    def fit_transform(self, frame: pd.DataFrame):
        self.fit(frame)
        return self.transform(frame)


def build_gpu_cuml_dense_preprocessor_from_schema(
    feature_schema: ResolvedFeatureSchema,
    numeric_preprocessor_id: str,
    categorical_preprocessor_id: str,
) -> GpuCumlDensePreprocessor:
    if categorical_preprocessor_id != "onehot":
        raise ValueError(
            "Explicit cuML dense preprocessing currently supports categorical_preprocessor='onehot' only. "
            f"Got '{categorical_preprocessor_id}'."
        )
    if numeric_preprocessor_id not in {"median", "standardize", "kbins"}:
        raise ValueError(
            "Explicit cuML dense preprocessing currently supports numeric_preprocessor in "
            "['kbins', 'median', 'standardize'] only. "
            f"Got '{numeric_preprocessor_id}'."
        )
    return GpuCumlDensePreprocessor(
        feature_schema=feature_schema,
        numeric_preprocessor_id=numeric_preprocessor_id,
        categorical_preprocessor_id=categorical_preprocessor_id,
    )
