from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from tabular_shenanigans.representations.types import FitMode, FittedStep


def _coerce_object(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object)


def _ensure_dense_array(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


# --- Fitted step wrappers ---


@dataclass
class _FittedSklearnStep:
    transformer: object
    columns: list[str]
    output_columns: list[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return X
        transformed_values = self.transformer.transform(X.loc[:, self.columns])
        transformed_df = pd.DataFrame(
            _ensure_dense_array(transformed_values),
            index=X.index,
            columns=self.output_columns,
        )
        passthrough_columns = [c for c in X.columns if c not in self.columns]
        if not passthrough_columns:
            return transformed_df
        return pd.concat([transformed_df, X.loc[:, passthrough_columns]], axis=1)


@dataclass
class _FittedFrequencyEncodeStep:
    numeric_columns: list[str]
    categorical_columns: list[str]
    category_frequencies: dict[str, dict[str, float]]
    missing_value: str

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_columns:
            return X
        encoded_columns: dict[str, pd.Series] = {}
        for column in self.categorical_columns:
            normalized = _normalize_categorical_series(X[column], self.missing_value)
            encoded_columns[column] = normalized.map(
                self.category_frequencies[column]
            ).fillna(0.0)
        result = X.copy()
        for column, encoded in encoded_columns.items():
            result[column] = encoded.astype(float)
        return result


@dataclass
class _FittedOrdinalEncodeStep:
    columns: list[str]
    transformer: object

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return X
        result = X.copy()
        transformed = self.transformer.transform(X.loc[:, self.columns])
        transformed_df = pd.DataFrame(
            _ensure_dense_array(transformed),
            index=X.index,
            columns=self.columns,
        )
        for column in self.columns:
            result[column] = transformed_df[column]
        return result


@dataclass
class _FittedOneHotEncodeStep:
    columns: list[str]
    transformer: object
    feature_names: list[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return X
        non_cat = X.drop(columns=self.columns)
        transformed = self.transformer.transform(X.loc[:, self.columns])
        onehot_df = pd.DataFrame(
            _ensure_dense_array(transformed),
            index=X.index,
            columns=self.feature_names,
        )
        return pd.concat([non_cat, onehot_df], axis=1)


@dataclass
class _FittedNativePassthroughStep:
    categorical_columns: list[str]
    missing_value: str

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_columns:
            return X
        result = X.copy()
        for column in self.categorical_columns:
            result[column] = _normalize_categorical_series(result[column], self.missing_value)
        return result


@dataclass
class _FittedStatelessStep:
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.transform_fn(X)


# --- Helpers ---


def _normalize_categorical_series(series: pd.Series, missing_value: str) -> pd.Series:
    return series.astype(object).where(series.notna(), missing_value).astype(str)


def _resolve_transformed_columns(transformer: object, columns: list[str]) -> list[str]:
    if not columns:
        return []
    if hasattr(transformer, "get_feature_names_out"):
        return [str(name) for name in transformer.get_feature_names_out(columns)]
    return list(columns)


# --- Column binding ---

# Maps step classes to whether they operate on numeric or categorical columns.
_NUMERIC_STEP_TYPES: set[type] = set()
_CATEGORICAL_STEP_TYPES: set[type] = set()


def _mark_numeric(cls: type) -> type:
    _NUMERIC_STEP_TYPES.add(cls)
    return cls


def _mark_categorical(cls: type) -> type:
    _CATEGORICAL_STEP_TYPES.add(cls)
    return cls


def is_numeric_step(step: object) -> bool:
    return type(step) in _NUMERIC_STEP_TYPES


def is_categorical_step(step: object) -> bool:
    return type(step) in _CATEGORICAL_STEP_TYPES


def bind_step_columns(step: object, columns: list[str]) -> object:
    """Return a copy of the step with its columns field bound to the given list."""
    from dataclasses import replace
    if not hasattr(step, "columns"):
        return step
    return replace(step, columns=columns)


# --- Step implementations ---


@_mark_numeric
@dataclass(frozen=True)
class MedianImputeStep:
    step_id: str = "median_impute"
    fit_mode: FitMode = "unsupervised"
    columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        target_columns = self.columns if self.columns is not None else X.columns.tolist()
        numeric_columns = [c for c in target_columns if c in X.columns]
        transformer = SimpleImputer(strategy="median")
        if numeric_columns:
            transformer.fit(X.loc[:, numeric_columns])
        output_columns = _resolve_transformed_columns(transformer, numeric_columns)
        return _FittedSklearnStep(
            transformer=transformer,
            columns=numeric_columns,
            output_columns=output_columns,
        )


@_mark_numeric
@dataclass(frozen=True)
class StandardizeStep:
    step_id: str = "standardize"
    fit_mode: FitMode = "unsupervised"
    columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        target_columns = self.columns if self.columns is not None else X.columns.tolist()
        numeric_columns = [c for c in target_columns if c in X.columns]
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        if numeric_columns:
            transformer.fit(X.loc[:, numeric_columns])
        output_columns = _resolve_transformed_columns(transformer, numeric_columns)
        return _FittedSklearnStep(
            transformer=transformer,
            columns=numeric_columns,
            output_columns=output_columns,
        )


@_mark_numeric
@dataclass(frozen=True)
class KBinsStep:
    step_id: str = "kbins"
    fit_mode: FitMode = "unsupervised"
    columns: list[str] | None = None
    sparse_output: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        target_columns = self.columns if self.columns is not None else X.columns.tolist()
        numeric_columns = [c for c in target_columns if c in X.columns]
        kbins_encode = "onehot" if self.sparse_output else "onehot-dense"
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "kbins",
                    KBinsDiscretizer(
                        n_bins=5,
                        encode=kbins_encode,
                        strategy="quantile",
                        quantile_method="averaged_inverted_cdf",
                    ),
                ),
            ]
        )
        if numeric_columns:
            transformer.fit(X.loc[:, numeric_columns])
        output_columns = _resolve_transformed_columns(transformer, numeric_columns)
        return _FittedSklearnStep(
            transformer=transformer,
            columns=numeric_columns,
            output_columns=output_columns,
        )


@_mark_categorical
@dataclass(frozen=True)
class OrdinalEncodeStep:
    step_id: str = "ordinal_encode"
    fit_mode: FitMode = "unsupervised"
    columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        target_columns = self.columns if self.columns is not None else X.columns.tolist()
        categorical_columns = [c for c in target_columns if c in X.columns]
        transformer = Pipeline(
            steps=[
                ("coerce_object", FunctionTransformer(_coerce_object)),
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
        if categorical_columns:
            transformer.fit(X.loc[:, categorical_columns])
        return _FittedOrdinalEncodeStep(
            columns=categorical_columns,
            transformer=transformer,
        )


@_mark_categorical
@dataclass(frozen=True)
class OneHotEncodeStep:
    step_id: str = "onehot_encode"
    fit_mode: FitMode = "unsupervised"
    columns: list[str] | None = None
    sparse_output: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        target_columns = self.columns if self.columns is not None else X.columns.tolist()
        categorical_columns = [c for c in target_columns if c in X.columns]
        transformer = Pipeline(
            steps=[
                ("coerce_object", FunctionTransformer(_coerce_object)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=self.sparse_output)),
            ]
        )
        feature_names: list[str] = []
        if categorical_columns:
            transformer.fit(X.loc[:, categorical_columns])
            onehot_encoder = transformer.named_steps["onehot"]
            feature_names = [str(name) for name in onehot_encoder.get_feature_names_out(categorical_columns)]
        return _FittedOneHotEncodeStep(
            columns=categorical_columns,
            transformer=transformer,
            feature_names=feature_names,
        )


@_mark_categorical
@dataclass(frozen=True)
class FrequencyEncodeStep:
    step_id: str = "frequency_encode"
    fit_mode: FitMode = "unsupervised"
    columns: list[str] | None = None
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        target_columns = self.columns if self.columns is not None else X.columns.tolist()
        categorical_columns = [c for c in target_columns if c in X.columns]
        category_frequencies: dict[str, dict[str, float]] = {}
        for column in categorical_columns:
            normalized = _normalize_categorical_series(X[column], self.missing_value)
            value_frequencies = normalized.value_counts(normalize=True, dropna=False)
            category_frequencies[column] = {
                str(cat_value): float(freq)
                for cat_value, freq in value_frequencies.items()
            }
        numeric_columns = [c for c in X.columns if c not in categorical_columns]
        return _FittedFrequencyEncodeStep(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            category_frequencies=category_frequencies,
            missing_value=self.missing_value,
        )


@_mark_categorical
@dataclass(frozen=True)
class NativeCategoricalStep:
    step_id: str = "native_categorical"
    fit_mode: FitMode = "stateless"
    columns: list[str] | None = None
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        target_columns = self.columns if self.columns is not None else X.columns.tolist()
        categorical_columns = [c for c in target_columns if c in X.columns]
        return _FittedNativePassthroughStep(
            categorical_columns=categorical_columns,
            missing_value=self.missing_value,
        )


@dataclass(frozen=True)
class FeatureEngineeringStep:
    step_id: str
    fit_mode: Literal["stateless"] = "stateless"
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep:
        return _FittedStatelessStep(transform_fn=self.transform_fn)
