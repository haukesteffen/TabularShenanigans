from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from tabular_shenanigans.representations.feature_schema import ResolvedFeatureSchema
from tabular_shenanigans.representations.types import (
    FeatureBlock,
    FeatureGenerator,
    FeaturePruner,
    FeatureBundle,
    FitMode,
    FittedFeatureGenerator,
    FittedFeaturePruner,
    OutputKind,
)


def _normalize_categorical_series(series: pd.Series, missing_value: str) -> pd.Series:
    return series.astype(object).where(series.notna(), missing_value).astype(str)


def _validate_no_unknown_params(
    component_kind: str,
    component_id: str,
    params: dict[str, object],
    allowed_keys: set[str],
) -> None:
    unknown_keys = sorted(key for key in params if key not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"{component_kind} '{component_id}' does not support parameters {unknown_keys}. "
            f"Supported parameters: {sorted(allowed_keys)}"
        )


@dataclass(frozen=True)
class _FittedDenseFrameGenerator:
    block_id: str
    output_kind: OutputKind = "dense_numeric"
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda X: X

    def transform(self, X: pd.DataFrame) -> FeatureBlock:
        frame = self.transform_fn(X)
        return FeatureBlock(
            block_id=self.block_id,
            output_kind=self.output_kind,
            values=frame,
            feature_names=tuple(frame.columns.tolist()),
        )


@dataclass(frozen=True)
class _FittedNativeFrameGenerator:
    block_id: str
    columns: tuple[str, ...]
    normalize_categoricals: bool = False
    missing_value: str = "__missing__"
    output_kind: OutputKind = "native_tabular"

    def transform(self, X: pd.DataFrame) -> FeatureBlock:
        if not self.columns:
            frame = pd.DataFrame(index=X.index)
        else:
            frame = X.loc[:, list(self.columns)].copy()
        if self.normalize_categoricals:
            for column in frame.columns:
                frame[column] = _normalize_categorical_series(frame[column], self.missing_value)
        return FeatureBlock(
            block_id=self.block_id,
            output_kind=self.output_kind,
            values=frame,
            feature_names=tuple(frame.columns.tolist()),
        )


@dataclass(frozen=True)
class _FittedSparseGenerator:
    block_id: str
    transformer: object
    columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    output_kind: OutputKind = "sparse_numeric"

    def transform(self, X: pd.DataFrame) -> FeatureBlock:
        if not self.columns:
            matrix = sparse.csr_matrix((len(X.index), 0), dtype=float)
        else:
            matrix = sparse.csr_matrix(self.transformer.transform(X.loc[:, list(self.columns)]))
        return FeatureBlock(
            block_id=self.block_id,
            output_kind=self.output_kind,
            values=matrix,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True)
class _FittedHighCorrelationPruner:
    dropped_columns: frozenset[str]

    def transform(self, bundle: FeatureBundle) -> FeatureBundle:
        return bundle.drop_dense_columns(set(self.dropped_columns))


@dataclass(frozen=True)
class NativeNumericGenerator:
    operator_id: str = "native_numeric"
    fit_mode: FitMode = "stateless"
    output_kind: OutputKind = "native_tabular"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del X, y
        return _FittedNativeFrameGenerator(
            block_id=self.operator_id,
            columns=self.columns,
        )


@dataclass(frozen=True)
class NativeCategoricalGenerator:
    operator_id: str = "native_categorical"
    fit_mode: FitMode = "stateless"
    output_kind: OutputKind = "native_tabular"
    columns: tuple[str, ...] = ()
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del X, y
        return _FittedNativeFrameGenerator(
            block_id=self.operator_id,
            columns=self.columns,
            normalize_categoricals=True,
            missing_value=self.missing_value,
        )


@dataclass(frozen=True)
class StandardizeNumericGenerator:
    operator_id: str = "standardize_numeric"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        if self.columns:
            transformer.fit(X.loc[:, list(self.columns)])
        output_columns = [f"{self.operator_id}__{column}" for column in self.columns]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if not self.columns:
                return pd.DataFrame(index=frame.index)
            transformed = transformer.transform(frame.loc[:, list(self.columns)])
            return pd.DataFrame(transformed, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class FrequencyEncodeCategoricalsGenerator:
    operator_id: str = "frequency_encode_categoricals"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        category_frequencies: dict[str, dict[str, float]] = {}
        for column in self.columns:
            normalized = _normalize_categorical_series(X[column], self.missing_value)
            frequencies = normalized.value_counts(normalize=True, dropna=False)
            category_frequencies[column] = {
                str(category_value): float(freq_value)
                for category_value, freq_value in frequencies.items()
            }
        output_columns = [f"{self.operator_id}__{column}" for column in self.columns]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            encoded_columns: dict[str, pd.Series] = {}
            for input_column, output_column in zip(self.columns, output_columns, strict=True):
                normalized = _normalize_categorical_series(frame[input_column], self.missing_value)
                encoded_columns[output_column] = normalized.map(category_frequencies[input_column]).fillna(0.0)
            return pd.DataFrame(encoded_columns, index=frame.index).astype(float)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class OrdinalEncodeCategoricalsGenerator:
    operator_id: str = "ordinal_encode_categoricals"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        transformer = Pipeline(
            steps=[
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
        prepared = X.loc[:, list(self.columns)].astype(object)
        if self.columns:
            transformer.fit(prepared)
        output_columns = [f"{self.operator_id}__{column}" for column in self.columns]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if not self.columns:
                return pd.DataFrame(index=frame.index)
            transformed = transformer.transform(frame.loc[:, list(self.columns)].astype(object))
            return pd.DataFrame(transformed, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class OneHotLowCardinalityCategoricalsGenerator:
    operator_id: str = "onehot_encode_low_cardinality_categoricals"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "sparse_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        prepared = X.loc[:, list(self.columns)].astype(object)
        feature_names: tuple[str, ...] = ()
        if self.columns:
            transformer.fit(prepared)
            onehot = transformer.named_steps["onehot"]
            feature_names = tuple(
                f"{self.operator_id}__{feature_name}"
                for feature_name in onehot.get_feature_names_out(list(self.columns))
            )
        return _FittedSparseGenerator(
            block_id=self.operator_id,
            transformer=transformer,
            columns=self.columns,
            feature_names=feature_names,
        )


@dataclass(frozen=True)
class RowMissingCountGenerator:
    operator_id: str = "row_missing_count"
    fit_mode: FitMode = "stateless"
    output_kind: OutputKind = "dense_numeric"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del X, y

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {f"{self.operator_id}__value": frame.isna().sum(axis=1).astype(float)},
                index=frame.index,
            )

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class HighCorrelationPruner:
    pruner_id: str = "high_correlation_prune"
    fit_mode: FitMode = "unsupervised"
    threshold: float = 0.98

    def fit(self, bundle: FeatureBundle, y: pd.Series | None) -> FittedFeaturePruner:
        del y
        dense_frame = bundle.dense_frame()
        if dense_frame.empty or dense_frame.shape[1] < 2:
            return _FittedHighCorrelationPruner(dropped_columns=frozenset())

        correlation_frame = dense_frame.corr().abs()
        upper_triangle = correlation_frame.where(
            np.triu(np.ones(correlation_frame.shape), k=1).astype(bool)
        )
        dropped_columns = frozenset(
            column for column in upper_triangle.columns if (upper_triangle[column] > self.threshold).any()
        )
        return _FittedHighCorrelationPruner(dropped_columns=dropped_columns)


def _select_low_cardinality_categoricals(
    X: pd.DataFrame,
    columns: list[str],
    max_cardinality: int,
) -> tuple[str, ...]:
    selected_columns = []
    for column in columns:
        unique_count = int(X[column].nunique(dropna=True))
        if unique_count <= max_cardinality:
            selected_columns.append(column)
    return tuple(selected_columns)


def build_feature_generator(
    component_id: str,
    params: dict[str, object],
    feature_schema: ResolvedFeatureSchema,
    X_sample: pd.DataFrame,
) -> FeatureGenerator:
    if component_id == "native_numeric":
        _validate_no_unknown_params("operator", component_id, params, set())
        return NativeNumericGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "native_categorical":
        _validate_no_unknown_params("operator", component_id, params, {"missing_value"})
        missing_value = str(params.get("missing_value", "__missing__"))
        return NativeCategoricalGenerator(
            columns=tuple(feature_schema.categorical_columns),
            missing_value=missing_value,
        )

    if component_id == "standardize_numeric":
        _validate_no_unknown_params("operator", component_id, params, set())
        return StandardizeNumericGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "frequency_encode_categoricals":
        _validate_no_unknown_params("operator", component_id, params, {"missing_value"})
        missing_value = str(params.get("missing_value", "__missing__"))
        return FrequencyEncodeCategoricalsGenerator(
            columns=tuple(feature_schema.categorical_columns),
            missing_value=missing_value,
        )

    if component_id == "ordinal_encode_categoricals":
        _validate_no_unknown_params("operator", component_id, params, set())
        return OrdinalEncodeCategoricalsGenerator(columns=tuple(feature_schema.categorical_columns))

    if component_id == "onehot_encode_low_cardinality_categoricals":
        _validate_no_unknown_params("operator", component_id, params, {"max_cardinality"})
        max_cardinality = int(params.get("max_cardinality", 16))
        if max_cardinality < 1:
            raise ValueError("operator 'onehot_encode_low_cardinality_categoricals' requires max_cardinality >= 1.")
        eligible_columns = _select_low_cardinality_categoricals(
            X=X_sample,
            columns=feature_schema.categorical_columns,
            max_cardinality=max_cardinality,
        )
        return OneHotLowCardinalityCategoricalsGenerator(columns=eligible_columns)

    if component_id == "row_missing_count":
        _validate_no_unknown_params("operator", component_id, params, set())
        return RowMissingCountGenerator()

    raise ValueError(f"Unsupported operator id '{component_id}'.")


def build_feature_pruner(component_id: str, params: dict[str, object]) -> FeaturePruner:
    if component_id == "high_correlation_prune":
        _validate_no_unknown_params("pruner", component_id, params, {"threshold"})
        threshold = float(params.get("threshold", 0.98))
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError("pruner 'high_correlation_prune' requires threshold in (0.0, 1.0].")
        return HighCorrelationPruner(threshold=threshold)

    raise ValueError(f"Unsupported pruner id '{component_id}'.")


def validate_component_params(component_id: str, params: dict[str, object], component_kind: str) -> None:
    if component_kind == "operator":
        if component_id in {"native_numeric", "standardize_numeric", "ordinal_encode_categoricals", "row_missing_count"}:
            _validate_no_unknown_params(component_kind, component_id, params, set())
            return
        if component_id in {"native_categorical", "frequency_encode_categoricals"}:
            _validate_no_unknown_params(component_kind, component_id, params, {"missing_value"})
            return
        if component_id == "onehot_encode_low_cardinality_categoricals":
            _validate_no_unknown_params(component_kind, component_id, params, {"max_cardinality"})
            max_cardinality = int(params.get("max_cardinality", 16))
            if max_cardinality < 1:
                raise ValueError(
                    "operator 'onehot_encode_low_cardinality_categoricals' requires max_cardinality >= 1."
                )
            return
    if component_kind == "pruner":
        if component_id == "high_correlation_prune":
            _validate_no_unknown_params(component_kind, component_id, params, {"threshold"})
            threshold = float(params.get("threshold", 0.98))
            if threshold <= 0.0 or threshold > 1.0:
                raise ValueError("pruner 'high_correlation_prune' requires threshold in (0.0, 1.0].")
            return
    raise ValueError(f"Unsupported {component_kind} id '{component_id}'.")


SUPPORTED_OPERATOR_IDS = frozenset(
    {
        "native_numeric",
        "native_categorical",
        "standardize_numeric",
        "frequency_encode_categoricals",
        "ordinal_encode_categoricals",
        "onehot_encode_low_cardinality_categoricals",
        "row_missing_count",
    }
)

SUPPORTED_PRUNER_IDS = frozenset({"high_correlation_prune"})
