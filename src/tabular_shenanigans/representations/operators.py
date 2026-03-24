from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from tabular_shenanigans._array_utils import _normalize_categorical_series
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
class RobustScaleNumericGenerator:
    operator_id: str = "robust_scale_numeric"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
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
class SignedLogExpandNumericGenerator:
    operator_id: str = "signed_log_expand_numeric"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        transformer = SimpleImputer(strategy="median")
        if self.columns:
            transformer.fit(X.loc[:, list(self.columns)])
        output_columns = [f"{self.operator_id}__{column}" for column in self.columns]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if not self.columns:
                return pd.DataFrame(index=frame.index)
            imputed = transformer.transform(frame.loc[:, list(self.columns)])
            transformed = np.sign(imputed) * np.log1p(np.abs(imputed))
            return pd.DataFrame(transformed, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class QuantileBinNumericGenerator:
    operator_id: str = "quantile_bin_numeric"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "sparse_numeric"
    columns: tuple[str, ...] = ()
    n_bins: int = 5

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "kbins",
                    KBinsDiscretizer(
                        n_bins=self.n_bins,
                        encode="onehot",
                        strategy="quantile",
                        quantile_method="averaged_inverted_cdf",
                    ),
                ),
            ]
        )
        feature_names: tuple[str, ...] = ()
        if self.columns:
            transformer.fit(X.loc[:, list(self.columns)])
            kbins = transformer.named_steps["kbins"]
            feature_names = tuple(
                f"{self.operator_id}__{feature_name}"
                for feature_name in kbins.get_feature_names_out(list(self.columns))
            )
        return _FittedSparseGenerator(
            block_id=self.operator_id,
            transformer=transformer,
            columns=self.columns,
            feature_names=feature_names,
        )


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
class TargetEncodeCategoricalsGenerator:
    operator_id: str = "target_encode_categoricals"
    fit_mode: FitMode = "supervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()
    smoothing: float = 1.0
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        if y is None:
            raise ValueError("target_encode_categoricals requires y (supervised fit_mode).")
        global_mean = float(y.mean())
        category_encodings: dict[str, dict[str, float]] = {}
        for column in self.columns:
            normalized = _normalize_categorical_series(X[column], self.missing_value)
            stats = pd.DataFrame({"category": normalized, "target": y})
            agg = stats.groupby("category")["target"].agg(["mean", "count"])
            encoding = (
                (agg["count"] * agg["mean"] + self.smoothing * global_mean)
                / (agg["count"] + self.smoothing)
            )
            category_encodings[column] = {
                str(category): float(value) for category, value in encoding.items()
            }
        output_columns = [f"{self.operator_id}__{column}" for column in self.columns]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            encoded_columns: dict[str, pd.Series] = {}
            for input_column, output_column in zip(self.columns, output_columns, strict=True):
                normalized = _normalize_categorical_series(frame[input_column], self.missing_value)
                encoded_columns[output_column] = normalized.map(
                    category_encodings[input_column]
                ).fillna(global_mean)
            return pd.DataFrame(encoded_columns, index=frame.index).astype(float)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class RareCategoryBucketGenerator:
    operator_id: str = "rare_category_bucket"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "sparse_numeric"
    columns: tuple[str, ...] = ()
    min_frequency: float = 0.01
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        rare_categories_per_column: dict[str, frozenset[str]] = {}
        for column in self.columns:
            normalized = _normalize_categorical_series(X[column], self.missing_value)
            frequencies = normalized.value_counts(normalize=True, dropna=False)
            rare = frozenset(
                str(category) for category, freq in frequencies.items() if freq < self.min_frequency
            )
            rare_categories_per_column[column] = rare

        non_rare_sentinel = "__non_rare__"

        def _mask_non_rare(frame: pd.DataFrame) -> pd.DataFrame:
            masked = pd.DataFrame(index=frame.index)
            for column in self.columns:
                normalized = _normalize_categorical_series(frame[column], self.missing_value)
                rare_set = rare_categories_per_column[column]
                masked[column] = normalized.where(normalized.isin(rare_set), non_rare_sentinel)
            return masked

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        feature_names: tuple[str, ...] = ()
        if self.columns:
            masked_fit = _mask_non_rare(X)
            encoder.fit(masked_fit)
            # Drop the non_rare_sentinel columns from the encoder output
            raw_feature_names = list(encoder.get_feature_names_out(list(self.columns)))
            keep_indices = [
                i for i, name in enumerate(raw_feature_names)
                if not name.endswith(f"_{non_rare_sentinel}")
            ]
            feature_names = tuple(
                f"{self.operator_id}__{raw_feature_names[i]}" for i in keep_indices
            )

        return _FittedRareCategorySparseGenerator(
            block_id=self.operator_id,
            encoder=encoder,
            columns=self.columns,
            feature_names=feature_names,
            mask_fn=_mask_non_rare,
            keep_indices=tuple(keep_indices) if self.columns else (),
        )


@dataclass(frozen=True)
class _FittedRareCategorySparseGenerator:
    block_id: str
    encoder: object
    columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    mask_fn: Callable[[pd.DataFrame], pd.DataFrame]
    keep_indices: tuple[int, ...]
    output_kind: OutputKind = "sparse_numeric"

    def transform(self, X: pd.DataFrame) -> FeatureBlock:
        if not self.columns:
            matrix = sparse.csr_matrix((len(X.index), 0), dtype=float)
        else:
            masked = self.mask_fn(X)
            full_matrix = self.encoder.transform(masked)
            matrix = sparse.csr_matrix(full_matrix.tocsc()[:, list(self.keep_indices)])
        return FeatureBlock(
            block_id=self.block_id,
            output_kind=self.output_kind,
            values=matrix,
            feature_names=self.feature_names,
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


def _validate_operator_params(component_id: str, params: dict[str, object]) -> None:
    if component_id in {
        "native_numeric",
        "standardize_numeric",
        "robust_scale_numeric",
        "signed_log_expand_numeric",
        "quantile_bin_numeric",
        "ordinal_encode_categoricals",
        "row_missing_count",
        "multiply_numeric_pairs",
        "ratio_numeric_pairs",
        "difference_numeric_pairs",
        "sum_numeric_pairs",
    }:
        _validate_no_unknown_params("operator", component_id, params, set())
        return
    if component_id in {"native_categorical", "frequency_encode_categoricals"}:
        _validate_no_unknown_params("operator", component_id, params, {"missing_value"})
        return
    if component_id == "rare_category_bucket":
        _validate_no_unknown_params("operator", component_id, params, {"min_frequency", "missing_value"})
        min_frequency = float(params.get("min_frequency", 0.01))
        if min_frequency <= 0.0 or min_frequency >= 1.0:
            raise ValueError("operator 'rare_category_bucket' requires min_frequency in (0.0, 1.0).")
        return
    if component_id == "target_encode_categoricals":
        _validate_no_unknown_params("operator", component_id, params, {"smoothing", "missing_value"})
        smoothing = float(params.get("smoothing", 1.0))
        if smoothing < 0.0:
            raise ValueError("operator 'target_encode_categoricals' requires smoothing >= 0.")
        return
    if component_id == "onehot_encode_low_cardinality_categoricals":
        _validate_no_unknown_params("operator", component_id, params, {"max_cardinality"})
        max_cardinality = int(params.get("max_cardinality", 16))
        if max_cardinality < 1:
            raise ValueError(
                "operator 'onehot_encode_low_cardinality_categoricals' requires max_cardinality >= 1."
            )
        return
    if component_id in {"cross_low_cardinality_categoricals", "frequency_encode_categorical_crosses"}:
        _validate_no_unknown_params("operator", component_id, params, {"max_cardinality", "missing_value"})
        max_cardinality = int(params.get("max_cardinality", 10))
        if max_cardinality < 1:
            raise ValueError(f"operator '{component_id}' requires max_cardinality >= 1.")
        return
    if component_id == "cross_categorical_with_binned_numeric":
        _validate_no_unknown_params(
            "operator", component_id, params, {"n_bins", "max_cardinality", "missing_value"}
        )
        n_bins = int(params.get("n_bins", 5))
        if n_bins < 2:
            raise ValueError("operator 'cross_categorical_with_binned_numeric' requires n_bins >= 2.")
        max_cardinality = int(params.get("max_cardinality", 10))
        if max_cardinality < 1:
            raise ValueError("operator 'cross_categorical_with_binned_numeric' requires max_cardinality >= 1.")
        return
    if component_id == "groupwise_deviation_features":
        _validate_no_unknown_params("operator", component_id, params, {"max_cardinality", "missing_value"})
        max_cardinality = int(params.get("max_cardinality", 20))
        if max_cardinality < 1:
            raise ValueError("operator 'groupwise_deviation_features' requires max_cardinality >= 1.")
        return
    raise ValueError(f"Unsupported operator id '{component_id}'.")


def _validate_pruner_params(component_id: str, params: dict[str, object]) -> None:
    if component_id == "high_correlation_prune":
        _validate_no_unknown_params("pruner", component_id, params, {"threshold"})
        threshold = float(params.get("threshold", 0.98))
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError("pruner 'high_correlation_prune' requires threshold in (0.0, 1.0].")
        return
    raise ValueError(f"Unsupported pruner id '{component_id}'.")


def build_feature_generator(
    component_id: str,
    params: dict[str, object],
    feature_schema: ResolvedFeatureSchema,
    X_sample: pd.DataFrame,
) -> FeatureGenerator:
    _validate_operator_params(component_id, params)

    if component_id == "native_numeric":
        return NativeNumericGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "native_categorical":
        missing_value = str(params.get("missing_value", "__missing__"))
        return NativeCategoricalGenerator(
            columns=tuple(feature_schema.categorical_columns),
            missing_value=missing_value,
        )

    if component_id == "standardize_numeric":
        return StandardizeNumericGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "robust_scale_numeric":
        return RobustScaleNumericGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "signed_log_expand_numeric":
        return SignedLogExpandNumericGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "quantile_bin_numeric":
        return QuantileBinNumericGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "frequency_encode_categoricals":
        missing_value = str(params.get("missing_value", "__missing__"))
        return FrequencyEncodeCategoricalsGenerator(
            columns=tuple(feature_schema.categorical_columns),
            missing_value=missing_value,
        )

    if component_id == "ordinal_encode_categoricals":
        return OrdinalEncodeCategoricalsGenerator(columns=tuple(feature_schema.categorical_columns))

    if component_id == "onehot_encode_low_cardinality_categoricals":
        max_cardinality = int(params.get("max_cardinality", 16))
        eligible_columns = _select_low_cardinality_categoricals(
            X=X_sample,
            columns=feature_schema.categorical_columns,
            max_cardinality=max_cardinality,
        )
        return OneHotLowCardinalityCategoricalsGenerator(columns=eligible_columns)

    if component_id == "target_encode_categoricals":
        smoothing = float(params.get("smoothing", 1.0))
        missing_value = str(params.get("missing_value", "__missing__"))
        return TargetEncodeCategoricalsGenerator(
            columns=tuple(feature_schema.categorical_columns),
            smoothing=smoothing,
            missing_value=missing_value,
        )

    if component_id == "rare_category_bucket":
        min_frequency = float(params.get("min_frequency", 0.01))
        missing_value = str(params.get("missing_value", "__missing__"))
        return RareCategoryBucketGenerator(
            columns=tuple(feature_schema.categorical_columns),
            min_frequency=min_frequency,
            missing_value=missing_value,
        )

    if component_id == "row_missing_count":
        return RowMissingCountGenerator()

    # Lazy import to avoid circular dependency (interaction_operators imports from this module).
    from tabular_shenanigans.representations.interaction_operators import (
        CrossCategoricalWithBinnedNumericGenerator,
        CrossLowCardinalityCategoricalsGenerator,
        DifferenceNumericPairsGenerator,
        FrequencyEncodeCategoricalCrossesGenerator,
        GroupwiseDeviationFeaturesGenerator,
        MultiplyNumericPairsGenerator,
        RatioNumericPairsGenerator,
        SumNumericPairsGenerator,
    )

    if component_id == "multiply_numeric_pairs":
        return MultiplyNumericPairsGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "ratio_numeric_pairs":
        return RatioNumericPairsGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "difference_numeric_pairs":
        return DifferenceNumericPairsGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "sum_numeric_pairs":
        return SumNumericPairsGenerator(columns=tuple(feature_schema.numeric_columns))

    if component_id == "cross_low_cardinality_categoricals":
        max_cardinality = int(params.get("max_cardinality", 10))
        missing_value = str(params.get("missing_value", "__missing__"))
        return CrossLowCardinalityCategoricalsGenerator(
            columns=tuple(feature_schema.categorical_columns),
            max_cardinality=max_cardinality,
            missing_value=missing_value,
        )

    if component_id == "cross_categorical_with_binned_numeric":
        n_bins = int(params.get("n_bins", 5))
        max_cardinality = int(params.get("max_cardinality", 10))
        missing_value = str(params.get("missing_value", "__missing__"))
        return CrossCategoricalWithBinnedNumericGenerator(
            categorical_columns=tuple(feature_schema.categorical_columns),
            numeric_columns=tuple(feature_schema.numeric_columns),
            n_bins=n_bins,
            max_cardinality=max_cardinality,
            missing_value=missing_value,
        )

    if component_id == "groupwise_deviation_features":
        max_cardinality = int(params.get("max_cardinality", 20))
        missing_value = str(params.get("missing_value", "__missing__"))
        return GroupwiseDeviationFeaturesGenerator(
            categorical_columns=tuple(feature_schema.categorical_columns),
            numeric_columns=tuple(feature_schema.numeric_columns),
            max_cardinality=max_cardinality,
            missing_value=missing_value,
        )

    if component_id == "frequency_encode_categorical_crosses":
        max_cardinality = int(params.get("max_cardinality", 10))
        missing_value = str(params.get("missing_value", "__missing__"))
        return FrequencyEncodeCategoricalCrossesGenerator(
            columns=tuple(feature_schema.categorical_columns),
            max_cardinality=max_cardinality,
            missing_value=missing_value,
        )

    raise ValueError(f"Unsupported operator id '{component_id}'.")


def build_feature_pruner(component_id: str, params: dict[str, object]) -> FeaturePruner:
    _validate_pruner_params(component_id, params)

    if component_id == "high_correlation_prune":
        threshold = float(params.get("threshold", 0.98))
        return HighCorrelationPruner(threshold=threshold)

    raise ValueError(f"Unsupported pruner id '{component_id}'.")


def validate_component_params(component_id: str, params: dict[str, object], component_kind: str) -> None:
    if component_kind == "operator":
        _validate_operator_params(component_id, params)
    elif component_kind == "pruner":
        _validate_pruner_params(component_id, params)
    else:
        raise ValueError(f"Unsupported component kind '{component_kind}'.")


SUPPORTED_OPERATOR_IDS = frozenset(
    {
        "native_numeric",
        "native_categorical",
        "standardize_numeric",
        "robust_scale_numeric",
        "signed_log_expand_numeric",
        "quantile_bin_numeric",
        "frequency_encode_categoricals",
        "ordinal_encode_categoricals",
        "onehot_encode_low_cardinality_categoricals",
        "target_encode_categoricals",
        "rare_category_bucket",
        "row_missing_count",
        "multiply_numeric_pairs",
        "ratio_numeric_pairs",
        "difference_numeric_pairs",
        "sum_numeric_pairs",
        "cross_low_cardinality_categoricals",
        "cross_categorical_with_binned_numeric",
        "groupwise_deviation_features",
        "frequency_encode_categorical_crosses",
    }
)

SUPPORTED_PRUNER_IDS = frozenset({"high_correlation_prune"})
