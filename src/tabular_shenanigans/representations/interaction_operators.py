from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from tabular_shenanigans.representations.operators import (
    _FittedDenseFrameGenerator,
    _normalize_categorical_series,
    _select_low_cardinality_categoricals,
)
from tabular_shenanigans.representations.types import (
    FeatureBlock,
    FitMode,
    FittedFeatureGenerator,
    OutputKind,
)


def _fit_median_imputer(X: pd.DataFrame, columns: tuple[str, ...]) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    if columns:
        imputer.fit(X.loc[:, list(columns)])
    return imputer


def _generate_unordered_pairs(columns: tuple[str, ...]) -> list[tuple[str, str]]:
    return list(itertools.combinations(columns, 2))


@dataclass(frozen=True)
class _FittedCrossedSparseGenerator:
    block_id: str
    build_cross_frame_fn: Callable[[pd.DataFrame], pd.DataFrame]
    encoder: OneHotEncoder
    feature_names: tuple[str, ...]
    output_kind: OutputKind = "sparse_numeric"

    def transform(self, X: pd.DataFrame) -> FeatureBlock:
        crossed = self.build_cross_frame_fn(X)
        if crossed.empty:
            matrix = sparse.csr_matrix((len(X.index), 0), dtype=float)
        else:
            matrix = sparse.csr_matrix(self.encoder.transform(crossed))
        return FeatureBlock(
            block_id=self.block_id,
            output_kind=self.output_kind,
            values=matrix,
            feature_names=self.feature_names,
        )


# --- Numeric pair operators ---


@dataclass(frozen=True)
class MultiplyNumericPairsGenerator:
    operator_id: str = "multiply_numeric_pairs"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        imputer = _fit_median_imputer(X, self.columns)
        pairs = _generate_unordered_pairs(self.columns)
        output_columns = [f"{self.operator_id}__{a}_x_{b}" for a, b in pairs]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if len(self.columns) < 2:
                return pd.DataFrame(index=frame.index)
            imputed = imputer.transform(frame.loc[:, list(self.columns)])
            col_index = {col: i for i, col in enumerate(self.columns)}
            result = np.empty((len(frame), len(pairs)), dtype=float)
            for k, (a, b) in enumerate(pairs):
                result[:, k] = imputed[:, col_index[a]] * imputed[:, col_index[b]]
            return pd.DataFrame(result, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class RatioNumericPairsGenerator:
    operator_id: str = "ratio_numeric_pairs"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        imputer = _fit_median_imputer(X, self.columns)
        pairs = _generate_unordered_pairs(self.columns)
        output_columns = [f"{self.operator_id}__{a}_over_{b}" for a, b in pairs]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if len(self.columns) < 2:
                return pd.DataFrame(index=frame.index)
            imputed = imputer.transform(frame.loc[:, list(self.columns)])
            col_index = {col: i for i, col in enumerate(self.columns)}
            result = np.empty((len(frame), len(pairs)), dtype=float)
            for k, (a, b) in enumerate(pairs):
                result[:, k] = imputed[:, col_index[a]] / (imputed[:, col_index[b]] + 1e-8)
            return pd.DataFrame(result, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class DifferenceNumericPairsGenerator:
    operator_id: str = "difference_numeric_pairs"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        imputer = _fit_median_imputer(X, self.columns)
        pairs = _generate_unordered_pairs(self.columns)
        output_columns = [f"{self.operator_id}__{a}_minus_{b}" for a, b in pairs]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if len(self.columns) < 2:
                return pd.DataFrame(index=frame.index)
            imputed = imputer.transform(frame.loc[:, list(self.columns)])
            col_index = {col: i for i, col in enumerate(self.columns)}
            result = np.empty((len(frame), len(pairs)), dtype=float)
            for k, (a, b) in enumerate(pairs):
                result[:, k] = imputed[:, col_index[a]] - imputed[:, col_index[b]]
            return pd.DataFrame(result, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class SumNumericPairsGenerator:
    operator_id: str = "sum_numeric_pairs"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        imputer = _fit_median_imputer(X, self.columns)
        pairs = _generate_unordered_pairs(self.columns)
        output_columns = [f"{self.operator_id}__{a}_plus_{b}" for a, b in pairs]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if len(self.columns) < 2:
                return pd.DataFrame(index=frame.index)
            imputed = imputer.transform(frame.loc[:, list(self.columns)])
            col_index = {col: i for i, col in enumerate(self.columns)}
            result = np.empty((len(frame), len(pairs)), dtype=float)
            for k, (a, b) in enumerate(pairs):
                result[:, k] = imputed[:, col_index[a]] + imputed[:, col_index[b]]
            return pd.DataFrame(result, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


# --- Categorical cross operators ---


@dataclass(frozen=True)
class CrossLowCardinalityCategoricalsGenerator:
    operator_id: str = "cross_low_cardinality_categoricals"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "sparse_numeric"
    columns: tuple[str, ...] = ()
    max_cardinality: int = 10
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        eligible = _select_low_cardinality_categoricals(
            X, list(self.columns), self.max_cardinality,
        )
        pairs = _generate_unordered_pairs(eligible)
        missing_value = self.missing_value
        operator_id = self.operator_id

        def _build_cross_frame(frame: pd.DataFrame) -> pd.DataFrame:
            if not pairs:
                return pd.DataFrame(index=frame.index)
            crossed: dict[str, pd.Series] = {}
            for a, b in pairs:
                norm_a = _normalize_categorical_series(frame[a], missing_value)
                norm_b = _normalize_categorical_series(frame[b], missing_value)
                crossed[f"{operator_id}__{a}__x__{b}"] = norm_a + "__x__" + norm_b
            return pd.DataFrame(crossed, index=frame.index)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        feature_names: tuple[str, ...] = ()
        if pairs:
            crossed_fit = _build_cross_frame(X)
            encoder.fit(crossed_fit)
            feature_names = tuple(
                f"{operator_id}__{name}"
                for name in encoder.get_feature_names_out(list(crossed_fit.columns))
            )

        return _FittedCrossedSparseGenerator(
            block_id=self.operator_id,
            build_cross_frame_fn=_build_cross_frame,
            encoder=encoder,
            feature_names=feature_names,
        )


@dataclass(frozen=True)
class CrossCategoricalWithBinnedNumericGenerator:
    operator_id: str = "cross_categorical_with_binned_numeric"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "sparse_numeric"
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    n_bins: int = 5
    max_cardinality: int = 10
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        eligible_cats = _select_low_cardinality_categoricals(
            X, list(self.categorical_columns), self.max_cardinality,
        )
        numeric_cols = self.numeric_columns
        missing_value = self.missing_value
        operator_id = self.operator_id

        binning_pipeline: Pipeline | None = None
        if numeric_cols:
            binning_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "kbins",
                        KBinsDiscretizer(
                            n_bins=self.n_bins,
                            encode="ordinal",
                            strategy="quantile",
                            quantile_method="averaged_inverted_cdf",
                        ),
                    ),
                ]
            )
            binning_pipeline.fit(X.loc[:, list(numeric_cols)])

        cross_pairs = list(itertools.product(numeric_cols, eligible_cats))

        def _build_cross_frame(frame: pd.DataFrame) -> pd.DataFrame:
            if not cross_pairs or binning_pipeline is None:
                return pd.DataFrame(index=frame.index)
            binned = binning_pipeline.transform(frame.loc[:, list(numeric_cols)])
            binned_strs = {
                col: pd.Series(
                    ["bin_" + str(int(v)) for v in binned[:, i]],
                    index=frame.index,
                )
                for i, col in enumerate(numeric_cols)
            }
            crossed: dict[str, pd.Series] = {}
            for num_col, cat_col in cross_pairs:
                norm_cat = _normalize_categorical_series(frame[cat_col], missing_value)
                crossed[f"{operator_id}__{num_col}__x__{cat_col}"] = (
                    binned_strs[num_col] + "__x__" + norm_cat
                )
            return pd.DataFrame(crossed, index=frame.index)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        feature_names: tuple[str, ...] = ()
        if cross_pairs:
            crossed_fit = _build_cross_frame(X)
            encoder.fit(crossed_fit)
            feature_names = tuple(
                f"{operator_id}__{name}"
                for name in encoder.get_feature_names_out(list(crossed_fit.columns))
            )

        return _FittedCrossedSparseGenerator(
            block_id=self.operator_id,
            build_cross_frame_fn=_build_cross_frame,
            encoder=encoder,
            feature_names=feature_names,
        )


# --- Dense interaction operators ---


@dataclass(frozen=True)
class GroupwiseDeviationFeaturesGenerator:
    operator_id: str = "groupwise_deviation_features"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    max_cardinality: int = 20
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        eligible_cats = _select_low_cardinality_categoricals(
            X, list(self.categorical_columns), self.max_cardinality,
        )
        numeric_cols = self.numeric_columns
        missing_value = self.missing_value
        operator_id = self.operator_id

        imputer = _fit_median_imputer(X, numeric_cols)

        # Compute per-group stats: {num_col: {cat_col: {cat_val: (mean, std)}}}
        group_stats: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
        # Global fallback stats: {num_col: (global_mean, global_std)}
        global_stats: dict[str, tuple[float, float]] = {}

        if numeric_cols and eligible_cats:
            imputed = imputer.transform(X.loc[:, list(numeric_cols)])
            imputed_frame = pd.DataFrame(imputed, index=X.index, columns=list(numeric_cols))

            for num_col in numeric_cols:
                num_values = imputed_frame[num_col]
                g_mean = float(num_values.mean())
                g_std = float(num_values.std())
                global_stats[num_col] = (g_mean, g_std)

                group_stats[num_col] = {}
                for cat_col in eligible_cats:
                    norm_cat = _normalize_categorical_series(X[cat_col], missing_value)
                    grouped = pd.DataFrame({"value": num_values, "group": norm_cat}).groupby("group")["value"]
                    agg = grouped.agg(["mean", "std"]).fillna(0.0)
                    group_stats[num_col][cat_col] = {
                        str(group_val): (float(row["mean"]), float(row["std"]))
                        for group_val, row in agg.iterrows()
                    }

        cross_pairs = list(itertools.product(numeric_cols, eligible_cats))
        output_columns = [f"{operator_id}__{num}_dev_by_{cat}" for num, cat in cross_pairs]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if not cross_pairs:
                return pd.DataFrame(index=frame.index)
            imputed = imputer.transform(frame.loc[:, list(numeric_cols)])
            imputed_frame = pd.DataFrame(imputed, index=frame.index, columns=list(numeric_cols))
            result = np.empty((len(frame), len(cross_pairs)), dtype=float)
            for k, (num_col, cat_col) in enumerate(cross_pairs):
                norm_cat = _normalize_categorical_series(frame[cat_col], missing_value)
                num_values = imputed_frame[num_col].values
                g_mean, g_std = global_stats[num_col]
                cat_stats = group_stats[num_col][cat_col]
                means = norm_cat.map(lambda v, cs=cat_stats, gm=g_mean: cs.get(v, (gm, 0.0))[0]).values
                stds = norm_cat.map(lambda v, cs=cat_stats, gs=g_std: cs.get(v, (0.0, gs))[1]).values
                result[:, k] = (num_values - means) / (stds + 1e-8)
            return pd.DataFrame(result, index=frame.index, columns=output_columns)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)


@dataclass(frozen=True)
class FrequencyEncodeCategoricalCrossesGenerator:
    operator_id: str = "frequency_encode_categorical_crosses"
    fit_mode: FitMode = "unsupervised"
    output_kind: OutputKind = "dense_numeric"
    columns: tuple[str, ...] = ()
    max_cardinality: int = 10
    missing_value: str = "__missing__"

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator:
        del y
        eligible = _select_low_cardinality_categoricals(
            X, list(self.columns), self.max_cardinality,
        )
        pairs = _generate_unordered_pairs(eligible)
        missing_value = self.missing_value
        operator_id = self.operator_id

        # Build frequency maps per pair
        frequency_maps: dict[tuple[str, str], dict[str, float]] = {}
        for a, b in pairs:
            norm_a = _normalize_categorical_series(X[a], missing_value)
            norm_b = _normalize_categorical_series(X[b], missing_value)
            crossed = norm_a + "__x__" + norm_b
            freqs = crossed.value_counts(normalize=True)
            frequency_maps[(a, b)] = {
                str(val): float(freq) for val, freq in freqs.items()
            }

        output_columns = [f"{operator_id}__{a}__x__{b}" for a, b in pairs]

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            if not pairs:
                return pd.DataFrame(index=frame.index)
            result: dict[str, pd.Series] = {}
            for (a, b), out_col in zip(pairs, output_columns, strict=True):
                norm_a = _normalize_categorical_series(frame[a], missing_value)
                norm_b = _normalize_categorical_series(frame[b], missing_value)
                crossed = norm_a + "__x__" + norm_b
                result[out_col] = crossed.map(frequency_maps[(a, b)]).fillna(0.0)
            return pd.DataFrame(result, index=frame.index).astype(float)

        return _FittedDenseFrameGenerator(block_id=self.operator_id, transform_fn=_transform)
