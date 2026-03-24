"""Operator-to-property mappings. No runtime dependencies (pandas, sklearn, etc.)."""

from __future__ import annotations

SPARSE_PRODUCING_OPERATOR_IDS: frozenset[str] = frozenset({
    "quantile_bin_numeric",
    "onehot_encode_low_cardinality_categoricals",
    "rare_category_bucket",
    "cross_low_cardinality_categoricals",
    "cross_categorical_with_binned_numeric",
})

DENSE_PRODUCING_OPERATOR_IDS: frozenset[str] = frozenset({
    "standardize_numeric",
    "robust_scale_numeric",
    "signed_log_expand_numeric",
    "frequency_encode_categoricals",
    "ordinal_encode_categoricals",
    "target_encode_categoricals",
    "row_missing_count",
    "multiply_numeric_pairs",
    "ratio_numeric_pairs",
    "difference_numeric_pairs",
    "sum_numeric_pairs",
    "groupwise_deviation_features",
    "frequency_encode_categorical_crosses",
})
