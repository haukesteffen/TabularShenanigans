from dataclasses import dataclass

import pandas as pd

from tabular_shenanigans.gpu_cuml_preprocess import fit_kbins_transformer, transform_kbins_values
from tabular_shenanigans.preprocess import ResolvedFeatureSchema

SUPPORTED_GPU_NATIVE_NUMERIC_PREPROCESSOR_IDS = frozenset({"median", "standardize", "kbins"})
SUPPORTED_GPU_NATIVE_CATEGORICAL_PREPROCESSOR_IDS = frozenset({"frequency"})


def _import_cudf():
    try:
        import cudf
    except ImportError as exc:
        raise RuntimeError(
            "gpu_native preprocessing requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc
    return cudf


@dataclass(frozen=True)
class _ResolvedNumericStatistics:
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]


class GpuNativeFrequencyPreprocessor:
    CATEGORICAL_MISSING_VALUE = "__missing__"

    def __init__(
        self,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        numeric_preprocessor_id: str,
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.numeric_preprocessor_id = numeric_preprocessor_id
        self.numeric_statistics = _ResolvedNumericStatistics(medians={}, means={}, scales={})
        self.category_frequencies: dict[str, dict[str, float]] = {}
        self.kbins_transformer: object | None = None

    def _normalize_numeric_frame(self, frame: pd.DataFrame):
        cudf = _import_cudf()
        if not self.numeric_columns:
            return cudf.DataFrame(index=frame.index)
        return cudf.from_pandas(frame.loc[:, self.numeric_columns]).astype("float64")

    def _normalize_categorical_frame(self, frame: pd.DataFrame):
        cudf = _import_cudf()
        if not self.categorical_columns:
            return cudf.DataFrame(index=frame.index)
        categorical_frame = cudf.from_pandas(frame.loc[:, self.categorical_columns])
        return categorical_frame.fillna(self.CATEGORICAL_MISSING_VALUE).astype("str")

    def fit(self, frame: pd.DataFrame) -> "GpuNativeFrequencyPreprocessor":
        numeric_frame = self._normalize_numeric_frame(frame)
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}

        for column in self.numeric_columns:
            column_values = numeric_frame[column]
            median_value = column_values.dropna().median()
            if pd.isna(median_value):
                raise ValueError(
                    "gpu_native preprocessing does not support all-null numeric columns yet. "
                    f"Column: {column}"
                )
            median_float = float(median_value)
            medians[column] = median_float

            if self.numeric_preprocessor_id == "standardize":
                imputed_values = column_values.fillna(median_float)
                mean_float = float(imputed_values.mean())
                scale_float = float(imputed_values.std(ddof=0))
                if pd.isna(scale_float) or scale_float == 0.0:
                    scale_float = 1.0
                means[column] = mean_float
                scales[column] = scale_float

        if self.numeric_preprocessor_id == "kbins" and self.numeric_columns:
            try:
                from cuml.preprocessing import KBinsDiscretizer
            except ImportError as exc:
                raise RuntimeError(
                    "gpu_native kbins preprocessing requires the optional GPU dependencies. "
                    "Install them with `uv sync --extra boosters --extra gpu`."
                ) from exc
            cudf = _import_cudf()
            imputed_numeric_frame = cudf.from_pandas(
                frame.loc[:, self.numeric_columns]
            ).astype("float64")
            for column in self.numeric_columns:
                imputed_numeric_frame[column] = imputed_numeric_frame[column].fillna(medians[column])
            self.kbins_transformer, _ = fit_kbins_transformer(imputed_numeric_frame, KBinsDiscretizer)

        categorical_frame = self._normalize_categorical_frame(frame)
        for column in self.categorical_columns:
            value_frequencies = categorical_frame[column].value_counts(normalize=True, dropna=False)
            self.category_frequencies[column] = {
                str(category_value): float(frequency)
                for category_value, frequency in value_frequencies.to_pandas().items()
            }

        self.numeric_statistics = _ResolvedNumericStatistics(
            medians=medians,
            means=means,
            scales=scales,
        )
        return self

    def _transform_numeric_frame(self, frame: pd.DataFrame):
        cudf = _import_cudf()
        if not self.numeric_columns:
            return cudf.DataFrame(index=frame.index)

        if self.numeric_preprocessor_id == "kbins":
            if self.kbins_transformer is None:
                raise ValueError("kbins preprocessor must be fit before transform.")
            imputed_frame = cudf.from_pandas(frame.loc[:, self.numeric_columns]).astype("float64")
            for column in self.numeric_columns:
                imputed_frame[column] = imputed_frame[column].fillna(self.numeric_statistics.medians[column])
            kbins_result = transform_kbins_values(self.kbins_transformer, imputed_frame)
            return cudf.DataFrame(kbins_result, index=frame.index)

        numeric_frame = self._normalize_numeric_frame(frame)
        transformed_columns: dict[str, object] = {}
        for column in self.numeric_columns:
            transformed_values = numeric_frame[column].fillna(self.numeric_statistics.medians[column])
            if self.numeric_preprocessor_id == "standardize":
                transformed_values = (
                    transformed_values - self.numeric_statistics.means[column]
                ) / self.numeric_statistics.scales[column]
            transformed_columns[column] = transformed_values.astype("float64")
        return cudf.DataFrame(transformed_columns, index=numeric_frame.index)

    def _transform_categorical_frame(self, frame: pd.DataFrame):
        cudf = _import_cudf()
        if not self.categorical_columns:
            return cudf.DataFrame(index=frame.index)

        categorical_frame = self._normalize_categorical_frame(frame)
        transformed_columns: dict[str, object] = {}
        for column in self.categorical_columns:
            transformed_columns[column] = (
                categorical_frame[column]
                .map(self.category_frequencies[column])
                .fillna(0.0)
                .astype("float64")
            )
        return cudf.DataFrame(transformed_columns, index=categorical_frame.index)

    def transform(self, frame: pd.DataFrame):
        cudf = _import_cudf()
        transformed_frames = [
            self._transform_numeric_frame(frame),
            self._transform_categorical_frame(frame),
        ]
        populated_frames = [transformed_frame for transformed_frame in transformed_frames if len(transformed_frame.columns) > 0]
        if not populated_frames:
            return cudf.DataFrame(index=frame.index)
        return cudf.concat(populated_frames, axis=1)

    def fit_transform(self, frame: pd.DataFrame):
        return self.fit(frame).transform(frame)


def build_gpu_native_preprocessor_from_schema(
    feature_schema: ResolvedFeatureSchema,
    numeric_preprocessor_id: str,
    categorical_preprocessor_id: str,
) -> GpuNativeFrequencyPreprocessor:
    if not feature_schema.numeric_columns and not feature_schema.categorical_columns:
        raise ValueError("No modeled features remain after excluding id_column and applying drop_columns.")

    if categorical_preprocessor_id not in SUPPORTED_GPU_NATIVE_CATEGORICAL_PREPROCESSOR_IDS:
        raise ValueError(
            "gpu_native preprocessing currently supports categorical_preprocessor='frequency' only. "
            f"Got '{categorical_preprocessor_id}'."
        )

    if numeric_preprocessor_id not in SUPPORTED_GPU_NATIVE_NUMERIC_PREPROCESSOR_IDS:
        raise ValueError(
            "gpu_native preprocessing currently supports numeric_preprocessor in "
            f"{sorted(SUPPORTED_GPU_NATIVE_NUMERIC_PREPROCESSOR_IDS)} only. "
            f"Got '{numeric_preprocessor_id}'."
        )

    return GpuNativeFrequencyPreprocessor(
        feature_columns=feature_schema.feature_columns,
        numeric_columns=feature_schema.numeric_columns,
        categorical_columns=feature_schema.categorical_columns,
        numeric_preprocessor_id=numeric_preprocessor_id,
    )
