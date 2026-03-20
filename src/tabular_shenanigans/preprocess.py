import numpy as np
import pandas as pd

from tabular_shenanigans.representations.feature_schema import (
    ResolvedFeatureSchema,
    resolve_feature_schema,
    resolve_feature_types,
)


def _validate_column_names(config_name: str, columns: list[str], available_columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in available_columns]
    if missing_columns:
        raise ValueError(f"{config_name} has unknown columns: {missing_columns}")


def _normalize_categorical_series(series: pd.Series, missing_value: str) -> pd.Series:
    return series.astype(object).where(series.notna(), missing_value).astype(str)


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
