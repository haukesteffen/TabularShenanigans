import numpy as np
import pandas as pd

from tabular_shenanigans.feature_recipes.base import FeatureRecipeDefinition

REQUIRED_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

SERVICE_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

STREAMING_COLUMNS = ["StreamingTV", "StreamingMovies"]
SUPPORT_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


def _require_columns(frame: pd.DataFrame, dataset_name: str) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Feature recipe 'fr1' requires the Telco churn columns used by "
            f"playground-series-s6e3. Missing columns in {dataset_name}: {missing_columns}"
        )


def _yes_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).eq("Yes").astype(int)


def _build_service_count(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.concat([_yes_flag(frame[column]) for column in columns], axis=1).sum(axis=1)


def _build_tenure_bucket(series: pd.Series) -> pd.Series:
    buckets = pd.cut(
        series.astype(float),
        bins=[-1.0, 12.0, 24.0, 48.0, np.inf],
        labels=["tenure_00_12", "tenure_13_24", "tenure_25_48", "tenure_49_plus"],
        include_lowest=True,
    )
    return buckets.astype(str)


def _transform_frame(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = frame.copy()

    tenure = transformed["tenure"].astype(float)
    monthly_charges = transformed["MonthlyCharges"].astype(float)
    total_charges = transformed["TotalCharges"].astype(float)
    safe_tenure = tenure.clip(lower=1.0)

    transformed["has_internet_service"] = transformed["InternetService"].astype(str).ne("No").astype(int)
    transformed["service_addon_count"] = _build_service_count(transformed, SERVICE_ADDON_COLUMNS)
    transformed["streaming_count"] = _build_service_count(transformed, STREAMING_COLUMNS)
    transformed["support_count"] = _build_service_count(transformed, SUPPORT_COLUMNS)
    transformed["charges_per_month"] = total_charges / safe_tenure
    transformed["charges_gap"] = total_charges - (tenure * monthly_charges)
    transformed["tenure_monthlycharges_interaction"] = tenure * monthly_charges
    transformed["tenure_bucket"] = _build_tenure_bucket(tenure)
    transformed["tenure_contract"] = transformed["tenure_bucket"] + "__" + transformed["Contract"].astype(str)
    transformed["internet_payment_profile"] = (
        transformed["InternetService"].astype(str) + "__" + transformed["PaymentMethod"].astype(str)
    )
    transformed["household_profile"] = transformed["Partner"].astype(str) + "__" + transformed["Dependents"].astype(str)
    transformed["senior_household_profile"] = (
        transformed["SeniorCitizen"].astype(str) + "__" + transformed["Partner"].astype(str)
    )
    transformed["paperless_payment_profile"] = (
        transformed["PaperlessBilling"].astype(str) + "__" + transformed["PaymentMethod"].astype(str)
    )
    return transformed


def build_s6e3_v1_features(
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(x_train_raw, dataset_name="train features")
    _require_columns(x_test_raw, dataset_name="test features")
    return _transform_frame(x_train_raw), _transform_frame(x_test_raw)


S6E3_V1_FEATURE_RECIPE = FeatureRecipeDefinition(
    recipe_id="fr1",
    recipe_name="TelcoChurnFeatureSetV1",
    recipe_description="Playground Series S6E3 engineered feature set for the Telco churn schema.",
    transform=build_s6e3_v1_features,
)
