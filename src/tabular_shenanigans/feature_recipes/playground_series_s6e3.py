import numpy as np
import pandas as pd

from tabular_shenanigans.feature_recipes.base import FeatureRecipeDefinition

FR1_REQUIRED_COLUMNS = [
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

FR2_REQUIRED_COLUMNS = FR1_REQUIRED_COLUMNS + ["PhoneService"]

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
SERVICE_COLUMNS = ["PhoneService", "MultipleLines", *SERVICE_ADDON_COLUMNS]


def _require_columns(
    frame: pd.DataFrame,
    dataset_name: str,
    recipe_id: str,
    required_columns: list[str],
) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Feature recipe '{recipe_id}' requires the Telco churn columns used by "
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


def _automatic_payment_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains("automatic", case=False, regex=False).astype(int)


def _transform_v1_frame(frame: pd.DataFrame) -> pd.DataFrame:
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


def _transform_v2_frame(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = _transform_v1_frame(frame)

    tenure = transformed["tenure"].astype(float)
    monthly_charges = transformed["MonthlyCharges"].astype(float)
    total_charges = transformed["TotalCharges"].astype(float)
    service_count = _build_service_count(transformed, SERVICE_COLUMNS).astype(float)

    transformed["service_count"] = service_count
    transformed["is_monthly_contract"] = transformed["Contract"].astype(str).eq("Month-to-month").astype(int)
    transformed["is_autopay"] = _automatic_payment_flag(transformed["PaymentMethod"])
    transformed["charges_per_tenure"] = monthly_charges / (tenure + 1.0)
    transformed["total_vs_expected"] = total_charges / ((tenure * monthly_charges) + 1.0)
    transformed["tenure_service_interaction"] = tenure * service_count
    return transformed


def build_s6e3_v1_features(
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        x_train_raw,
        dataset_name="train features",
        recipe_id="fr1",
        required_columns=FR1_REQUIRED_COLUMNS,
    )
    _require_columns(
        x_test_raw,
        dataset_name="test features",
        recipe_id="fr1",
        required_columns=FR1_REQUIRED_COLUMNS,
    )
    return _transform_v1_frame(x_train_raw), _transform_v1_frame(x_test_raw)


def build_s6e3_v2_features(
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        x_train_raw,
        dataset_name="train features",
        recipe_id="fr2",
        required_columns=FR2_REQUIRED_COLUMNS,
    )
    _require_columns(
        x_test_raw,
        dataset_name="test features",
        recipe_id="fr2",
        required_columns=FR2_REQUIRED_COLUMNS,
    )
    return _transform_v2_frame(x_train_raw), _transform_v2_frame(x_test_raw)


S6E3_V1_FEATURE_RECIPE = FeatureRecipeDefinition(
    recipe_id="fr1",
    recipe_name="TelcoChurnFeatureSetV1",
    recipe_description="Playground Series S6E3 engineered feature set for the Telco churn schema.",
    transform=build_s6e3_v1_features,
)

S6E3_V2_FEATURE_RECIPE = FeatureRecipeDefinition(
    recipe_id="fr2",
    recipe_name="TelcoChurnFeatureSetV2",
    recipe_description=(
        "Expanded Playground Series S6E3 engineered feature set with contract, payment, "
        "service-count, and charge-consistency features."
    ),
    transform=build_s6e3_v2_features,
)
