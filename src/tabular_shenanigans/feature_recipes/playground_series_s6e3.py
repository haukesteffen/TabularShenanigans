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
FR2_CHARGE_CONSISTENCY_COLUMNS = [
    "charges_per_month",
    "charges_gap",
    "charges_per_tenure",
    "total_vs_expected",
    "tenure_monthlycharges_interaction",
]
FR2_SERVICE_COUNT_COLUMNS = [
    "service_addon_count",
    "streaming_count",
    "support_count",
    "service_count",
    "tenure_service_interaction",
]
FR2_CONTRACT_PAYMENT_COLUMNS = [
    "is_monthly_contract",
    "is_autopay",
]
FR2_BUCKET_PROFILE_COLUMNS = [
    "tenure_bucket",
    "tenure_contract",
    "internet_payment_profile",
    "household_profile",
    "senior_household_profile",
    "paperless_payment_profile",
]
FR2_CONTRACT_PROFILE_COLUMNS = [
    *FR2_CONTRACT_PAYMENT_COLUMNS,
    *FR2_BUCKET_PROFILE_COLUMNS,
]
FR3_DEMOGRAPHICS_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
]
FR3_TENURE_CHARGES_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month",
    "charges_gap",
    "charges_per_tenure",
    "total_vs_expected",
    "tenure_monthlycharges_interaction",
    "tenure_service_interaction",
]
FR3_SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "has_internet_service",
    "service_addon_count",
    "streaming_count",
    "support_count",
    "service_count",
    "tenure_service_interaction",
]
FR3_BILLING_CONTRACT_COLUMNS = [
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


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
    transformed = frame

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


def _transform_v2_ablation_frame(
    frame: pd.DataFrame,
    dropped_columns: list[str],
) -> pd.DataFrame:
    transformed = _transform_v2_frame(frame)
    return transformed.drop(columns=dropped_columns)


def _transform_v3_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _transform_v2_ablation_frame(frame, dropped_columns=FR2_CONTRACT_PROFILE_COLUMNS)


def _transform_v3_ablation_frame(
    frame: pd.DataFrame,
    dropped_columns: list[str],
) -> pd.DataFrame:
    transformed = _transform_v3_frame(frame)
    return transformed.drop(columns=dropped_columns)


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


def build_s6e3_v3_features(
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        x_train_raw,
        dataset_name="train features",
        recipe_id="fr3",
        required_columns=FR2_REQUIRED_COLUMNS,
    )
    _require_columns(
        x_test_raw,
        dataset_name="test features",
        recipe_id="fr3",
        required_columns=FR2_REQUIRED_COLUMNS,
    )
    return _transform_v3_frame(x_train_raw), _transform_v3_frame(x_test_raw)


def _build_s6e3_v2_ablation_features(
    recipe_id: str,
    dropped_columns: list[str],
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        x_train_raw,
        dataset_name="train features",
        recipe_id=recipe_id,
        required_columns=FR2_REQUIRED_COLUMNS,
    )
    _require_columns(
        x_test_raw,
        dataset_name="test features",
        recipe_id=recipe_id,
        required_columns=FR2_REQUIRED_COLUMNS,
    )
    return (
        _transform_v2_ablation_frame(x_train_raw, dropped_columns=dropped_columns),
        _transform_v2_ablation_frame(x_test_raw, dropped_columns=dropped_columns),
    )


def _make_s6e3_v2_ablation_recipe(
    recipe_id: str,
    recipe_name: str,
    recipe_description: str,
    dropped_columns: list[str],
) -> FeatureRecipeDefinition:
    def _transform(
        x_train_raw: pd.DataFrame,
        x_test_raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _build_s6e3_v2_ablation_features(
            recipe_id=recipe_id,
            dropped_columns=dropped_columns,
            x_train_raw=x_train_raw,
            x_test_raw=x_test_raw,
        )

    return FeatureRecipeDefinition(
        recipe_id=recipe_id,
        recipe_name=recipe_name,
        recipe_description=recipe_description,
        transform=_transform,
    )


def _make_s6e3_v3_ablation_recipe(
    recipe_id: str,
    recipe_name: str,
    recipe_description: str,
    dropped_columns: list[str],
) -> FeatureRecipeDefinition:
    def _transform(
        x_train_raw: pd.DataFrame,
        x_test_raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        _require_columns(
            x_train_raw,
            dataset_name="train features",
            recipe_id=recipe_id,
            required_columns=FR2_REQUIRED_COLUMNS,
        )
        _require_columns(
            x_test_raw,
            dataset_name="test features",
            recipe_id=recipe_id,
            required_columns=FR2_REQUIRED_COLUMNS,
        )
        return (
            _transform_v3_ablation_frame(x_train_raw, dropped_columns=dropped_columns),
            _transform_v3_ablation_frame(x_test_raw, dropped_columns=dropped_columns),
        )

    return FeatureRecipeDefinition(
        recipe_id=recipe_id,
        recipe_name=recipe_name,
        recipe_description=recipe_description,
        transform=_transform,
    )


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

S6E3_V3_FEATURE_RECIPE = FeatureRecipeDefinition(
    recipe_id="fr3",
    recipe_name="TelcoChurnFeatureSetV3",
    recipe_description=(
        "Reduced Playground Series S6E3 engineered feature set that keeps the charge-consistency "
        "and service-count features from fr2 while dropping the contract/payment and profile-cross features."
    ),
    transform=build_s6e3_v3_features,
)

S6E3_V3_ABLATE_DEMOGRAPHICS_FEATURE_RECIPE = _make_s6e3_v3_ablation_recipe(
    recipe_id="fr3_ablate_demographics",
    recipe_name="TelcoChurnFeatureSetV3AblateDemographics",
    recipe_description="fr3 ablation variant with the demographics signal family removed.",
    dropped_columns=FR3_DEMOGRAPHICS_COLUMNS,
)

S6E3_V3_ABLATE_TENURE_CHARGES_FEATURE_RECIPE = _make_s6e3_v3_ablation_recipe(
    recipe_id="fr3_ablate_tenure_charges",
    recipe_name="TelcoChurnFeatureSetV3AblateTenureCharges",
    recipe_description="fr3 ablation variant with the tenure and charges signal family removed.",
    dropped_columns=FR3_TENURE_CHARGES_COLUMNS,
)

S6E3_V3_ABLATE_SERVICES_FEATURE_RECIPE = _make_s6e3_v3_ablation_recipe(
    recipe_id="fr3_ablate_services",
    recipe_name="TelcoChurnFeatureSetV3AblateServices",
    recipe_description="fr3 ablation variant with the services signal family removed.",
    dropped_columns=FR3_SERVICE_COLUMNS,
)

S6E3_V3_ABLATE_BILLING_CONTRACT_FEATURE_RECIPE = _make_s6e3_v3_ablation_recipe(
    recipe_id="fr3_ablate_billing_contract",
    recipe_name="TelcoChurnFeatureSetV3AblateBillingContract",
    recipe_description="fr3 ablation variant with the billing/contract raw signal family removed.",
    dropped_columns=FR3_BILLING_CONTRACT_COLUMNS,
)

S6E3_V2_ABLATE_CHARGE_CONSISTENCY_FEATURE_RECIPE = _make_s6e3_v2_ablation_recipe(
    recipe_id="fr2_ablate_charge",
    recipe_name="TelcoChurnFeatureSetV2AblateChargeConsistency",
    recipe_description="fr2 ablation variant with charge-consistency engineered features removed.",
    dropped_columns=FR2_CHARGE_CONSISTENCY_COLUMNS,
)

S6E3_V2_ABLATE_SERVICE_COUNTS_FEATURE_RECIPE = _make_s6e3_v2_ablation_recipe(
    recipe_id="fr2_ablate_service",
    recipe_name="TelcoChurnFeatureSetV2AblateServiceCounts",
    recipe_description="fr2 ablation variant with service-count engineered features removed.",
    dropped_columns=FR2_SERVICE_COUNT_COLUMNS,
)

S6E3_V2_ABLATE_CONTRACT_PAYMENT_FEATURE_RECIPE = _make_s6e3_v2_ablation_recipe(
    recipe_id="fr2_ablate_contract",
    recipe_name="TelcoChurnFeatureSetV2AblateContractPayment",
    recipe_description="fr2 ablation variant with contract/payment engineered features removed.",
    dropped_columns=FR2_CONTRACT_PAYMENT_COLUMNS,
)

S6E3_V2_ABLATE_BUCKET_PROFILE_FEATURE_RECIPE = _make_s6e3_v2_ablation_recipe(
    recipe_id="fr2_ablate_profiles",
    recipe_name="TelcoChurnFeatureSetV2AblateBucketProfiles",
    recipe_description="fr2 ablation variant with tenure/profile cross engineered features removed.",
    dropped_columns=FR2_BUCKET_PROFILE_COLUMNS,
)

S6E3_V2_ABLATE_CONTRACT_PROFILE_FEATURE_RECIPE = _make_s6e3_v2_ablation_recipe(
    recipe_id="fr2_ablate_contract_profiles",
    recipe_name="TelcoChurnFeatureSetV2AblateContractProfiles",
    recipe_description="fr2 ablation variant with contract/payment and tenure/profile engineered features removed.",
    dropped_columns=FR2_CONTRACT_PROFILE_COLUMNS,
)

S6E3_ABLATION_FEATURE_RECIPES = [
    S6E3_V2_ABLATE_CHARGE_CONSISTENCY_FEATURE_RECIPE,
    S6E3_V2_ABLATE_SERVICE_COUNTS_FEATURE_RECIPE,
    S6E3_V2_ABLATE_CONTRACT_PAYMENT_FEATURE_RECIPE,
    S6E3_V2_ABLATE_BUCKET_PROFILE_FEATURE_RECIPE,
    S6E3_V2_ABLATE_CONTRACT_PROFILE_FEATURE_RECIPE,
    S6E3_V3_ABLATE_DEMOGRAPHICS_FEATURE_RECIPE,
    S6E3_V3_ABLATE_TENURE_CHARGES_FEATURE_RECIPE,
    S6E3_V3_ABLATE_SERVICES_FEATURE_RECIPE,
    S6E3_V3_ABLATE_BILLING_CONTRACT_FEATURE_RECIPE,
]
