import numpy as np
import pandas as pd

from tabular_shenanigans.representations.output_adapters import (
    DenseArrayAdapter,
    NativeFrameAdapter,
    SparseCSRAdapter,
)
from tabular_shenanigans.representations.registry import REPRESENTATION_REGISTRY
from tabular_shenanigans.representations.steps import (
    FeatureEngineeringStep,
    FrequencyEncodeStep,
    KBinsStep,
    MedianImputeStep,
    NativeCategoricalStep,
    OneHotEncodeStep,
    OrdinalEncodeStep,
    StandardizeStep,
)
from tabular_shenanigans.representations.types import RepresentationDefinition

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


def _require_columns(frame: pd.DataFrame, recipe_id: str, required_columns: list[str]) -> None:
    missing = [c for c in required_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Representation '{recipe_id}' requires Telco churn columns. "
            f"Missing columns: {missing}"
        )


def _yes_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).eq("Yes").astype(int)


def _build_service_count(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.concat([_yes_flag(frame[c]) for c in columns], axis=1).sum(axis=1)


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


def _transform_v2_ablation_frame(frame: pd.DataFrame, dropped_columns: list[str]) -> pd.DataFrame:
    return _transform_v2_frame(frame).drop(columns=dropped_columns)


def _transform_v3_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _transform_v2_ablation_frame(frame, dropped_columns=FR2_CONTRACT_PROFILE_COLUMNS)


def _transform_v3_ablation_frame(frame: pd.DataFrame, dropped_columns: list[str]) -> pd.DataFrame:
    return _transform_v3_frame(frame).drop(columns=dropped_columns)


def _make_fr1_step() -> FeatureEngineeringStep:
    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, "fr1", FR1_REQUIRED_COLUMNS)
        return _transform_v1_frame(frame.copy())
    return FeatureEngineeringStep(step_id="s6e3_fr1", transform_fn=_transform)


def _make_fr2_step() -> FeatureEngineeringStep:
    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, "fr2", FR2_REQUIRED_COLUMNS)
        return _transform_v2_frame(frame.copy())
    return FeatureEngineeringStep(step_id="s6e3_fr2", transform_fn=_transform)


def _make_fr3_step() -> FeatureEngineeringStep:
    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, "fr3", FR2_REQUIRED_COLUMNS)
        return _transform_v3_frame(frame.copy())
    return FeatureEngineeringStep(step_id="s6e3_fr3", transform_fn=_transform)


def _make_fr2_ablation_step(step_id: str, dropped_columns: list[str]) -> FeatureEngineeringStep:
    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, step_id, FR2_REQUIRED_COLUMNS)
        return _transform_v2_ablation_frame(frame.copy(), dropped_columns=dropped_columns)
    return FeatureEngineeringStep(step_id=step_id, transform_fn=_transform)


def _make_fr3_ablation_step(step_id: str, dropped_columns: list[str]) -> FeatureEngineeringStep:
    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, step_id, FR2_REQUIRED_COLUMNS)
        return _transform_v3_ablation_frame(frame.copy(), dropped_columns=dropped_columns)
    return FeatureEngineeringStep(step_id=step_id, transform_fn=_transform)


def _register_s6e3_representation(
    fe_step: FeatureEngineeringStep,
    num_id: str,
    cat_id: str,
    num_step,
    cat_step,
    adapter,
    representation_name: str,
) -> None:
    representation_id = f"{fe_step.step_id}-{num_id}-{cat_id}"
    REPRESENTATION_REGISTRY[representation_id] = RepresentationDefinition(
        representation_id=representation_id,
        representation_name=representation_name,
        steps=(fe_step, num_step, cat_step),
        output_adapter=adapter,
        numeric_preprocessor_id=num_id,
        categorical_preprocessor_id=cat_id,
    )


def _register_s6e3_representations() -> None:
    fe_steps = [
        (_make_fr1_step(), "TelcoChurnV1"),
        (_make_fr2_step(), "TelcoChurnV2"),
        (_make_fr3_step(), "TelcoChurnV3"),
        (_make_fr2_ablation_step("s6e3_fr2_ablate_charge", FR2_CHARGE_CONSISTENCY_COLUMNS), "TelcoChurnV2AblateCharge"),
        (_make_fr2_ablation_step("s6e3_fr2_ablate_service", FR2_SERVICE_COUNT_COLUMNS), "TelcoChurnV2AblateService"),
        (_make_fr2_ablation_step("s6e3_fr2_ablate_contract", FR2_CONTRACT_PAYMENT_COLUMNS), "TelcoChurnV2AblateContract"),
        (_make_fr2_ablation_step("s6e3_fr2_ablate_profiles", FR2_BUCKET_PROFILE_COLUMNS), "TelcoChurnV2AblateProfiles"),
        (_make_fr2_ablation_step("s6e3_fr2_ablate_contract_profiles", FR2_CONTRACT_PROFILE_COLUMNS), "TelcoChurnV2AblateContractProfiles"),
        (_make_fr3_ablation_step("s6e3_fr3_ablate_demographics", FR3_DEMOGRAPHICS_COLUMNS), "TelcoChurnV3AblateDemographics"),
        (_make_fr3_ablation_step("s6e3_fr3_ablate_tenure_charges", FR3_TENURE_CHARGES_COLUMNS), "TelcoChurnV3AblateTenureCharges"),
        (_make_fr3_ablation_step("s6e3_fr3_ablate_services", FR3_SERVICE_COLUMNS), "TelcoChurnV3AblateServices"),
        (_make_fr3_ablation_step("s6e3_fr3_ablate_billing_contract", FR3_BILLING_CONTRACT_COLUMNS), "TelcoChurnV3AblateBillingContract"),
    ]

    preproc_configs = [
        ("median", MedianImputeStep(), "ordinal", OrdinalEncodeStep(), DenseArrayAdapter()),
        ("median", MedianImputeStep(), "onehot", OneHotEncodeStep(), DenseArrayAdapter()),
        ("median", MedianImputeStep(), "frequency", FrequencyEncodeStep(), DenseArrayAdapter()),
        ("median", MedianImputeStep(), "native", NativeCategoricalStep(), NativeFrameAdapter()),
        ("standardize", StandardizeStep(), "ordinal", OrdinalEncodeStep(), DenseArrayAdapter()),
        ("standardize", StandardizeStep(), "onehot", OneHotEncodeStep(), DenseArrayAdapter()),
        ("standardize", StandardizeStep(), "frequency", FrequencyEncodeStep(), DenseArrayAdapter()),
        ("standardize", StandardizeStep(), "native", NativeCategoricalStep(), NativeFrameAdapter()),
        ("kbins", KBinsStep(), "ordinal", OrdinalEncodeStep(), DenseArrayAdapter()),
        ("kbins", KBinsStep(sparse_output=True), "onehot", OneHotEncodeStep(sparse_output=True), SparseCSRAdapter()),
        ("kbins", KBinsStep(), "frequency", FrequencyEncodeStep(), DenseArrayAdapter()),
        ("kbins", KBinsStep(), "native", NativeCategoricalStep(), NativeFrameAdapter()),
    ]

    for fe_step, fe_name in fe_steps:
        for num_id, num_step, cat_id, cat_step, adapter in preproc_configs:
            _register_s6e3_representation(
                fe_step=fe_step,
                num_id=num_id,
                cat_id=cat_id,
                num_step=num_step,
                cat_step=cat_step,
                adapter=adapter,
                representation_name=f"{fe_name}{num_id.capitalize()}{cat_id.capitalize()}",
            )


_register_s6e3_representations()
