from __future__ import annotations

from tabular_shenanigans.naming import _short_hash
from tabular_shenanigans.representations.compilation import (
    CompiledRepresentation,
    FittedRepresentation,
    compile_representation,
    materialize_feature_bundle,
)
from tabular_shenanigans.representations.feature_schema import (
    ResolvedFeatureSchema,
    resolve_feature_schema,
    resolve_feature_types,
)
from tabular_shenanigans.operator_properties import (
    DENSE_PRODUCING_OPERATOR_IDS,
    SPARSE_PRODUCING_OPERATOR_IDS,
)
from tabular_shenanigans.representations.operators import (
    SUPPORTED_OPERATOR_IDS,
    SUPPORTED_PRUNER_IDS,
    validate_component_params,
)
from tabular_shenanigans.representations.types import (
    FeatureBlock,
    FeatureBundle,
    FitMode,
    MaterializedRepresentation,
    MatrixOutputKind,
    OutputKind,
    RepresentationComponentSpec,
    RepresentationContract,
    RepresentationSpec,
)


def normalize_representation_component_payload(component: object, component_kind: str) -> dict[str, object]:
    if not isinstance(component, dict):
        raise ValueError(f"{component_kind} entries must be mappings.")
    component_id = component.get("id")
    if not isinstance(component_id, str) or not component_id:
        raise ValueError(f"{component_kind} entries must include a non-empty string 'id'.")
    params = {str(key): value for key, value in component.items() if key != "id"}
    return {"id": component_id, "params": params}


def build_representation_spec_from_payload(payload: dict[str, object]) -> RepresentationSpec:
    operators_payload = payload.get("operators")
    pruners_payload = payload.get("pruners", [])
    if not isinstance(operators_payload, list) or not operators_payload:
        raise ValueError("representation.operators must be a non-empty list.")
    if not isinstance(pruners_payload, list):
        raise ValueError("representation.pruners must be a list when provided.")

    normalized_operators = [
        normalize_representation_component_payload(component, "representation.operators")
        for component in operators_payload
    ]
    normalized_pruners = [
        normalize_representation_component_payload(component, "representation.pruners")
        for component in pruners_payload
    ]

    fingerprint_payload = {
        "operators": normalized_operators,
        "pruners": normalized_pruners,
    }
    representation_id = f"repr-{_short_hash(fingerprint_payload)}"
    return RepresentationSpec(
        representation_id=representation_id,
        operators=tuple(
            RepresentationComponentSpec(
                component_id=component["id"],
                params=dict(component["params"]),
            )
            for component in normalized_operators
        ),
        pruners=tuple(
            RepresentationComponentSpec(
                component_id=component["id"],
                params=dict(component["params"]),
            )
            for component in normalized_pruners
        ),
        fingerprint_payload=fingerprint_payload,
    )


def build_representation_contract(
    representation_spec: RepresentationSpec,
) -> RepresentationContract:
    operator_ids = {operator.component_id for operator in representation_spec.operators}

    has_dense_numeric = bool(operator_ids.intersection(DENSE_PRODUCING_OPERATOR_IDS))
    has_sparse_numeric = bool(operator_ids.intersection(SPARSE_PRODUCING_OPERATOR_IDS))
    has_native_categorical = "native_categorical" in operator_ids
    has_native_numeric = "native_numeric" in operator_ids
    has_native_tabular = has_native_categorical or has_native_numeric

    if has_native_tabular and has_sparse_numeric:
        raise ValueError(
            "Representations cannot mix native tabular operators with sparse operators. "
            "Use dense/native operators together or sparse/dense operators together."
        )

    if has_native_tabular:
        matrix_output_kind: MatrixOutputKind = "native_frame"
    elif has_sparse_numeric:
        matrix_output_kind = "sparse_csr"
    else:
        matrix_output_kind = "dense_array"

    has_frequency_categorical = "frequency_encode_categoricals" in operator_ids
    has_cuml_compatible_numerics = (
        "standardize_numeric" in operator_ids and not has_native_numeric
    ) or (
        has_native_numeric and "standardize_numeric" not in operator_ids
    )

    return RepresentationContract(
        representation_id=representation_spec.representation_id,
        output_kinds=frozenset(
            kind
            for kind, enabled in (
                ("dense_numeric", has_dense_numeric),
                ("sparse_numeric", has_sparse_numeric),
                ("native_tabular", has_native_tabular),
            )
            if enabled
        ),
        has_dense_numeric=has_dense_numeric,
        has_sparse_numeric=has_sparse_numeric,
        has_native_tabular=has_native_tabular,
        has_native_categorical=has_native_categorical,
        has_native_numeric=has_native_numeric,
        has_frequency_categorical=has_frequency_categorical,
        has_cuml_compatible_numerics=has_cuml_compatible_numerics,
        matrix_output_kind=matrix_output_kind,
    )


def validate_representation_spec(representation_spec: RepresentationSpec) -> None:
    unsupported_operator_ids = sorted(
        operator.component_id
        for operator in representation_spec.operators
        if operator.component_id not in SUPPORTED_OPERATOR_IDS
    )
    if unsupported_operator_ids:
        raise ValueError(
            f"Unsupported representation operators: {unsupported_operator_ids}. "
            f"Supported operators: {sorted(SUPPORTED_OPERATOR_IDS)}"
        )
    for operator in representation_spec.operators:
        validate_component_params(operator.component_id, operator.params, "operator")

    unsupported_pruner_ids = sorted(
        pruner.component_id
        for pruner in representation_spec.pruners
        if pruner.component_id not in SUPPORTED_PRUNER_IDS
    )
    if unsupported_pruner_ids:
        raise ValueError(
            f"Unsupported representation pruners: {unsupported_pruner_ids}. "
            f"Supported pruners: {sorted(SUPPORTED_PRUNER_IDS)}"
        )
    for pruner in representation_spec.pruners:
        validate_component_params(pruner.component_id, pruner.params, "pruner")

    build_representation_contract(representation_spec)


__all__ = [
    "CompiledRepresentation",
    "FeatureBlock",
    "FeatureBundle",
    "FittedRepresentation",
    "FitMode",
    "MaterializedRepresentation",
    "MatrixOutputKind",
    "OutputKind",
    "RepresentationComponentSpec",
    "RepresentationContract",
    "RepresentationSpec",
    "ResolvedFeatureSchema",
    "SUPPORTED_OPERATOR_IDS",
    "SUPPORTED_PRUNER_IDS",
    "build_representation_contract",
    "build_representation_spec_from_payload",
    "compile_representation",
    "materialize_feature_bundle",
    "resolve_feature_schema",
    "resolve_feature_types",
    "validate_representation_spec",
]
