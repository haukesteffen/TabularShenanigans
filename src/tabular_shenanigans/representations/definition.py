from __future__ import annotations

from collections.abc import Mapping, Sequence

from tabular_shenanigans.naming import _short_hash
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
    MatrixOutputKind,
    OutputKind,
    RepresentationComponentConfigLike,
    RepresentationComponentSpec,
    RepresentationConfigLike,
)


def _normalize_component_config(
    component: RepresentationComponentConfigLike,
    field_name: str,
) -> RepresentationComponentSpec:
    component_id = getattr(component, "id", None)
    if not isinstance(component_id, str) or not component_id:
        raise ValueError(f"{field_name} entries must include a non-empty string 'id'.")

    params_fn = getattr(component, "params", None)
    if not callable(params_fn):
        raise ValueError(f"{field_name} entries must expose a params() method.")

    params = params_fn()
    if not isinstance(params, dict):
        raise ValueError(f"{field_name} params() must return a dict.")

    return RepresentationComponentSpec(
        component_id=component_id,
        params={str(key): value for key, value in params.items()},
    )


def _normalize_component_payload(
    component: object,
    field_name: str,
) -> RepresentationComponentSpec:
    if not isinstance(component, Mapping):
        raise ValueError(f"{field_name} entries must be mappings.")

    component_id = component.get("id")
    if not isinstance(component_id, str) or not component_id:
        raise ValueError(f"{field_name} entries must include a non-empty string 'id'.")

    return RepresentationComponentSpec(
        component_id=component_id,
        params={str(key): value for key, value in component.items() if key != "id"},
    )


def _normalize_component_list(
    components: object,
    *,
    field_name: str,
    component_kind: str,
) -> list[RepresentationComponentSpec]:
    if not isinstance(components, Sequence) or isinstance(components, str):
        raise ValueError(f"{field_name} must be a list when provided.")
    if component_kind == "operator" and not components:
        raise ValueError("representation.operators must be a non-empty list.")
    return [
        _normalize_component_payload(component, field_name)
        for component in components
    ]


def normalize_representation_config(
    representation: RepresentationConfigLike,
) -> tuple[tuple[RepresentationComponentSpec, ...], tuple[RepresentationComponentSpec, ...]]:
    operators = getattr(representation, "operators", None)
    pruners = getattr(representation, "pruners", None)
    if not isinstance(operators, Sequence) or isinstance(operators, str) or not operators:
        raise ValueError("representation.operators must be a non-empty list.")
    if pruners is None:
        pruners = ()
    if not isinstance(pruners, Sequence) or isinstance(pruners, str):
        raise ValueError("representation.pruners must be a list when provided.")

    return (
        tuple(
            _normalize_component_config(component, "representation.operators")
            for component in operators
        ),
        tuple(
            _normalize_component_config(component, "representation.pruners")
            for component in pruners
        ),
    )


def normalize_representation_payload(
    representation: object,
) -> tuple[tuple[RepresentationComponentSpec, ...], tuple[RepresentationComponentSpec, ...]]:
    if not isinstance(representation, Mapping):
        raise ValueError("representation must be a mapping.")

    operators = representation.get("operators")
    pruners = representation.get("pruners", [])
    normalized_operators = _normalize_component_list(
        operators,
        field_name="representation.operators",
        component_kind="operator",
    )
    normalized_pruners = _normalize_component_list(
        pruners,
        field_name="representation.pruners",
        component_kind="pruner",
    )
    return tuple(normalized_operators), tuple(normalized_pruners)


def build_representation_fingerprint_payload(
    operators: Sequence[RepresentationComponentSpec],
    pruners: Sequence[RepresentationComponentSpec],
) -> dict[str, object]:
    return {
        "operators": [
            {"id": component.component_id, "params": dict(component.params)}
            for component in operators
        ],
        "pruners": [
            {"id": component.component_id, "params": dict(component.params)}
            for component in pruners
        ],
    }


def build_representation_id(
    operators: Sequence[RepresentationComponentSpec],
    pruners: Sequence[RepresentationComponentSpec],
) -> str:
    fingerprint_payload = build_representation_fingerprint_payload(operators, pruners)
    return f"repr-{_short_hash(fingerprint_payload)}"


def build_representation_id_from_config(representation: RepresentationConfigLike) -> str:
    operators, pruners = normalize_representation_config(representation)
    return build_representation_id(operators, pruners)


def build_representation_id_from_payload(representation: object) -> str:
    operators, pruners = normalize_representation_payload(representation)
    return build_representation_id(operators, pruners)


def _operator_ids(operators: Sequence[RepresentationComponentSpec]) -> set[str]:
    return {operator.component_id for operator in operators}


def representation_has_dense_numeric(operators: Sequence[RepresentationComponentSpec]) -> bool:
    return bool(_operator_ids(operators).intersection(DENSE_PRODUCING_OPERATOR_IDS))


def representation_has_sparse_numeric(operators: Sequence[RepresentationComponentSpec]) -> bool:
    return bool(_operator_ids(operators).intersection(SPARSE_PRODUCING_OPERATOR_IDS))


def representation_has_native_categorical(operators: Sequence[RepresentationComponentSpec]) -> bool:
    return "native_categorical" in _operator_ids(operators)


def representation_has_native_numeric(operators: Sequence[RepresentationComponentSpec]) -> bool:
    return "native_numeric" in _operator_ids(operators)


def representation_has_native_tabular(operators: Sequence[RepresentationComponentSpec]) -> bool:
    return representation_has_native_categorical(operators) or representation_has_native_numeric(operators)


def representation_has_frequency_categorical(operators: Sequence[RepresentationComponentSpec]) -> bool:
    return "frequency_encode_categoricals" in _operator_ids(operators)


def representation_has_cuml_compatible_numerics(operators: Sequence[RepresentationComponentSpec]) -> bool:
    operator_ids = _operator_ids(operators)
    has_native_numeric = "native_numeric" in operator_ids
    has_standardized_numeric = "standardize_numeric" in operator_ids
    return (
        has_standardized_numeric and not has_native_numeric
    ) or (
        has_native_numeric and not has_standardized_numeric
    )


def resolve_representation_output_kinds(
    operators: Sequence[RepresentationComponentSpec],
) -> frozenset[OutputKind]:
    has_dense_numeric = representation_has_dense_numeric(operators)
    has_sparse_numeric = representation_has_sparse_numeric(operators)
    has_native_tabular = representation_has_native_tabular(operators)
    return frozenset(
        kind
        for kind, enabled in (
            ("dense_numeric", has_dense_numeric),
            ("sparse_numeric", has_sparse_numeric),
            ("native_tabular", has_native_tabular),
        )
        if enabled
    )


def resolve_representation_matrix_output_kind(
    operators: Sequence[RepresentationComponentSpec],
) -> MatrixOutputKind:
    has_sparse_numeric = representation_has_sparse_numeric(operators)
    has_native_tabular = representation_has_native_tabular(operators)

    if has_native_tabular and has_sparse_numeric:
        raise ValueError(
            "Representations cannot mix native tabular operators with sparse operators. "
            "Use dense/native operators together or sparse/dense operators together."
        )

    if has_native_tabular:
        return "native_frame"
    if has_sparse_numeric:
        return "sparse_csr"
    return "dense_array"


def validate_representation_definition(
    operators: Sequence[RepresentationComponentSpec],
    pruners: Sequence[RepresentationComponentSpec],
) -> None:
    unsupported_operator_ids = sorted(
        operator.component_id
        for operator in operators
        if operator.component_id not in SUPPORTED_OPERATOR_IDS
    )
    if unsupported_operator_ids:
        raise ValueError(
            f"Unsupported representation operators: {unsupported_operator_ids}. "
            f"Supported operators: {sorted(SUPPORTED_OPERATOR_IDS)}"
        )
    for operator in operators:
        validate_component_params(operator.component_id, operator.params, "operator")

    unsupported_pruner_ids = sorted(
        pruner.component_id
        for pruner in pruners
        if pruner.component_id not in SUPPORTED_PRUNER_IDS
    )
    if unsupported_pruner_ids:
        raise ValueError(
            f"Unsupported representation pruners: {unsupported_pruner_ids}. "
            f"Supported pruners: {sorted(SUPPORTED_PRUNER_IDS)}"
        )
    for pruner in pruners:
        validate_component_params(pruner.component_id, pruner.params, "pruner")

    resolve_representation_matrix_output_kind(operators)


def validate_representation_config(representation: RepresentationConfigLike) -> None:
    operators, pruners = normalize_representation_config(representation)
    validate_representation_definition(operators, pruners)


def validate_representation_payload(representation: object) -> None:
    operators, pruners = normalize_representation_payload(representation)
    validate_representation_definition(operators, pruners)
