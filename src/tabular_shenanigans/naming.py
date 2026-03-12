import hashlib
import json


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _short_hash(payload: dict[str, object]) -> str:
    payload_json = json.dumps(_json_ready(payload), sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:8]


def normalize_blend_weights(
    base_candidate_ids: list[str],
    configured_weights: list[float] | None,
) -> list[float]:
    if configured_weights is None:
        equal_weight = 1.0 / len(base_candidate_ids)
        return [equal_weight] * len(base_candidate_ids)

    weight_sum = float(sum(configured_weights))
    return [float(weight / weight_sum) for weight in configured_weights]


def build_model_candidate_id(
    feature_recipe_id: str,
    preprocessing_scheme_id: str,
    model_registry_key: str,
    fingerprint_payload: dict[str, object],
) -> str:
    return (
        f"{feature_recipe_id}--{preprocessing_scheme_id}--{model_registry_key}--"
        f"{_short_hash(fingerprint_payload)}"
    )


def build_blend_candidate_id(
    base_candidate_ids: list[str],
    normalized_weights: list[float],
    fingerprint_payload: dict[str, object],
) -> str:
    if len(base_candidate_ids) != len(normalized_weights):
        raise ValueError("Blend candidate id building requires one normalized weight per base candidate.")
    return f"blend__{_short_hash(fingerprint_payload)}"
