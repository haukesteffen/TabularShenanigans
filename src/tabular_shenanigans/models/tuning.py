from __future__ import annotations

import math
import random
from typing import Any


def sample_param(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Tuning search space list cannot be empty.")
        return rng.choice(spec)

    if not isinstance(spec, dict):
        return spec

    ptype = str(spec.get("type", "")).lower()
    if ptype == "categorical":
        choices = spec.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise ValueError("Categorical tuning spec must include non-empty 'choices'.")
        return rng.choice(choices)

    if ptype == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        if step <= 0:
            raise ValueError("Integer tuning spec 'step' must be > 0.")
        values = list(range(low, high + 1, step))
        if not values:
            raise ValueError("Integer tuning spec produced no values.")
        return rng.choice(values)

    if ptype == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        use_log = bool(spec.get("log", False))
        if use_log:
            if low <= 0 or high <= 0:
                raise ValueError("Log-scale float tuning requires low/high > 0.")
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)

    if ptype:
        raise ValueError(f"Unsupported tuning spec type '{ptype}'.")

    # If dict has no type, treat it as fixed parameter mapping value.
    return spec


def sample_params(search_space: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("training.tune.search_space must be a non-empty mapping.")

    sampled: dict[str, Any] = {}
    for key, spec in search_space.items():
        sampled[key] = sample_param(spec, rng)
    return sampled
