from tabular_shenanigans.models import MODEL_REGISTRY, GpuRoutingRule, ModelDefinition, get_model_definition
from tabular_shenanigans.runtime_execution import (
    CPU_GPU_BACKEND,
    NATIVE_GPU_BACKEND,
    PATCH_GPU_BACKEND,
    RuntimeCapabilitySnapshot,
    RuntimeExecutionContext,
)


GPU_EXECUTION_PATHS = (NATIVE_GPU_BACKEND, PATCH_GPU_BACKEND)


def _build_cpu_only_model_families() -> frozenset[str]:
    cpu_only_model_families: set[str] = set()
    for task_registry in MODEL_REGISTRY.values():
        for model_id, model_definition in task_registry.items():
            if model_definition.is_cpu_only:
                cpu_only_model_families.add(model_id)
    return frozenset(cpu_only_model_families)


CPU_ONLY_MODEL_FAMILIES = _build_cpu_only_model_families()


def is_cpu_only_model_family(model_family: str) -> bool:
    return model_family in CPU_ONLY_MODEL_FAMILIES


def _find_matching_gpu_rule(
    model_definition: ModelDefinition,
    has_native_categorical: bool,
) -> GpuRoutingRule | None:
    for rule in model_definition.gpu_routing_rules:
        if rule.requires_native_categorical == has_native_categorical:
            return rule
    return None


def describe_missing_gpu_implementation(model_family: str) -> str:
    if is_cpu_only_model_family(model_family):
        return (
            f"{model_family} is intentionally CPU-only in this runtime because no maintained "
            "official GPU implementation is registered for this model family."
        )
    return f"No supported GPU implementation is registered for model_family={model_family!r}."


def resolve_model_candidate_runtime_execution(
    *,
    requested_compute_target: str,
    requested_gpu_backend: str,
    capabilities: RuntimeCapabilitySnapshot,
    task_type: str,
    model_family: str,
    has_native_categorical: bool,
    has_sparse_numeric: bool,
) -> RuntimeExecutionContext:
    model_definition = get_model_definition(task_type, model_family)
    matched_rule = _find_matching_gpu_rule(model_definition, has_native_categorical)

    if requested_compute_target == "cpu":
        return RuntimeExecutionContext(
            requested_compute_target=requested_compute_target,
            resolved_compute_target="cpu",
            requested_gpu_backend=requested_gpu_backend,
            resolved_gpu_backend=CPU_GPU_BACKEND,
            capabilities=capabilities,
            fallback_reason=None,
            rapids_hooks_installed=False,
        )

    if not capabilities.gpu_available:
        if requested_compute_target == "gpu" or requested_gpu_backend != "auto":
            raise RuntimeError(
                f"Configured GPU execution is unavailable for model_family={model_family!r}. "
                f"Reason: {capabilities.unavailable_reason}"
            )
        return RuntimeExecutionContext(
            requested_compute_target=requested_compute_target,
            resolved_compute_target="cpu",
            requested_gpu_backend=requested_gpu_backend,
            resolved_gpu_backend=CPU_GPU_BACKEND,
            capabilities=capabilities,
            fallback_reason=capabilities.unavailable_reason,
            rapids_hooks_installed=False,
        )

    if matched_rule is not None and matched_rule.rejects_sparse and has_sparse_numeric:
        if matched_rule.supports_dense_fallback:
            return RuntimeExecutionContext(
                requested_compute_target=requested_compute_target,
                resolved_compute_target="gpu",
                requested_gpu_backend=requested_gpu_backend,
                resolved_gpu_backend=matched_rule.gpu_backends[0],
                capabilities=capabilities,
                fallback_reason=None,
                rapids_hooks_installed=False,
                sparse_to_dense_coercion=True,
            )
        if requested_compute_target == "gpu" or requested_gpu_backend != "auto":
            raise RuntimeError(
                f"Model family {model_family!r} does not support sparse numeric representations "
                "with its GPU backend. Use a dense representation or run with compute_target=cpu."
            )
        return RuntimeExecutionContext(
            requested_compute_target=requested_compute_target,
            resolved_compute_target="cpu",
            requested_gpu_backend=requested_gpu_backend,
            resolved_gpu_backend=CPU_GPU_BACKEND,
            capabilities=capabilities,
            fallback_reason=f"GPU backend for {model_family!r} rejects sparse numeric input.",
            rapids_hooks_installed=False,
        )

    if requested_gpu_backend == "native":
        if matched_rule is None or NATIVE_GPU_BACKEND not in matched_rule.gpu_backends:
            raise RuntimeError(
                f"Configured gpu_backend='native' is unsupported for model_family={model_family!r}."
            )
        return RuntimeExecutionContext(
            requested_compute_target=requested_compute_target,
            resolved_compute_target="gpu",
            requested_gpu_backend=requested_gpu_backend,
            resolved_gpu_backend=NATIVE_GPU_BACKEND,
            capabilities=capabilities,
            fallback_reason=None,
            rapids_hooks_installed=False,
        )

    if requested_gpu_backend == "patch":
        if matched_rule is None or PATCH_GPU_BACKEND not in matched_rule.gpu_backends:
            raise RuntimeError(
                f"Configured gpu_backend='patch' is unsupported for model_family={model_family!r}."
            )
        return RuntimeExecutionContext(
            requested_compute_target=requested_compute_target,
            resolved_compute_target="gpu",
            requested_gpu_backend=requested_gpu_backend,
            resolved_gpu_backend=PATCH_GPU_BACKEND,
            capabilities=capabilities,
            fallback_reason=None,
            rapids_hooks_installed=False,
        )

    if matched_rule is not None:
        return RuntimeExecutionContext(
            requested_compute_target=requested_compute_target,
            resolved_compute_target="gpu",
            requested_gpu_backend=requested_gpu_backend,
            resolved_gpu_backend=matched_rule.gpu_backends[0],
            capabilities=capabilities,
            fallback_reason=None,
            rapids_hooks_installed=False,
        )

    return RuntimeExecutionContext(
        requested_compute_target=requested_compute_target,
        resolved_compute_target="cpu",
        requested_gpu_backend=requested_gpu_backend,
        resolved_gpu_backend=CPU_GPU_BACKEND,
        capabilities=capabilities,
        fallback_reason=describe_missing_gpu_implementation(model_family),
        rapids_hooks_installed=False,
    )
