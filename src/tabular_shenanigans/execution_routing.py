from dataclasses import dataclass

from tabular_shenanigans.runtime_execution import (
    CPU_GPU_BACKEND,
    NATIVE_GPU_BACKEND,
    PATCH_GPU_BACKEND,
    RuntimeCapabilitySnapshot,
    RuntimeExecutionContext,
)


GPU_EXECUTION_PATHS = (NATIVE_GPU_BACKEND, PATCH_GPU_BACKEND)


@dataclass(frozen=True)
class ModelExecutionRoutingKey:
    task_type: str
    model_family: str
    numeric_preprocessor: str
    categorical_preprocessor: str

    def to_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.task_type,
            self.model_family,
            self.numeric_preprocessor,
            self.categorical_preprocessor,
        )


def _register_model_paths(
    registry: dict[tuple[str, str, str, str], tuple[str, ...]],
    *,
    task_types: tuple[str, ...],
    model_family: str,
    numeric_preprocessors: tuple[str, ...],
    categorical_preprocessors: tuple[str, ...],
    gpu_paths: tuple[str, ...],
) -> None:
    for task_type in task_types:
        for numeric_preprocessor in numeric_preprocessors:
            for categorical_preprocessor in categorical_preprocessors:
                key = ModelExecutionRoutingKey(
                    task_type=task_type,
                    model_family=model_family,
                    numeric_preprocessor=numeric_preprocessor,
                    categorical_preprocessor=categorical_preprocessor,
                )
                registry[key.to_tuple()] = gpu_paths


def _build_gpu_support_registry() -> dict[tuple[str, str, str, str], tuple[str, ...]]:
    registry: dict[tuple[str, str, str, str], tuple[str, ...]] = {}
    _register_model_paths(
        registry,
        task_types=("binary",),
        model_family="logistic_regression",
        numeric_preprocessors=("median", "standardize", "kbins"),
        categorical_preprocessors=("frequency",),
        gpu_paths=(PATCH_GPU_BACKEND,),
    )
    _register_model_paths(
        registry,
        task_types=("binary",),
        model_family="logistic_regression",
        numeric_preprocessors=("standardize",),
        categorical_preprocessors=("frequency",),
        gpu_paths=(NATIVE_GPU_BACKEND, PATCH_GPU_BACKEND),
    )
    _register_model_paths(
        registry,
        task_types=("binary", "regression"),
        model_family="xgboost",
        numeric_preprocessors=("median", "standardize", "kbins"),
        categorical_preprocessors=("ordinal", "frequency"),
        gpu_paths=(PATCH_GPU_BACKEND,),
    )
    _register_model_paths(
        registry,
        task_types=("binary", "regression"),
        model_family="xgboost",
        numeric_preprocessors=("median", "standardize"),
        categorical_preprocessors=("frequency",),
        gpu_paths=(NATIVE_GPU_BACKEND, PATCH_GPU_BACKEND),
    )
    return registry


GPU_SUPPORT_REGISTRY = _build_gpu_support_registry()


def format_model_execution_routing_key(key: ModelExecutionRoutingKey) -> str:
    return (
        f"(task_type={key.task_type}, model_family={key.model_family}, "
        f"numeric_preprocessor={key.numeric_preprocessor}, "
        f"categorical_preprocessor={key.categorical_preprocessor})"
    )


def get_registered_gpu_execution_paths(key: ModelExecutionRoutingKey) -> tuple[str, ...]:
    return GPU_SUPPORT_REGISTRY.get(key.to_tuple(), ())


def resolve_model_candidate_runtime_execution(
    *,
    requested_compute_target: str,
    requested_gpu_backend: str,
    capabilities: RuntimeCapabilitySnapshot,
    task_type: str,
    model_family: str,
    numeric_preprocessor: str,
    categorical_preprocessor: str,
) -> RuntimeExecutionContext:
    routing_key = ModelExecutionRoutingKey(
        task_type=task_type,
        model_family=model_family,
        numeric_preprocessor=numeric_preprocessor,
        categorical_preprocessor=categorical_preprocessor,
    )
    supported_gpu_paths = get_registered_gpu_execution_paths(routing_key)

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
                "Configured GPU execution is unavailable for "
                f"{format_model_execution_routing_key(routing_key)}. "
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

    if requested_gpu_backend == "native":
        if NATIVE_GPU_BACKEND not in supported_gpu_paths:
            raise RuntimeError(
                "Configured gpu_backend='native' is unsupported for "
                f"{format_model_execution_routing_key(routing_key)}."
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
        if PATCH_GPU_BACKEND not in supported_gpu_paths:
            raise RuntimeError(
                "Configured gpu_backend='patch' is unsupported for "
                f"{format_model_execution_routing_key(routing_key)}."
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

    if supported_gpu_paths:
        return RuntimeExecutionContext(
            requested_compute_target=requested_compute_target,
            resolved_compute_target="gpu",
            requested_gpu_backend=requested_gpu_backend,
            resolved_gpu_backend=supported_gpu_paths[0],
            capabilities=capabilities,
            fallback_reason=None,
            rapids_hooks_installed=False,
        )

    if requested_compute_target == "gpu":
        raise RuntimeError(
            "Configured experiment.runtime.compute_target='gpu' but no supported GPU implementation is "
            f"registered for {format_model_execution_routing_key(routing_key)}."
        )

    return RuntimeExecutionContext(
        requested_compute_target=requested_compute_target,
        resolved_compute_target="cpu",
        requested_gpu_backend=requested_gpu_backend,
        resolved_gpu_backend=CPU_GPU_BACKEND,
        capabilities=capabilities,
        fallback_reason=(
            "No supported GPU implementation is registered for "
            f"{format_model_execution_routing_key(routing_key)}"
        ),
        rapids_hooks_installed=False,
    )
