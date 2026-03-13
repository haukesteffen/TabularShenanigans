import importlib
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

REQUESTED_COMPUTE_TARGET_ENV = "TABULAR_SHENANIGANS_REQUESTED_COMPUTE_TARGET"
RESOLVED_COMPUTE_TARGET_ENV = "TABULAR_SHENANIGANS_RESOLVED_COMPUTE_TARGET"
GPU_AVAILABLE_ENV = "TABULAR_SHENANIGANS_GPU_AVAILABLE"
FALLBACK_REASON_ENV = "TABULAR_SHENANIGANS_RUNTIME_FALLBACK_REASON"
PLATFORM_SYSTEM_ENV = "TABULAR_SHENANIGANS_RUNTIME_PLATFORM_SYSTEM"
VISIBLE_NVIDIA_DEVICES_ENV = "TABULAR_SHENANIGANS_VISIBLE_NVIDIA_DEVICES"
ACCELERATION_BACKEND_ENV = "TABULAR_SHENANIGANS_ACCELERATION_BACKEND"
RAPIDS_HOOKS_INSTALLED_ENV = "TABULAR_SHENANIGANS_RAPIDS_HOOKS_INSTALLED"
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
CPU_ACCELERATION_BACKEND = "cpu"
PENDING_ACCELERATION_BACKEND = "pending"
RAPIDS_ACCELERATION_BACKEND = "rapids"


@dataclass(frozen=True)
class RuntimeCapabilitySnapshot:
    platform_system: str
    gpu_available: bool
    visible_nvidia_devices: tuple[str, ...]
    cuda_visible_devices: str | None
    unavailable_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "platform_system": self.platform_system,
            "gpu_available": self.gpu_available,
            "visible_nvidia_devices": list(self.visible_nvidia_devices),
            "cuda_visible_devices": self.cuda_visible_devices,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class RuntimeExecutionContext:
    requested_compute_target: str
    resolved_compute_target: str
    capabilities: RuntimeCapabilitySnapshot
    fallback_reason: str | None
    acceleration_backend: str
    rapids_hooks_installed: bool

    @property
    def gpu_available(self) -> bool:
        return self.capabilities.gpu_available

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_compute_target": self.requested_compute_target,
            "resolved_compute_target": self.resolved_compute_target,
            "gpu_available": self.gpu_available,
            "fallback_reason": self.fallback_reason,
            "acceleration_backend": self.acceleration_backend,
            "rapids_hooks_installed": self.rapids_hooks_installed,
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass(frozen=True)
class RapidsHookInstallers:
    cudf_pandas_install: Callable[[], object]
    cuml_accel_install: Callable[[], object]


def _validate_compute_target(value: str) -> str:
    normalized_value = value.strip().lower()
    if normalized_value in {"auto", "cpu", "gpu"}:
        return normalized_value
    raise ValueError(
        f"Unsupported compute target '{value}'. Expected one of ['auto', 'cpu', 'gpu']."
    )


def _cuda_visible_devices_disables_gpu(cuda_visible_devices: str | None) -> bool:
    if cuda_visible_devices is None:
        return False
    normalized_value = cuda_visible_devices.strip().lower()
    return normalized_value in {"", "-1", "none", "void"}


def _discover_visible_nvidia_devices() -> tuple[str, ...]:
    device_patterns = (
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
    )
    device_paths = [pattern for pattern in device_patterns if Path(pattern).exists()]
    device_paths.extend(str(path) for path in sorted(Path("/dev").glob("nvidia[0-9]*")))
    return tuple(dict.fromkeys(device_paths))


@lru_cache(maxsize=1)
def detect_runtime_capabilities() -> RuntimeCapabilitySnapshot:
    platform_system = platform.system()
    cuda_visible_devices = os.getenv(CUDA_VISIBLE_DEVICES_ENV)
    visible_nvidia_devices = _discover_visible_nvidia_devices()

    if platform_system != "Linux":
        return RuntimeCapabilitySnapshot(
            platform_system=platform_system,
            gpu_available=False,
            visible_nvidia_devices=visible_nvidia_devices,
            cuda_visible_devices=cuda_visible_devices,
            unavailable_reason=f"platform '{platform_system}' does not expose NVIDIA CUDA devices",
        )

    if _cuda_visible_devices_disables_gpu(cuda_visible_devices):
        return RuntimeCapabilitySnapshot(
            platform_system=platform_system,
            gpu_available=False,
            visible_nvidia_devices=visible_nvidia_devices,
            cuda_visible_devices=cuda_visible_devices,
            unavailable_reason=f"{CUDA_VISIBLE_DEVICES_ENV} disables GPU visibility",
        )

    if not visible_nvidia_devices:
        return RuntimeCapabilitySnapshot(
            platform_system=platform_system,
            gpu_available=False,
            visible_nvidia_devices=visible_nvidia_devices,
            cuda_visible_devices=cuda_visible_devices,
            unavailable_reason="no visible NVIDIA device files were detected under /dev",
        )

    return RuntimeCapabilitySnapshot(
        platform_system=platform_system,
        gpu_available=True,
        visible_nvidia_devices=visible_nvidia_devices,
        cuda_visible_devices=cuda_visible_devices,
        unavailable_reason=None,
    )


def resolve_runtime_execution(requested_compute_target: str) -> RuntimeExecutionContext:
    normalized_target = _validate_compute_target(requested_compute_target)
    capabilities = detect_runtime_capabilities()

    if normalized_target == "cpu":
        return RuntimeExecutionContext(
            requested_compute_target=normalized_target,
            resolved_compute_target="cpu",
            capabilities=capabilities,
            fallback_reason=None,
            acceleration_backend=CPU_ACCELERATION_BACKEND,
            rapids_hooks_installed=False,
        )

    if normalized_target == "gpu":
        if capabilities.gpu_available:
            return RuntimeExecutionContext(
                requested_compute_target=normalized_target,
                resolved_compute_target="gpu",
                capabilities=capabilities,
                fallback_reason=None,
                acceleration_backend=PENDING_ACCELERATION_BACKEND,
                rapids_hooks_installed=False,
            )
        raise RuntimeError(
            "Configured experiment.runtime.compute_target='gpu' but GPU execution is unavailable. "
            f"Detection summary: {describe_runtime_capabilities(capabilities)}"
        )

    if capabilities.gpu_available:
        return RuntimeExecutionContext(
            requested_compute_target=normalized_target,
            resolved_compute_target="gpu",
            capabilities=capabilities,
            fallback_reason=None,
            acceleration_backend=PENDING_ACCELERATION_BACKEND,
            rapids_hooks_installed=False,
        )

    return RuntimeExecutionContext(
        requested_compute_target=normalized_target,
        resolved_compute_target="cpu",
        capabilities=capabilities,
        fallback_reason=capabilities.unavailable_reason,
        acceleration_backend=CPU_ACCELERATION_BACKEND,
        rapids_hooks_installed=False,
    )


def _load_rapids_hook_installers() -> RapidsHookInstallers:
    try:
        cudf_pandas = importlib.import_module("cudf.pandas")
    except ImportError as exc:
        raise RuntimeError("cudf.pandas is not importable in this environment") from exc

    try:
        cuml_accel = importlib.import_module("cuml.accel")
    except ImportError as exc:
        raise RuntimeError("cuml.accel is not importable in this environment") from exc

    cudf_pandas_install = getattr(cudf_pandas, "install", None)
    if not callable(cudf_pandas_install):
        raise RuntimeError("cudf.pandas.install() is unavailable in this environment")

    cuml_accel_install = getattr(cuml_accel, "install", None)
    if not callable(cuml_accel_install):
        raise RuntimeError("cuml.accel.install() is unavailable in this environment")

    return RapidsHookInstallers(
        cudf_pandas_install=cudf_pandas_install,
        cuml_accel_install=cuml_accel_install,
    )


def activate_runtime_acceleration(context: RuntimeExecutionContext) -> RuntimeExecutionContext:
    if context.resolved_compute_target != "gpu":
        return context

    try:
        rapids_installers = _load_rapids_hook_installers()
    except RuntimeError as exc:
        if context.requested_compute_target == "gpu":
            raise RuntimeError(
                "Configured experiment.runtime.compute_target='gpu' but RAPIDS hooks are unavailable. "
                f"Reason: {exc}. Detection summary: {describe_runtime_capabilities(context.capabilities)}"
            ) from exc
        return replace(
            context,
            resolved_compute_target="cpu",
            fallback_reason=f"RAPIDS hooks unavailable: {exc}",
            acceleration_backend=CPU_ACCELERATION_BACKEND,
            rapids_hooks_installed=False,
        )

    # Once either install call runs, rollback is not guaranteed. Treat install failures as hard errors.
    try:
        rapids_installers.cudf_pandas_install()
        rapids_installers.cuml_accel_install()
    except Exception as exc:
        raise RuntimeError(
            "RAPIDS hook installation failed after preflight succeeded. "
            f"Reason: {exc}. Detection summary: {describe_runtime_capabilities(context.capabilities)}"
        ) from exc

    return replace(
        context,
        acceleration_backend=RAPIDS_ACCELERATION_BACKEND,
        rapids_hooks_installed=True,
    )


def export_runtime_execution_context(context: RuntimeExecutionContext) -> None:
    os.environ[REQUESTED_COMPUTE_TARGET_ENV] = context.requested_compute_target
    os.environ[RESOLVED_COMPUTE_TARGET_ENV] = context.resolved_compute_target
    os.environ[GPU_AVAILABLE_ENV] = "true" if context.gpu_available else "false"
    os.environ[PLATFORM_SYSTEM_ENV] = context.capabilities.platform_system
    os.environ[VISIBLE_NVIDIA_DEVICES_ENV] = ",".join(context.capabilities.visible_nvidia_devices)
    os.environ[ACCELERATION_BACKEND_ENV] = context.acceleration_backend
    os.environ[RAPIDS_HOOKS_INSTALLED_ENV] = "true" if context.rapids_hooks_installed else "false"
    if context.fallback_reason is None:
        os.environ.pop(FALLBACK_REASON_ENV, None)
    else:
        os.environ[FALLBACK_REASON_ENV] = context.fallback_reason


def _parse_exported_runtime_execution_context() -> RuntimeExecutionContext | None:
    requested_compute_target = os.getenv(REQUESTED_COMPUTE_TARGET_ENV)
    resolved_compute_target = os.getenv(RESOLVED_COMPUTE_TARGET_ENV)
    gpu_available = os.getenv(GPU_AVAILABLE_ENV)
    platform_system = os.getenv(PLATFORM_SYSTEM_ENV)
    acceleration_backend = os.getenv(ACCELERATION_BACKEND_ENV)
    rapids_hooks_installed = os.getenv(RAPIDS_HOOKS_INSTALLED_ENV)

    if (
        requested_compute_target is None
        or resolved_compute_target is None
        or gpu_available is None
        or platform_system is None
        or acceleration_backend is None
        or rapids_hooks_installed is None
    ):
        return None

    visible_nvidia_devices_raw = os.getenv(VISIBLE_NVIDIA_DEVICES_ENV, "")
    visible_nvidia_devices = tuple(
        device_path for device_path in visible_nvidia_devices_raw.split(",") if device_path
    )
    fallback_reason = os.getenv(FALLBACK_REASON_ENV)
    capabilities = RuntimeCapabilitySnapshot(
        platform_system=platform_system,
        gpu_available=gpu_available == "true",
        visible_nvidia_devices=visible_nvidia_devices,
        cuda_visible_devices=os.getenv(CUDA_VISIBLE_DEVICES_ENV),
        unavailable_reason=fallback_reason,
    )
    return RuntimeExecutionContext(
        requested_compute_target=requested_compute_target,
        resolved_compute_target=resolved_compute_target,
        capabilities=capabilities,
        fallback_reason=fallback_reason,
        acceleration_backend=acceleration_backend,
        rapids_hooks_installed=rapids_hooks_installed == "true",
    )


def get_runtime_execution_context(requested_compute_target: str = "auto") -> RuntimeExecutionContext:
    exported_context = _parse_exported_runtime_execution_context()
    if exported_context is not None:
        return exported_context
    return resolve_runtime_execution(requested_compute_target)


def describe_runtime_capabilities(capabilities: RuntimeCapabilitySnapshot) -> str:
    visible_devices = ",".join(capabilities.visible_nvidia_devices) or "none"
    cuda_visible_devices = capabilities.cuda_visible_devices or "unset"
    summary = (
        f"platform={capabilities.platform_system}, "
        f"gpu_available={capabilities.gpu_available}, "
        f"visible_nvidia_devices={visible_devices}, "
        f"{CUDA_VISIBLE_DEVICES_ENV}={cuda_visible_devices}"
    )
    if capabilities.unavailable_reason is not None:
        summary = f"{summary}, unavailable_reason={capabilities.unavailable_reason}"
    return summary


def format_runtime_execution_context(context: RuntimeExecutionContext) -> str:
    summary = (
        f"requested_compute_target={context.requested_compute_target}, "
        f"resolved_compute_target={context.resolved_compute_target}, "
        f"gpu_available={context.gpu_available}, "
        f"acceleration_backend={context.acceleration_backend}, "
        f"rapids_hooks_installed={context.rapids_hooks_installed}, "
        f"platform={context.capabilities.platform_system}"
    )
    visible_devices = ",".join(context.capabilities.visible_nvidia_devices)
    if visible_devices:
        summary = f"{summary}, visible_nvidia_devices={visible_devices}"
    if context.fallback_reason is not None:
        summary = f"{summary}, fallback_reason={context.fallback_reason}"
    return summary
