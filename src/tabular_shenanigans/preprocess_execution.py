from dataclasses import dataclass

from tabular_shenanigans.runtime_execution import PATCH_GPU_BACKEND, RuntimeExecutionContext

CPU_SKLEARN_PREPROCESSING_BACKEND = "cpu_sklearn"
CPU_FREQUENCY_PREPROCESSING_BACKEND = "cpu_frequency"
CPU_NATIVE_FRAME_PREPROCESSING_BACKEND = "cpu_native_frame"
GPU_CUML_PREPROCESSING_BACKEND = "gpu_cuml"
GPU_PATCH_PREPROCESSING_BACKEND = "gpu_patch"
GPU_NATIVE_FREQUENCY_PREPROCESSING_BACKEND = "gpu_native_frequency"


@dataclass(frozen=True)
class PreprocessingExecutionPlan:
    preprocessing_backend: str
    matrix_output_kind: str

    @property
    def uses_repo_gpu_native_preprocessing(self) -> bool:
        return self.preprocessing_backend == GPU_NATIVE_FREQUENCY_PREPROCESSING_BACKEND


def resolve_preprocessing_execution_plan(
    *,
    runtime_execution_context: RuntimeExecutionContext,
    numeric_preprocessor_id: str,
    categorical_preprocessor_id: str,
    matrix_output_kind: str,
) -> PreprocessingExecutionPlan:
    if runtime_execution_context.requested_compute_target == "cpu":
        if categorical_preprocessor_id == "native":
            return PreprocessingExecutionPlan(
                preprocessing_backend=CPU_NATIVE_FRAME_PREPROCESSING_BACKEND,
                matrix_output_kind=matrix_output_kind,
            )
        if categorical_preprocessor_id == "frequency":
            return PreprocessingExecutionPlan(
                preprocessing_backend=CPU_FREQUENCY_PREPROCESSING_BACKEND,
                matrix_output_kind=matrix_output_kind,
            )
        return PreprocessingExecutionPlan(
            preprocessing_backend=CPU_SKLEARN_PREPROCESSING_BACKEND,
            matrix_output_kind=matrix_output_kind,
        )

    if categorical_preprocessor_id == "native":
        return PreprocessingExecutionPlan(
            preprocessing_backend=CPU_NATIVE_FRAME_PREPROCESSING_BACKEND,
            matrix_output_kind=matrix_output_kind,
        )

    gpu_available = runtime_execution_context.capabilities.gpu_available

    if (
        gpu_available
        and categorical_preprocessor_id == "ordinal"
        and numeric_preprocessor_id in {"median", "standardize", "kbins"}
    ):
        return PreprocessingExecutionPlan(
            preprocessing_backend=GPU_CUML_PREPROCESSING_BACKEND,
            matrix_output_kind="dense_array",
        )

    if (
        gpu_available
        and matrix_output_kind == "dense_array"
        and categorical_preprocessor_id == "onehot"
        and numeric_preprocessor_id in {"median", "standardize", "kbins"}
    ):
        return PreprocessingExecutionPlan(
            preprocessing_backend=GPU_CUML_PREPROCESSING_BACKEND,
            matrix_output_kind="dense_array",
        )

    if (
        gpu_available
        and categorical_preprocessor_id == "frequency"
        and numeric_preprocessor_id in {"median", "standardize", "kbins"}
    ):
        return PreprocessingExecutionPlan(
            preprocessing_backend=GPU_NATIVE_FREQUENCY_PREPROCESSING_BACKEND,
            matrix_output_kind=matrix_output_kind,
        )

    if runtime_execution_context.resolved_gpu_backend == PATCH_GPU_BACKEND:
        return PreprocessingExecutionPlan(
            preprocessing_backend=GPU_PATCH_PREPROCESSING_BACKEND,
            matrix_output_kind=matrix_output_kind,
        )

    if categorical_preprocessor_id == "frequency":
        return PreprocessingExecutionPlan(
            preprocessing_backend=CPU_FREQUENCY_PREPROCESSING_BACKEND,
            matrix_output_kind=matrix_output_kind,
        )

    return PreprocessingExecutionPlan(
        preprocessing_backend=CPU_SKLEARN_PREPROCESSING_BACKEND,
        matrix_output_kind=matrix_output_kind,
    )
