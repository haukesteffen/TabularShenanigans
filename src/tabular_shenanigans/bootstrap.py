from dataclasses import dataclass
import sys


def _should_skip_runtime_bootstrap(argv: list[str] | None) -> bool:
    if argv is None:
        return False
    return any(argument in {"-h", "--help"} for argument in argv)


@dataclass(frozen=True)
class BootstrapCliSelection:
    stage: str | None
    candidate_index: int | None
    candidate_id_requested: bool


def _extract_option_value(argv: list[str], option_name: str) -> str | None:
    for argument_index, argument in enumerate(argv):
        if argument == option_name:
            if argument_index + 1 >= len(argv):
                raise ValueError(f"Missing value for {option_name}.")
            return argv[argument_index + 1]
        if argument.startswith(f"{option_name}="):
            return argument.split("=", 1)[1]
    return None


def _parse_bootstrap_cli_selection(argv: list[str] | None) -> BootstrapCliSelection:
    if not argv:
        return BootstrapCliSelection(stage=None, candidate_index=None, candidate_id_requested=False)

    known_stages = {"fetch", "prepare", "eda", "train", "submit", "refresh-submissions"}
    stage = argv[0] if argv[0] in known_stages else None
    option_argv = argv[1:] if stage is not None else argv
    index_value = _extract_option_value(option_argv, "--index")
    candidate_index = int(index_value) if index_value is not None else None
    candidate_id_requested = _extract_option_value(option_argv, "--candidate-id") is not None
    return BootstrapCliSelection(
        stage=stage,
        candidate_index=candidate_index,
        candidate_id_requested=candidate_id_requested,
    )


def _stage_uses_training_runtime(stage: str | None) -> bool:
    return stage in {None, "train"}


def _resolve_selected_candidate_indices(
    runtime_config,
    selection: BootstrapCliSelection,
) -> tuple[int, ...]:
    if selection.candidate_index is not None:
        resolved_index = selection.candidate_index - 1
        if resolved_index < 0 or resolved_index >= len(runtime_config.candidates):
            raise ValueError(
                f"--index must be between 1 and {len(runtime_config.candidates)}. "
                f"Got {selection.candidate_index}."
            )
        return (resolved_index,)
    return tuple(range(len(runtime_config.candidates)))


def _raise_mixed_patch_runtime_error(selection: BootstrapCliSelection) -> None:
    if selection.candidate_id_requested:
        raise RuntimeError(
            "This config mixes gpu_patch candidates with non-gpu_patch candidates. "
            "Bootstrap happens before --candidate-id can be resolved safely, so this train invocation cannot "
            "install RAPIDS hooks without risking the wrong process-wide runtime. "
            "Use `uv run python main.py train --index <n>` or split the batch."
        )
    raise RuntimeError(
        "Batch train cannot mix gpu_patch candidates with non-gpu_patch candidates in one process because "
        "RAPIDS hook installation is process-global. Split the run with `uv run python main.py train --index <n>` "
        "or separate invocations."
    )


def _apply_runtime_bootstrap(argv: list[str] | None) -> None:
    if _should_skip_runtime_bootstrap(argv):
        return

    from tabular_shenanigans.bootstrap_config import load_bootstrap_runtime_config
    from tabular_shenanigans.execution_routing import resolve_model_candidate_runtime_execution
    from tabular_shenanigans.runtime_execution import (
        PATCH_GPU_BACKEND,
        activate_runtime_acceleration,
        detect_runtime_capabilities,
        export_runtime_execution_context,
    )

    runtime_config = load_bootstrap_runtime_config()
    selection = _parse_bootstrap_cli_selection(argv)
    if not _stage_uses_training_runtime(selection.stage):
        return

    selected_candidate_indices = _resolve_selected_candidate_indices(runtime_config, selection)
    if not selected_candidate_indices or runtime_config.task_type is None:
        return

    capabilities = detect_runtime_capabilities()
    from tabular_shenanigans.representations.registry import REPRESENTATION_REGISTRY

    selected_model_contexts = []
    for candidate_index in selected_candidate_indices:
        candidate = runtime_config.candidates[candidate_index]
        if (
            candidate.candidate_type != "model"
            or candidate.model_family is None
            or candidate.representation_id is None
        ):
            continue
        representation_definition = REPRESENTATION_REGISTRY.get(candidate.representation_id)
        if representation_definition is None:
            continue
        selected_model_contexts.append(
            resolve_model_candidate_runtime_execution(
                requested_compute_target=runtime_config.compute_target,
                requested_gpu_backend=runtime_config.gpu_backend,
                capabilities=capabilities,
                task_type=runtime_config.task_type,
                model_family=candidate.model_family,
                numeric_preprocessor=representation_definition.numeric_preprocessor_id,
                categorical_preprocessor=representation_definition.categorical_preprocessor_id,
            )
        )

    if not selected_model_contexts:
        return

    resolved_backends = {context.resolved_gpu_backend for context in selected_model_contexts}
    if PATCH_GPU_BACKEND in resolved_backends and len(resolved_backends) > 1:
        _raise_mixed_patch_runtime_error(selection)

    if resolved_backends == {PATCH_GPU_BACKEND}:
        runtime_context = activate_runtime_acceleration(selected_model_contexts[0])
        export_runtime_execution_context(runtime_context)
        return


def main(argv: list[str] | None = None) -> None:
    _apply_runtime_bootstrap(argv)

    from tabular_shenanigans.cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
