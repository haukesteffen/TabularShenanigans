import sys


def _should_skip_runtime_bootstrap(argv: list[str] | None) -> bool:
    if argv is None:
        return False
    return any(argument in {"-h", "--help"} for argument in argv)


def _apply_runtime_bootstrap(argv: list[str] | None) -> None:
    if _should_skip_runtime_bootstrap(argv):
        return

    from tabular_shenanigans.bootstrap_config import load_bootstrap_runtime_config
    from tabular_shenanigans.runtime_execution import (
        activate_runtime_acceleration,
        export_runtime_execution_context,
        resolve_runtime_execution,
    )

    runtime_config = load_bootstrap_runtime_config()
    runtime_context = resolve_runtime_execution(
        runtime_config.compute_target,
        runtime_config.gpu_backend,
    )
    runtime_context = activate_runtime_acceleration(runtime_context)
    export_runtime_execution_context(runtime_context)


def main(argv: list[str] | None = None) -> None:
    _apply_runtime_bootstrap(argv)

    from tabular_shenanigans.cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
