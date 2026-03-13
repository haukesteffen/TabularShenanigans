import sys


def _apply_runtime_bootstrap() -> None:
    # Future RAPIDS import hooks must run here before any app module imports.
    return


def main(argv: list[str] | None = None) -> None:
    _apply_runtime_bootstrap()

    from tabular_shenanigans.cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
