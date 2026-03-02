from tabular_shenanigans.config import load_config


def main() -> None:
    config = load_config()
    print(f"Config loaded: {config.competition_slug}")


if __name__ == "__main__":
    main()
