def main() -> None:
    from rich.console import Console

    from ask_the_code.cli import cli
    from ask_the_code.config import Config
    from ask_the_code.store import get_store

    config = Config.create()
    store = get_store(config)
    cli(config, store, Console())


if __name__ == "__main__":
    main()
