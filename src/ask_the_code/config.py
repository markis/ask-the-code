from __future__ import annotations

from argparse import SUPPRESS, ArgumentParser
from functools import cache
from pathlib import Path

from ask_the_code import utils


class ConfigSettingsMixin:
    """Mixin for loading settings from a TOML file and environment variables."""

    llm: str = "ollama"
    store: str = "chroma"

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    marqo_url: str = "http://localhost:8882"
    marqo_model: str = "hf/all_datasets_v4_MiniLM-L6"
    index_name: str = "knowledge-management"

    def init_mixin(self) -> None:
        from dynaconf import Dynaconf

        utils.copy_class_vars_to_instance(ConfigSettingsMixin, self)
        settings = Dynaconf(
            envvar_prefix="ASK",
            root_path=utils.config_home(),
            settings_files=["settings.toml"],
            environments=False,
        )
        for key in ConfigSettingsMixin.__annotations__:
            if key in settings:
                setattr(self, key, settings[key])


class ConfigArgsMixin:
    """Mixin for parsing command-line arguments."""

    question: str | None = None
    help: bool = False
    repo: Path = Path.cwd()
    glob: str = "**/*.md"
    search: bool = False

    def init_mixin(self) -> None:
        p = ArgumentParser(add_help=False)
        _ = p.add_argument("question", type=str, nargs="?")
        _ = p.add_argument("-h", "--help", action="help", default=SUPPRESS)
        _ = p.add_argument("-r", "--repo", type=Path, default=Path.cwd())
        _ = p.add_argument("-g", "--glob", type=str, default=SUPPRESS)
        _ = p.add_argument("-s", "--search", action="store_true")
        _ = p.parse_args(namespace=self)


class Config(ConfigArgsMixin, ConfigSettingsMixin):
    """Configuration class for the CLI."""

    def __init__(self, *, init_mixin: bool = True) -> None:
        if init_mixin:
            ConfigSettingsMixin.init_mixin(self)
            ConfigArgsMixin.init_mixin(self)

    @cache
    @staticmethod
    def create() -> Config:
        return Config()
