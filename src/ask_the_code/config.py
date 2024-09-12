from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ask_the_code import utils


class Config(BaseModel):
    """Configuration class for the CLI."""

    llm: str = "ollama"
    store: str = "chroma"
    repo: Path = Path.cwd()
    glob: str = "**/*.md"

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    marqo_url: str = "http://localhost:8882"
    marqo_model: str = "hf/all_datasets_v4_MiniLM-L6"
    index_name: str = "knowledge-management"

    reranker_model: str = "BAAI/bge-reranker-large"

    @staticmethod
    def create(**kwargs: Any) -> Config:
        from dynaconf import Dynaconf

        settings = Dynaconf(
            envvar_prefix="ASK",
            root_path=utils.config_home(),
            settings_files=["settings.toml"],
            environments=False,
        )
        config: dict[str, Any] = {key: settings[key] for key in Config.__annotations__ if key in settings}
        return Config(**config, **kwargs)
