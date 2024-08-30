from collections.abc import Collection
from functools import cache
from pathlib import Path
from typing import Protocol

from ask_the_code.config import Config
from ask_the_code.types import DocSource


class Store(Protocol):
    def create(self) -> None: ...
    def add_document(self, path: Path) -> None: ...
    def reset_index(self) -> None: ...
    def search(self, query: str) -> Collection[DocSource]: ...


@cache
def get_store(config: Config) -> Store:
    if config.store == "chroma":
        from ask_the_code.store.chroma import ChromaStore

        return ChromaStore(config)

    if config.store == "marqo":
        from ask_the_code.store.marqo import MarqoStore

        return MarqoStore(config)

    raise ValueError(f"Unknown store: {config.store}")
