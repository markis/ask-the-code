from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Protocol

from ask_the_code.config import Config
from ask_the_code.types import DocSource


class Store(Protocol):
    def create(self) -> Iterable[str]:
        """Create the store."""
        ...

    def add_document(self, path: Path) -> None:
        """Add a document to the store."""
        ...

    def reset_index(self) -> None:
        """Reset the index."""
        ...

    def search(self, query: str) -> Collection[DocSource]:
        """Search the index for a query."""
        ...


def get_store(config: Config) -> Store:
    if config.store == "chroma":
        from ask_the_code.store.chroma import ChromaStore

        return ChromaStore(config)

    err_msg = f"Unknown store: {config.store}"
    raise ValueError(err_msg)
