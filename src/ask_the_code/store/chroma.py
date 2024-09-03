import logging
from collections.abc import Collection
from itertools import islice
from pathlib import Path
from typing import Final
from uuid import UUID

from chromadb import PersistentClient
from chromadb.api import ClientAPI

from ask_the_code.chunkers import markdown_chunker
from ask_the_code.config import Config
from ask_the_code.types import DocSource
from ask_the_code.utils import data_home, get_repo_files, get_working_path

logger: Final = logging.getLogger(__name__)

CHROMA_NAMESPACE: Final = UUID("c0e5b3b8-0b1d-4d4c-8b1f-8a3f4c6b3b4d")
CHROMA_DIR: Final = "chroma"
MAX_BATCH_SIZE: Final = 128


class ChromaStore:
    config: Final[Config]
    client: Final[ClientAPI]
    working_path: Final[Path]

    @property
    def collection_name(self) -> str:
        return f"docs-{self.working_path.name}"

    def __init__(self, config: Config) -> None:
        """Initialize the ChromaStore."""
        self.config = config
        self.working_path = get_working_path(config.repo)

        self.client = PersistentClient(str(data_home() / CHROMA_DIR))

    def create(self) -> None:
        """Create the knowledge store."""
        self.reset_index()
        files = get_repo_files(self.working_path, self.config.glob)
        for file in files:
            self.add_document(file)

    def add_document(self, path: Path) -> None:
        """Add a document to the knowledge store."""
        client = self.client
        collection = client.get_collection(self.collection_name)

        chunks = markdown_chunker(path)

        iterator = iter({"source": source, "text": chunk} for source, chunk in chunks)
        while batch := list(islice(iterator, MAX_BATCH_SIZE)):
            ids = [doc["source"] for doc in batch]
            docs = [doc["text"] for doc in batch]
            collection.upsert(ids=ids, documents=docs)

    def reset_index(self) -> None:
        """Reset the knowledge store."""
        client = self.client
        try:
            client.delete_collection(self.collection_name)
        except ValueError:
            logger.warning("Failed to delete collection, it may not exist.")

        try:
            _ = client.create_collection(self.collection_name)
        except ValueError:
            logger.exception("Failed to create collection.")

    def search(self, query: str) -> Collection[DocSource]:
        """Query the knowledge store for content"""
        client = self.client
        collection = client.get_collection(self.collection_name)
        results = collection.query(query_texts=query, n_results=10)
        if not results or not (documents := results.get("documents")):
            return []

        return [
            DocSource(source=source, text=text)
            for source, text in zip(
                (source for ids in results["ids"] for source in ids),
                (text for texts in documents for text in texts),
            )
        ]
