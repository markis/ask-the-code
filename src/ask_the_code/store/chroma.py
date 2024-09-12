import contextlib
from collections.abc import Collection, Iterable
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import UUID

from chromadb import PersistentClient
from chromadb.api import ClientAPI
from FlagEmbedding import FlagReranker

from ask_the_code.chunkers import markdown_chunker
from ask_the_code.config import Config
from ask_the_code.types import DocSource
from ask_the_code.utils import cache_home, data_home, get_repo_files, get_working_path

if TYPE_CHECKING:
    from collections.abc import Iterable


CHROMA_NAMESPACE: Final = UUID("c0e5b3b8-0b1d-4d4c-8b1f-8a3f4c6b3b4d")
CHROMA_DIR: Final = "chroma"
MAX_BATCH_SIZE: Final = 128


class ChromaStore:
    config: Final[Config]

    @property
    def collection_name(self) -> str:
        return f"docs-{self.working_path.name}"

    @cached_property
    def client(self) -> ClientAPI:
        return PersistentClient(str(data_home() / CHROMA_DIR))

    @cached_property
    def reranker(self) -> FlagReranker:
        return FlagReranker(
            self.config.reranker_model,
            use_fp16=True,
            cache_dir=str(cache_home() / CHROMA_DIR),
        )

    @cached_property
    def working_path(self) -> Path:
        return get_working_path(self.config.repo)

    def __init__(self, config: Config) -> None:
        """Initialize the ChromaStore."""
        self.config = config

    def create(self) -> Iterable[None]:
        """Create the knowledge store."""
        self.reset_index()
        files = get_repo_files(self.working_path, self.config.glob)
        for file in files:
            yield self.add_document(file)

    def add_document(self, path: Path) -> None:
        """Add a document to the knowledge store."""
        client = self.client
        collection = client.get_collection(self.collection_name)
        relative_path = path.relative_to(self.working_path)

        chunks = markdown_chunker(path, relative_path)

        iterator = iter({"source": source, "text": chunk} for source, chunk in chunks)
        while batch := list(islice(iterator, MAX_BATCH_SIZE)):
            ids = [doc["source"] for doc in batch]
            docs = [doc["text"] for doc in batch]
            collection.upsert(ids=ids, documents=docs)

    def reset_index(self) -> None:
        """Reset the knowledge store."""
        client = self.client
        with contextlib.suppress(ValueError):
            client.delete_collection(self.collection_name)
        _ = client.create_collection(self.collection_name)

    def search(self, query: str, min_score: float = 0.0) -> Collection[DocSource]:
        """Query the knowledge store for content"""
        client = self.client
        collection = client.get_collection(self.collection_name)
        results = collection.query(query_texts=query, n_results=10)
        if not results or not (documents := results.get("documents")):
            return []

        ids = [source for ids in results["ids"] for source in ids]
        texts = [text for texts in documents for text in texts]
        scores = self.reranker.compute_score([[query, text] for text in texts])

        return sorted(
            [
                DocSource(source=source, text=text, score=score)
                for source, text, score in zip(ids, texts, scores)
                if score > min_score
            ],
            key=lambda doc: doc["score"],
            reverse=True,
        )
