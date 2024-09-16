from __future__ import annotations

import contextlib
from collections.abc import Collection, Iterable
from functools import cached_property
from pathlib import Path
from typing import Final, cast
from uuid import UUID

import polars as pl
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.types import Collection as ChromaCollection
from FlagEmbedding import FlagReranker

from ask_the_code.chunkers import markdown_chunker
from ask_the_code.config import Config
from ask_the_code.error import CollectionNotFoundError
from ask_the_code.types import DocSource
from ask_the_code.utils import cache_home, data_home, get_repo_files, get_working_path

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

    def _get_collection(self, name: str) -> ChromaCollection:
        try:
            client = self.client
            return cast(ChromaCollection, client.get_collection(name))
        except ValueError as e:
            raise CollectionNotFoundError(self.collection_name) from e

    def _compute_score(self, query: str, texts: Collection[str]) -> Collection[float]:
        scores: Collection[float] = self.reranker.compute_score([(query, text) for text in texts])
        return scores

    def create(self) -> Iterable[str]:
        """Create the knowledge store."""
        self.reset_index()
        files = get_repo_files(self.working_path, self.config.glob)
        for file in files:
            yield str(file)
            self.add_document(file)

    def add_document(self, path: Path) -> None:
        """Add a document to the knowledge store."""
        collection = self._get_collection(self.collection_name)
        relative_path = path.relative_to(self.working_path)

        md_chunks = markdown_chunker(path, relative_path)
        df = pl.from_records(list(md_chunks), schema=["id", "doc"], orient="row")
        for i in range(0, len(df), MAX_BATCH_SIZE):
            chunk = df.slice(i, MAX_BATCH_SIZE)
            collection.upsert(ids=chunk["id"].to_list(), documents=chunk["doc"].to_list())  # type: ignore[attr-defined]

    def reset_index(self) -> None:
        """Reset the knowledge store."""
        client = self.client
        with contextlib.suppress(ValueError):
            client.delete_collection(self.collection_name)
        _ = client.create_collection(self.collection_name)

    def search(self, query: str, min_score: float = 0.0) -> Collection[DocSource]:
        """Query the knowledge store for content"""
        collection = self._get_collection(self.collection_name)
        results = collection.query(query_texts=query, n_results=10)  # type: ignore[attr-defined]
        if not results or not (documents := results.get("documents")):
            return []

        df = pl.DataFrame({"source": results["ids"], "text": documents}).explode("source", "text")
        scores = self._compute_score(query, df["text"].to_list())
        df = (
            df.with_columns(pl.Series("score", scores))
            .filter(pl.col("score") > min_score)
            .sort("score", descending=True)
        )
        return cast(list[DocSource], df.to_dicts())
