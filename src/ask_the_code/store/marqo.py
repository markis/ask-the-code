from collections.abc import Collection
import logging
from functools import cache
from itertools import islice
from pathlib import Path

from git import Repo
from marqo.client import Client
from marqo.errors import IndexAlreadyExistsError, MarqoWebError
from marqo.index import Index
from typing_extensions import Final

from ask_the_code.chunkers import markdown_chunker
from ask_the_code.config import Config
from ask_the_code.types import (
    DocSource,
    is_marqo_knowledge_store_response,
)
from ask_the_code.utils import get_repo_files

MAX_BATCH_SIZE: Final = 128
RELAVANCE_SCORE: Final = 0.6
logger: Final = logging.getLogger(__name__)


@cache
def _get_client(marqo_url: str) -> Client:
    """Return a Marqo client."""
    return Client(marqo_url)


def _get_index(config: Config) -> Index:
    """Return the knowledge store index."""
    # TODO: make the store configurable
    client = _get_client(config.marqo_url)
    return client.index(config.index_name)


class MarqoStore:
    config: Final[Config]
    working_path: Final[Path]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.working_path = Path(
            Repo(config.repo, search_parent_directories=True).working_dir
        )

    def create(self) -> None:
        """Create the knowledge store."""
        self.reset_index()
        files = get_repo_files(self.working_path, self.config.glob)
        for file in files:
            self.add_document(file)

    def add_document(self, path: Path) -> None:
        """Add a document to the knowledge store."""
        index = _get_index(self.config)
        source = str(path.relative_to(self.config.repo))
        chunks = markdown_chunker(path)

        iterator = iter({"source": source, "text": chunk} for chunk in chunks)
        while batch := list(islice(iterator, MAX_BATCH_SIZE)):
            _ = index.add_documents(batch, tensor_fields=["text"])

    def search(self, query: str) -> Collection[DocSource]:
        """Query the knowledge store for content based on a query."""
        logger.info("QUERY: %s", query)
        index = _get_index(self.config)
        resp = index.search(q=query, show_highlights=False)
        if not is_marqo_knowledge_store_response(resp):
            raise ValueError(f"Unexpected response from Marqo: {resp}")

        return [res for res in resp["hits"] if res["_score"] > RELAVANCE_SCORE]

    def reset_index(self) -> None:
        """Reset the index by deleting it if it exists and creating a new one.

        This method will attempt to delete the existing index. If the index is not found,
        it will print a message and create a new one. If an error occurs during deletion
        or creation, it will print the error message.
        """
        index_name = self.config.index_name
        try:
            index = _get_index(self.config)
            _ = index.delete()
        except MarqoWebError:
            logger.exception("Index '%s' not found. Creating a new one.", index_name)

        try:
            client = _get_client(self.config.marqo_url)
            index = client.create_index(
                index_name=index_name, model=self.config.marqo_model
            )
        except IndexAlreadyExistsError:
            logger.exception(
                "Index '%s' already exists. Updating settings.", index_name
            )
