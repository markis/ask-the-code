from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp
from unittest.mock import Mock

import pytest
from git import Git

from ask_the_code.config import Config
from ask_the_code.error import CollectionNotFoundError
from ask_the_code.store.chroma import ChromaStore


@pytest.fixture(scope="session")  # type: ignore[misc]
def test_repo() -> Iterable[Path]:
    with TemporaryDirectory() as temp_dir:
        repo = Path(temp_dir) / "test_repo"
        repo.mkdir()
        (repo / "README.md").touch()
        git = Git(repo)
        git.init()
        git.add(".")
        git.commit("--no-gpg-sign", "-m", "Initial commit")
        yield repo


@pytest.fixture  # type: ignore[misc]
def test_file() -> Iterable[str]:
    path = None
    try:
        _, path = mkstemp()
        yield path
    finally:
        if path is not None:
            Path(path).unlink()


def test_chroma_store_init() -> None:
    config = Mock(spec=Config, repo=Path.cwd())
    store = ChromaStore(config)
    assert store.config == config


def test_collection_name(test_repo: Path) -> None:
    config = Mock(spec=Config, repo=test_repo)
    store = ChromaStore(config)
    assert store.collection_name == "docs-test_repo"


def test_get_collection_raises_error(test_repo: Path, test_file: str) -> None:
    config = Mock(spec=Config, repo=test_repo)
    store = ChromaStore(config)
    with pytest.raises(CollectionNotFoundError):
        store.add_document(Path(test_file))


@pytest.mark.skip
def test_reset_index():
    config = Mock(spec=Config, repo=test_repo)
    store = ChromaStore(config)
    store.client = Mock()
    store.reset_index()
    store.client.delete_collection.assert_called_once_with(store.collection_name)
    store.client.create_collection.assert_called_once_with(store.collection_name)


def test_search(test_repo: Path) -> None:
    config = Mock(spec=Config, repo=test_repo)
    store = ChromaStore(config)
    store.reranker = Mock()
    store.reranker.compute_score.return_value = [0.5, 0.6]
    store.client = Mock()
    store.client.get_collection().query = Mock(
        return_value={"documents": [["doc1", "doc2"]], "ids": [["id1", "id2"]]}
    )
    result = store.search("test query", min_score=0.4)
    assert result == [
        {"source": "id2", "text": "doc2", "score": 0.6},
        {"source": "id1", "text": "doc1", "score": 0.5},
    ]
