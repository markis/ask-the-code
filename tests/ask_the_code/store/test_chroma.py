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
def mock_config() -> Iterable[Mock]:
    with TemporaryDirectory() as temp_dir:
        repo = Path(temp_dir) / "test_repo"
        repo.mkdir()
        (repo / "README.md").touch()
        git = Git(repo)
        git.init()
        git.add(".")
        git.commit("--no-gpg-sign", '--author="Test <test@test.com>"', "-m", "Initial commit")
        yield Mock(spec=Config, repo=repo)


@pytest.fixture  # type: ignore[misc]
def test_file() -> Iterable[str]:
    path = None
    try:
        _, path = mkstemp()
        yield path
    finally:
        if path is not None:
            Path(path).unlink()


def test_chroma_store_init(mock_config: Mock) -> None:
    # Act
    store = ChromaStore(mock_config)
    # Assert
    assert store.config == mock_config


def test_collection_name(mock_config: Mock) -> None:
    # Act
    store = ChromaStore(mock_config)
    # Assert
    assert store.collection_name == "docs-test_repo"


def test_get_collection_raises_error(mock_config: Mock, test_file: str) -> None:
    store = ChromaStore(mock_config)
    # Act/Assert
    with pytest.raises(CollectionNotFoundError):
        store.add_document(Path(test_file))


def test_reset_index(mock_config: Mock) -> None:
    # Arrange
    store = ChromaStore(mock_config)
    store.client = Mock()
    # Act
    store.reset_index()
    # Assert
    store.client.delete_collection.assert_called_once_with(store.collection_name)
    store.client.create_collection.assert_called_once_with(store.collection_name)


def test_search(mock_config: Mock) -> None:
    # Arrange
    store = ChromaStore(mock_config)
    store.reranker = Mock()
    store.reranker.compute_score.return_value = [0.5, 0.6]
    store.client = Mock()
    store.client.get_collection().query = Mock(
        return_value={"documents": [["doc1", "doc2"]], "ids": [["id1", "id2"]]}
    )
    # Act
    result = store.search("test query", min_score=0.4)
    # Assert
    assert result == [
        {"source": "id2", "text": "doc2", "score": 0.6},
        {"source": "id1", "text": "doc1", "score": 0.5},
    ]
