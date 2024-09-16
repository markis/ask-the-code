from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

from ask_the_code.utils import (
    cache_home,
    chunks,
    clean_data_home,
    config_home,
    data_home,
    get_repo_files,
    get_working_path,
)


class TestConfigHome:
    def test_config_home(self) -> None:
        dir_ = config_home()
        assert isinstance(dir_, Path)
        assert dir_.name == "ask"


class TestDataHome:
    def test_data_home(self) -> None:
        dir_ = data_home()
        assert isinstance(dir_, Path)
        assert dir_.name == "ask"


class TestCacheHome:
    def test_cache_home(self) -> None:
        dir_ = cache_home()
        assert isinstance(dir_, Path)
        assert dir_.name == "ask"


class TestCleanDataHome:
    @patch("shutil.rmtree")
    def test_clean_data_home(self, mock_rmtree: Mock) -> None:
        clean_data_home()
        mock_rmtree.assert_has_calls(
            (
                call(data_home(), ignore_errors=True),
                call(cache_home(), ignore_errors=True),
            )
        )


class TestGetWorkingPath:
    @patch("git.repo.Repo")
    def test_get_working_path(self, mock_repo: Mock) -> None:
        mock_repo().__enter__().working_dir = "/path/to/repo"
        path = get_working_path(Path("/path/to/repo/nested"))
        assert path == Path("/path/to/repo")


class TestGetRepoFiles:
    @patch("git.repo.Repo")
    def test_get_repo_files(self, mock_repo: Mock) -> None:
        with TemporaryDirectory() as tmpdirname:
            working_dir = Path(tmpdirname)
            Path(working_dir / "file.txt").touch()
            mock_repo_obj = mock_repo().__enter__()
            mock_repo_obj.working_dir = working_dir
            mock_repo_obj.ignored.return_value = False

            files = list(get_repo_files(working_dir, "*.txt"))
            assert files == [Path(working_dir / "file.txt")]


class TestChunks:
    def test_chunks(self) -> None:
        iterable = [1, 2, 3, 4, 5]
        chunk_size = 2
        chunks_list = list(chunks(iterable, chunk_size))
        assert chunks_list == [[1, 2], [3, 4], [5]]
