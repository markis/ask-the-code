from __future__ import annotations

import sys
from collections.abc import Iterable
from functools import cache
from itertools import islice
from pathlib import Path

from platformdirs import PlatformDirsABC
from typing_extensions import TypeVar

T = TypeVar("T")


@cache
def _platform_dir() -> PlatformDirsABC:
    if sys.platform == "win32":
        from platformdirs.windows import Windows as Result
    else:
        from platformdirs.unix import Unix as Result

    return Result(
        appname="ask",
        appauthor="markis",
        ensure_exists=True,
    )


def config_home() -> Path:
    """Get the configuration home directory (~/.config/ask)."""
    return Path(_platform_dir().user_config_dir)


def data_home() -> Path:
    """Get the data home directory (~/.local/share/ask)."""
    return Path(_platform_dir().user_data_dir)


def cache_home() -> Path:
    """Get the cache home directory (~/.cache/ask)."""
    return Path(_platform_dir().user_cache_dir)


def clean_data_home() -> None:
    """Clean the data home directory."""
    import shutil

    shutil.rmtree(data_home(), ignore_errors=True)
    shutil.rmtree(cache_home(), ignore_errors=True)


def get_working_path(cwd: Path) -> Path:
    """Get the working path."""
    from git.repo import Repo

    with Repo(cwd, search_parent_directories=True) as repo:
        return Path(repo.working_dir)


def get_repo_files(path: Path, glob: str) -> Iterable[Path]:
    """Get the documents files."""
    from git.repo import Repo

    with Repo(path, search_parent_directories=True) as repo:
        working_path = Path(repo.working_dir)
        files = [file for file in working_path.glob(glob) if not repo.ignored(file)]
    yield from files


def chunks(iterable: Iterable[T], chunk_size: int) -> Iterable[list[T]]:
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk
