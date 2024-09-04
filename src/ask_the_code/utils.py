from __future__ import annotations

import sys
import shutil
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from git.repo import Repo
from typing_extensions import TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Iterable

    from platformdirs import PlatformDirsABC


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


def clean_data_home() -> None:
    """Clean the data home directory."""
    shutil.rmtree(data_home(), ignore_errors=True)


def copy_class_vars_to_instance(cls: type[T], instance: T) -> None:
    """Copy class variables to an instance."""
    for attr in cls.__dict__:
        value: object = getattr(cls, attr)
        if not attr.startswith("__") and not callable(value):
            setattr(instance, attr, value)


def get_working_path(cwd: Path) -> Path:
    """Get the working path."""
    with Repo(cwd, search_parent_directories=True) as repo:
        return Path(repo.working_dir)


def get_repo_files(path: Path, glob: str) -> Iterable[Path]:
    """Get the documents files."""
    with Repo(path, search_parent_directories=True) as repo:
        working_path = Path(repo.working_dir)
        files = [file for file in working_path.glob(glob) if not repo.ignored(file)]
    yield from files
