from __future__ import annotations

import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from git.cmd import Git
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


def copy_class_vars_to_instance(cls: type[T], instance: T) -> None:
    """Copy class variables to an instance."""
    for attr in cls.__dict__:
        value: object = getattr(cls, attr)
        if not attr.startswith("__") and not callable(value):
            setattr(instance, attr, value)


def get_working_path(cwd: Path) -> Path:
    """Get the working path."""
    return Path(Repo(cwd, search_parent_directories=True).working_dir)


def get_repo_files(path: Path, glob: str) -> Iterable[Path]:
    """Get the documents files."""
    working_path = get_working_path(path)
    output = str(Git(working_path).ls_files(glob))
    return (get_working_path(path) / file for file in output.splitlines())
