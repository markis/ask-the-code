from __future__ import annotations

import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

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


def cache_home() -> Path:
    """Get the cache home directory (~/.cache/ask)."""
    return Path(_platform_dir().user_cache_dir)


def clean_data_home() -> None:
    """Clean the data home directory."""
    import shutil

    shutil.rmtree(data_home(), ignore_errors=True)
    shutil.rmtree(cache_home(), ignore_errors=True)


def copy_class_vars_to_instance(cls: type[T], instance: T) -> None:
    """Copy class variables to an instance."""
    for attr in cls.__dict__:
        value: object = getattr(cls, attr)
        if not attr.startswith("__") and not callable(value):
            setattr(instance, attr, value)


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


def download_hf_model(repo_id: str) -> str:
    from huggingface_hub import hf_hub_download

    # Download the model from the Hugging Face Hub
    return hf_hub_download(repo_id=repo_id, filename="model.zip")
