from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

import click
from click.globals import get_current_context
from fast_depends import Depends
from rich.console import Console

if TYPE_CHECKING:
    from ask_the_code.config import Config
    from ask_the_code.llm import LLM
    from ask_the_code.store import Store


def get_console() -> Console:
    return Console()


def get_config() -> Config:
    from ask_the_code.config import Config

    ctx: click.Context = get_current_context()
    return Config.create(**ctx.params)


def get_store(config: Config = Depends(get_config)) -> Callable[[], Store]:
    """Return a store getter. This delays the store creation until it's needed."""

    @cache
    def store_getter() -> Store:
        from ask_the_code.store import get_store

        return get_store(config)

    return store_getter


def get_llm(config: Config = Depends(get_config)) -> Callable[[], LLM]:
    """Return a LLM getter. This delays the LLM creation until it's needed."""

    @cache
    def llm_getter() -> LLM:
        from ask_the_code.llm import get_llm

        return get_llm(config)

    return llm_getter
