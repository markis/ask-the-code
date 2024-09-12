from __future__ import annotations

from collections.abc import Collection, Iterable

from typing_extensions import Protocol

from ask_the_code.config import Config as AskConfig
from ask_the_code.types import DocSource


class LLM(Protocol):
    def answer(self, context: Collection[DocSource], question: str) -> Iterable[str]: ...


def get_llm(config: AskConfig) -> LLM:
    if config.llm == "ollama":
        from ask_the_code.llm.ollama import Ollama

        return Ollama(config)
    err_msg = f"Unknown LLM: {config.llm}"
    raise ValueError(err_msg)
