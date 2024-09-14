from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Final

from ollama import Client

from ask_the_code.config import Config
from ask_the_code.types import DocSource


class Ollama:
    _model: Final[str]

    def __init__(self, config: Config) -> None:
        self._model = config.ollama_model
        self._client = Client(config.ollama_url)

    def answer(self, context: Collection[DocSource], question: str) -> Iterable[str]:
        """Generate an answer based on user input using a LLM and Store."""
        sources = "\n".join((source["source"] + ":\n " + source["text"]) for source in context)

        prompt = f"""
        Given the following extracted parts of a document ("SOURCES") and a question ("QUESTION").
        Create a final answer one paragraph long.
        Answer the question and cite the sources in the answer.
        Don't try to make up an answer and use the text in the SOURCES only for the answer.
        If you don't know the answer, just say that you don't know.

        QUESTION: {question}
        =========
        SOURCES:
        {sources}
        """

        yield from self.generate(prompt)

    def generate(self, prompt: str) -> Iterable[str]:
        for resp in self._client.generate(self._model, prompt=prompt, stream=True):
            if "response" in resp and isinstance(resp["response"], str):
                yield resp["response"]
