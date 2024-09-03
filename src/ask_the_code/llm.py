import logging
from collections.abc import Callable, Iterable

from ask_the_code.config import Config
from ask_the_code.types import DocSource

logger = logging.getLogger(__name__)


def _get_llm_generate(config: Config) -> Callable[[str], Iterable[str]]:
    """Return a function that generates text using the LLM."""
    # TODO: make the llm configurable
    from ollama import Client

    def generate(prompt: str) -> Iterable[str]:
        client = Client(config.ollama_url)
        for resp in client.generate(config.ollama_model, prompt, stream=True):
            if "response" in resp and isinstance(resp["response"], str):
                logger.debug("Generated: %s", resp)
                yield resp["response"]

    return generate


def answer(config: Config, context: Iterable[DocSource], question: str) -> Iterable[str]:
    """Generate an answer based on user input using a LLM and Store."""
    generate = _get_llm_generate(config)

    sources = "\n".join(
        (str(i + 1) + ": " + source["source"] + "\n " + source["text"]) for i, source in enumerate(context)
    )

    prompt = f"""
    Given the following extracted parts of a long document ("SOURCES") and a question ("QUESTION").
    Create a final answer one paragraph long.
    Answer the question and cite the sources in the answer.
    Don't try to make up an answer and use the text in the SOURCES only for the answer.
    If you don't know the answer, just say that you don't know.

    QUESTION: {question}
    =========
    SOURCES:
    {sources}
    """

    yield from generate(prompt)
