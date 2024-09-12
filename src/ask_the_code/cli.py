from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import override

import click
from fast_depends import Depends, inject
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress

from ask_the_code.__about__ import __version__
from ask_the_code.config import Config
from ask_the_code.dependency import get_config, get_console, get_llm, get_store
from ask_the_code.llm import LLM
from ask_the_code.store import Store
from ask_the_code.utils import clean_data_home

REPO_HELP = "The repository path. Defaults to the current directory."
GLOB_HELP = 'The glob pattern to match files in the repository. Defaults to "**/*.md".'


def run() -> None:
    cli()


class DefaultGroup(click.Group):
    @override
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Default the command to 'ask' if the first argument is not a command."""
        arg0 = args[0] if args else ""
        if len(args) > 0 and arg0 not in self.commands and not arg0.startswith("-"):
            args.insert(0, "ask")
        return super().parse_args(ctx, args)


@click.group("ask", cls=DefaultGroup)
@click.version_option(__version__, prog_name="ask")
def cli() -> None:
    """A CLI for asking questions about documentation in a repository."""


@cli.command()
@click.argument("question")
@click.option("-r", "--repo", type=Path, default=Path.cwd(), help=REPO_HELP)
@inject
def ask(
    question: str,
    repo: Path,
    config: Config = Depends(get_config),
    console: Console = Depends(get_console),
    store: Callable[[], Store] = Depends(get_store),
    llm: Callable[[], LLM] = Depends(get_llm),
) -> None:
    """Ask a question about the documentation."""
    del config, repo  # Unused
    sources = store().search(question)
    response_stream = llm().answer(sources, question)

    buffer = ""
    with Live(console=console) as live:
        for resp in response_stream:
            buffer += resp
            live.update(Markdown(buffer))


@cli.command()
@inject
def clean() -> None:
    """Clean up the data directory."""

    return clean_data_home()


@cli.command()
@click.option("-g", "--glob", default="**/*.md", help=GLOB_HELP)
@click.option("-r", "--repo", type=Path, default=Path.cwd(), help=REPO_HELP)
@inject
def create(
    repo: Path,
    glob: str,
    store: Callable[[], Store] = Depends(get_store),
    console: Console = Depends(get_console),
) -> None:
    """Create and index the knowledge store."""
    del glob, repo  # Unused
    with Progress(console=console, transient=True) as progress:
        for _ in progress.track(store().create(), description="Indexing"):
            pass
    console.print("[green]Indexing complete![/green]")


if __name__ == "__main__":
    run()
