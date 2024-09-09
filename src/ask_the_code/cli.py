import logging
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import override

import click
from click.globals import get_current_context
from fast_depends import Depends, inject
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress
from rich.table import Table

from ask_the_code.__about__ import __version__
from ask_the_code.config import Config
from ask_the_code.store import Store, get_store

logging.basicConfig(level=logging.ERROR)

REPO_HELP = "The repository path to index. Defaults to the current directory."
GLOB_HELP = 'The glob pattern to match files in the repository. Defaults to "**/*.md".'


def console() -> Console:
    return Console()


def config() -> Config:
    ctx: click.Context = get_current_context()
    return Config.create(**ctx.params)


def store(config: Config = Depends(config)) -> Callable[[], Store]:
    """Return a store getter. This delays the store creation until it's needed."""

    @cache
    def store_getter() -> Store:
        return get_store(config)

    return store_getter


class DefaultGroup(click.Group):
    @override
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if len(args) > 0 and args[0] not in self.commands:
            args.insert(0, "ask")
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup)
@click.version_option(__version__)
def cli() -> None:
    """A CLI for asking questions about documentation in a repository."""


@cli.command()
@click.option("-r", "--repo", type=Path, default=Path.cwd(), help=REPO_HELP)
@click.option("-g", "--glob", default="**/*.md", help=GLOB_HELP)
def create(
    repo: Path,
    glob: str,
    store: Callable[[], Store] = Depends(store),
    console: Console = Depends(console),
) -> None:
    with Progress(console=console, transient=True) as progress:
        for _ in progress.track(store().create(), description="Indexing"):
            pass
    console.print("[green]Indexing complete![/green]")


@cli.command()
@click.argument("question")
@click.option("-r", "--repo", type=Path, default=Path.cwd(), help=REPO_HELP)
@inject
def ask(
    question: str,
    repo: Path,
    config: Config = Depends(config),
    store: Callable[[], Store] = Depends(store),
    console: Console = Depends(console),
) -> None:
    from ask_the_code import llm

    sources = store().search(question)
    response_stream = llm.answer(config, sources, question)

    buffer = ""
    with Live(console=console) as live:
        for resp in response_stream:
            buffer += resp
            live.update(Markdown(buffer))


HELP = f"""
[bold blue]Usage: ask [OPTIONS] <QUESTION>[/bold blue]
[bold blue]Version:[/bold blue] {__version__}

  A CLI for asking questions about documentation in a repository.

[blue]Options:[/blue]
  -h, --help          Show this message and exit.
  -r, --repo          The repository path to index. Defaults to the current directory.
  -g, --glob          The glob pattern to match files in the repository. Defaults to "**/*.md".

[blue]Description:[/blue]
  This CLI indexes Markdown files in a repository and allows you to ask
  specific questions about the documentation. Currently, only Markdown files
  are indexed, so questions can only be asked about content within those files.

[blue]Question Examples:[/blue]
  * Specific sentence or paragraph: "what's this sentence about?" (or point to a specific line)
  * Topic or section: "what's this project about?"

[blue]Examples:[/blue]
  $ ask "what's this project about?"
  $ ask "who wrote this section?"
""".strip()


def run_() -> None:
    console = Console()
    cli_(console)


def cli_(console: Console) -> None:
    try:
        config = Config.create()

        if config.help or config.question is None:
            return console.print(HELP)

        store = get_store(config)
        if config.question == "create":
            return create_(store, console)

        if config.question == "clean":
            from ask_the_code import utils

            return utils.clean_data_home()

        if config.question == "download":
            from ask_the_code import utils

            path = utils.download_hf_model("BAAI/bge-reranker-large")
            console.print(f"Model downloaded to [bold]{path}[/bold]")
            return

        if config.search:
            return search_(store, console, config.question)

        ask_(config, store, console, config.question)
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled![/red]")
    except Exception:
        console.print_exception()


def create_(store: Store, console: Console) -> None:
    with Progress(console=console, transient=True) as progress:
        for _ in progress.track(store.create(), description="Indexing"):
            pass
    console.print("[green]Indexing complete![/green]")


def search_(store: Store, console: Console, query: str) -> None:
    sources = store.search(query)
    source_table = Table(title="Sources")
    source_table.add_column("Score", style="bold blue")
    source_table.add_column("Source", style="bold green")
    source_table.add_column("Text")
    for source in sources:
        source_table.add_row(
            str(source["score"]), source["source"], Markdown(source["text"])
        )
    console.print(source_table)


def ask_(config: Config, store: Store, console: Console, question: str) -> None:
    from ask_the_code import llm

    sources = store.search(question)
    response_stream = llm.answer(config, sources, question)

    buffer = ""
    with Live(console=console) as live:
        for resp in response_stream:
            buffer += resp
            live.update(Markdown(buffer))


if __name__ == "__main__":
    cli()
