import logging

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress
from rich.table import Table

from ask_the_code import llm, utils
from ask_the_code.config import Config
from ask_the_code.store import Store, get_store

logging.basicConfig(level=logging.ERROR)

HELP = """
[bold blue]Usage: ask [OPTIONS] <QUESTION>[/bold blue]

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


def run() -> None:
    config = Config.create()
    store = get_store(config)
    console = Console()

    cli(config, store, console)


def cli(config: Config, store: Store, console: Console) -> None:
    try:
        if config.help or config.question is None:
            return console.print(HELP)

        if config.question == "create":
            return create(store, console)

        if config.question == "clean":
            return utils.clean_data_home()

        if config.question == "download":
            path = utils.download_hf_model("BAAI/bge-reranker-large")
            console.print(f"Model downloaded to [bold]{path}[/bold]")
            return

        if config.search:
            return search(store, console, config.question)

        ask(config, store, console, config.question)
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled![/red]")
    except Exception:
        console.print_exception()


def create(store: Store, console: Console) -> None:
    with Progress(console=console, transient=True) as progress:
        for _ in progress.track(store.create(), description="Indexing"):
            pass
    console.print("[green]Indexing complete![/green]")


def search(store: Store, console: Console, query: str) -> None:
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


def ask(config: Config, store: Store, console: Console, question: str) -> None:
    sources = store.search(question)
    response_stream = llm.answer(config, sources, question)

    buffer = ""
    with Live(console=console) as live:
        for resp in response_stream:
            buffer += resp
            live.update(Markdown(buffer))


if __name__ == "__main__":
    run()
