import logging
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.markdown import Markdown
from rich.table import Table

import ask_the_code.llm as llm
from ask_the_code.config import Config
from ask_the_code.store import get_store, Store

logging.basicConfig(level=logging.WARNING)

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


def main(config: Config, store: Store, console: Console) -> None:
    try:
        if config.help or config.question is None:
            console.print(HELP)
            return

        if config.question == "create":
            store.create()
            return

        if config.search:
            search(config, store, console, config.question)
            return

        ask(config, store, console, config.question)
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled![/red]")


def search(config: Config, store: Store, console: Console, query: str) -> None:
    sources = store.search(query)
    source_table = Table(title="Sources")
    source_table.add_column("Source", style="bold green")
    source_table.add_column("Text")
    for source in sources:
        source_table.add_row(source["source"], Markdown(source["text"]))
    console.print(source_table)


def ask(config: Config, store: Store, console: Console, question: str) -> None:
    buffer = ""
    with Progress(console=console) as progress:
        progress.start()
        sources = store.search(question)
        response_stream = llm.answer(config, sources, question)
        progress.stop()

    with Live(console=console) as live:
        for resp in response_stream:
            buffer += resp
            live.update(Markdown(buffer))


if __name__ == "__main__":
    config = Config.create()
    store = get_store(config)
    main(config, store, Console())
