import argparse
import io
from typing_extensions import override

HELP = """
[blue]Usage: ask [OPTIONS] [question][/blue]

  A CLI for asking questions about documentation in a repository.

[blue]Options:[/blue]
  -h, --help          Show this message and exit.

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


class ArgumentParser(argparse.ArgumentParser):
    @override
    def print_help(self, file: io.StringIO | None = None) -> None:
        print(HELP)
        print()
