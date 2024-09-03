from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import mistletoe
import mistletoe.block_token
from mistletoe.markdown_renderer import MarkdownRenderer

from ask_the_code.types import is_int

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

Source = str
Text = str


def markdown_chunker(path: Path) -> Iterable[tuple[Source, Text]]:
    """Split a markdown document into sections based on headings."""
    text = path.read_text().strip()

    doc = mistletoe.Document(text.splitlines())
    if not doc.children:
        yield (str(path), text)

    sections: dict[str, list[str]] = defaultdict(list)
    current_section: str = ""
    heirarchy: list[str] = []
    with MarkdownRenderer(normalize_whitespace=True) as renderer:
        for token in doc.children:
            if isinstance(token, mistletoe.block_token.Heading):
                rendered = renderer.render(token).strip()
                token_level = token.level if is_int(token.level) else 1
                heirarchy = heirarchy[: token_level - 1]
                heirarchy.append(rendered)
                current_section = " > ".join(heirarchy)
                sections[current_section].append(current_section)
            else:
                sections[current_section].append(renderer.render(token))

    for section, text_chunk in sections.items():
        yield str(path) + section, "\n".join(text_chunk)
