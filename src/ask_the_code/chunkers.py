from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from mistletoe import Document
from mistletoe.block_token import Heading
from mistletoe.markdown_renderer import MarkdownRenderer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

Source = str
Text = str


def _convert_to_github_headline_link(text: str) -> str:
    return text.strip("#").strip().replace(" ", "-").lower()


def markdown_chunker(path: Path, relative_path: Path) -> Iterable[tuple[Source, Text]]:
    """Split a markdown document into sections based on headings."""
    text = path.read_text().strip()

    doc = Document(text.splitlines())
    if not doc.children:
        yield (str(path), text)

    sections: dict[str, list[str]] = defaultdict(list)
    current_section: str = ""
    with MarkdownRenderer(normalize_whitespace=True) as renderer:
        for token in doc.children:
            if isinstance(token, Heading):
                current_section = renderer.render(token)
                current_section = _convert_to_github_headline_link(current_section)
            else:
                sections[current_section].append(renderer.render(token))

    for section, text_chunk in sections.items():
        yield f"{relative_path}#{section}", "\n".join(text_chunk)
