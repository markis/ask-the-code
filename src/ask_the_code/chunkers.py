from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

Source = str
Text = str


def markdown_chunker(path: Path, relative_path: Path) -> Iterable[tuple[Source, Text]]:
    """Split a markdown document into sections based on headings."""

    from mistletoe import Document
    from mistletoe.block_token import Heading
    from mistletoe.markdown_renderer import MarkdownRenderer

    text = path.read_text()
    relative_str = str(relative_path)

    doc = Document(text.splitlines())
    if not (children := doc.children):
        yield (str(path), text)

    section_hierarchy: list[str] = []
    current_text: list[str] = []
    with MarkdownRenderer(normalize_whitespace=True) as renderer:
        for token in children:
            if isinstance(token, Heading):
                if current_text:
                    yield (f"{relative_str}#{'-'.join(section_hierarchy)}", "\n".join(current_text))
                    current_text = []

                headline = renderer.render(token).strip("#").strip().replace(" ", "-").lower()
                section_hierarchy = section_hierarchy[: token.level - 1] + [headline]
            else:
                current_text.append(renderer.render(token))

    yield f"{relative_str}#{'-'.join(section_hierarchy)}", "\n".join(current_text)
