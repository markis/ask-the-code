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

    else:
        heading_path: list[str] = []
        section_content: list[str] = []
        relative_str = str(relative_path)
        seen_ids: set[str] = set()
        counter = 0

        with MarkdownRenderer(normalize_whitespace=True) as renderer:
            for token in children:
                if isinstance(token, Heading):
                    if section_content:
                        base_section_id = f"{relative_str}#{'-'.join(heading_path)}"
                        section_id = base_section_id
                        while section_id in seen_ids:
                            counter += 1
                            section_id = f"{base_section_id}-{counter}"
                        seen_ids.add(section_id)
                        yield section_id, "\n".join(section_content)
                        section_content = []

                    heading_text = (
                        renderer.render(token).strip("#").strip().replace(" ", "-").lower()
                    )
                    heading_path = heading_path[: token.level - 1] + [heading_text]
                else:
                    section_content.append(renderer.render(token))

            if section_content:
                base_section_id = f"{relative_str}#{'-'.join(heading_path)}"
                section_id = base_section_id
                while section_id in seen_ids:
                    counter += 1
                    section_id = f"{base_section_id}-{counter}"
                seen_ids.add(section_id)
                yield section_id, "\n".join(section_content)
