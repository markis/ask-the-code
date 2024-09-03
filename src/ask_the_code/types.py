from __future__ import annotations

from typing_extensions import TypedDict, TypeGuard


class DocSource(TypedDict):
    source: str
    text: str


def is_doc_source(obj: object) -> TypeGuard[DocSource]:
    return isinstance(obj, dict) and "source" in obj and "text" in obj


def is_int(obj: object) -> TypeGuard[int]:
    return isinstance(obj, int)
