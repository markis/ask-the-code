from typing_extensions import TypeGuard, TypedDict


class DocSource(TypedDict):
    source: str
    text: str


def is_doc_source(obj: object) -> TypeGuard[DocSource]:
    return isinstance(obj, dict) and "source" in obj and "text" in obj


class MarqoKnowledgeStoreResponseHit(DocSource):
    _score: float


class MarqoKnowledgeStoreResponse(TypedDict):
    hits: list[MarqoKnowledgeStoreResponseHit]


def is_marqo_knowledge_store_response(
    obj: object,
) -> TypeGuard[MarqoKnowledgeStoreResponse]:
    return isinstance(obj, dict) and "hits" in obj


def is_int(obj: object) -> TypeGuard[int]:
    return isinstance(obj, int)
