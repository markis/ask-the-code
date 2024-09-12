class AskError(Exception):
    pass


class CollectionNotFoundError(AskError):
    collection_name: str

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        super().__init__(f"Collection {collection_name} not found")
