"""
Modèles de données pour la couche d'ingestion.
Un Document est l'unité de base qui circule dans tout le pipeline RAG.
"""

from dataclasses import dataclass, field


@dataclass
class Document:
    """Représente un document chargé dans le pipeline."""

    content: str
    source: str
    file_type: str
    num_pages: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content.strip():
            raise ValueError(f"Le document '{self.source}' est vide.")

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def __repr__(self) -> str:
        return (
            f"Document(source='{self.source}', "
            f"type='{self.file_type}', "
            f"pages={self.num_pages}, "
            f"words={self.word_count})"
        )