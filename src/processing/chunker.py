"""
Découpage des documents en chunks (fragments de texte).

Le chunking est une étape critique du pipeline RAG :
- Trop grand → le modèle d'embeddings est moins précis
- Trop petit → on perd le contexte autour de l'information
"""

from dataclasses import dataclass, field
from src.ingestion.models import Document
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """
    Représente un fragment de texte extrait d'un Document.

    Chaque Chunk garde une référence à son document source
    et à sa position dans ce document.
    """

    # Le texte de ce fragment
    text: str

    # Nom du fichier source (hérité du Document parent)
    source: str

    # Position de ce chunk dans le document (0 = premier chunk)
    chunk_index: int

    # Caractère de début dans le document original
    start_char: int

    # Caractère de fin dans le document original
    end_char: int

    # Métadonnées héritées du Document parent
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(source='{self.source}', "
            f"index={self.chunk_index}, "
            f"chars={self.start_char}-{self.end_char}, "
            f"preview='{preview}...')"
        )


def chunk_document(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Découpe un Document en une liste de Chunks.

    Args:
        document    : Le Document nettoyé à découper
        chunk_size  : Taille maximale de chaque chunk en caractères
        chunk_overlap: Nombre de caractères partagés entre chunks consécutifs

    Returns:
        Liste de Chunks ordonnés
    """
    text = document.content
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # Définir la fin de ce chunk
        end = start + chunk_size

        # Si on dépasse la fin du texte, on prend jusqu'à la fin
        if end >= len(text):
            chunk_text = text[start:]
            chunks.append(Chunk(
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=len(text),
                metadata=document.metadata,
            ))
            break

        # Chercher le dernier espace avant "end" pour ne pas couper un mot
        # On cherche en arrière depuis "end" sur max 100 caractères
        split_pos = text.rfind(" ", start, end)

        # Si on ne trouve pas d'espace, on coupe brutalement à "end"
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunk_text = text[start:split_pos].strip()

        if chunk_text:  # ignorer les chunks vides
            chunks.append(Chunk(
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=split_pos,
                metadata=document.metadata,
            ))
            chunk_index += 1

        # Le prochain chunk commence "chunk_overlap" caractères avant la fin
        # C'est le chevauchement : on recule de "chunk_overlap" pour la continuité
        start = split_pos - chunk_overlap

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Découpe une liste de Documents en Chunks.

    Returns:
        Liste de tous les Chunks de tous les documents
    """
    all_chunks = []

    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
        logger.info("%s → %d chunks", doc.source, len(doc_chunks))

    logger.info("Total : %d chunks pour %d document(s)", len(all_chunks), len(documents))
    return all_chunks