"""
conftest.py — Fixtures partagées entre tous les fichiers de test.

pytest charge ce fichier automatiquement avant tout test.
Les fixtures définies ici sont disponibles dans TOUS les tests
sans avoir besoin de les importer.

Concept de fixture :
    Une fixture est une fonction décorée avec @pytest.fixture.
    Elle prépare des données ou des objets réutilisables.
    pytest injecte automatiquement la fixture dans les fonctions
    de test qui ont le même nom en paramètre.

    Exemple :
        @pytest.fixture
        def sample_text():
            return "Hello World"

        def test_something(sample_text):  # ← pytest injecte la fixture
            assert len(sample_text) > 0
"""

import pytest
from src.ingestion.models import Document
from src.processing.chunker import Chunk


# ---------------------------------------------------------------------------
# Fixtures : Données de base
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_text() -> str:
    """Texte brut typique d'un PDF d'assurance (avec bruit)."""
    return (
        "Die Versicherung bietet Schutz bei Unfall.\n"
        "Im Falle eines Schadens wenden Sie sich\n"
        "bitte an unsere Hotline.\n\n"
        "Die Kosten wer-\nden übernommen."
    )


@pytest.fixture
def clean_sample_text() -> str:
    """Version nettoyée du sample_text."""
    return (
        "Die Versicherung bietet Schutz bei Unfall. "
        "Im Falle eines Schadens wenden Sie sich "
        "bitte an unsere Hotline. "
        "Die Kosten werden übernommen."
    )


@pytest.fixture
def sample_document(clean_sample_text) -> Document:
    """
    Document de base pour les tests.

    Note : on utilise le texte nettoyé parce que Document.__post_init__
    valide que le contenu n'est pas vide.
    """
    return Document(
        content=clean_sample_text,
        source="test_doc.pdf",
        file_type="pdf",
        num_pages=1,
    )


@pytest.fixture
def long_document() -> Document:
    """Document assez long pour produire plusieurs chunks."""
    # 600 caractères → produira au moins 2 chunks avec chunk_size=512
    content = "Versicherungsschutz " * 30  # 600 chars
    return Document(
        content=content.strip(),
        source="long_doc.pdf",
        file_type="pdf",
        num_pages=3,
    )


@pytest.fixture
def sample_chunk(sample_document) -> Chunk:
    """Chunk de base pour les tests de vector store."""
    return Chunk(
        text="Die Versicherung bietet Schutz bei Unfall.",
        source="test_doc.pdf",
        chunk_index=0,
        start_char=0,
        end_char=42,
    )
