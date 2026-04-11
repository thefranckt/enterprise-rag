"""
Tests unitaires pour src/processing/chunker.py

Tests purs : aucun modèle chargé.
On vérifie la logique de découpage : taille, overlap, indices, métadonnées.
"""

import pytest
from src.processing.chunker import Chunk, chunk_document, chunk_documents


# ---------------------------------------------------------------------------
# Tests du dataclass Chunk
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_chunk_repr():
    """
    __repr__ doit retourner une chaîne non vide utile pour le debug.
    """
    chunk = Chunk(
        text="Test chunk text",
        source="doc.pdf",
        chunk_index=0,
        start_char=0,
        end_char=15,
    )
    r = repr(chunk)
    assert "doc.pdf" in r
    assert "0" in r


# ---------------------------------------------------------------------------
# Tests de chunk_document()
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_chunk_document_short_text_produces_one_chunk(sample_document):
    """
    Un texte plus court que chunk_size doit produire exactement 1 chunk.
    """
    chunks = chunk_document(sample_document, chunk_size=512, chunk_overlap=50)
    assert len(chunks) == 1


@pytest.mark.fast
def test_chunk_document_long_text_produces_multiple_chunks(long_document):
    """
    Un texte de 600 chars avec chunk_size=512 doit produire 2 chunks.
    Le deuxième chunk commence avant la fin du premier (overlap).
    """
    chunks = chunk_document(long_document, chunk_size=512, chunk_overlap=50)
    assert len(chunks) >= 2


@pytest.mark.fast
def test_chunk_document_preserves_source(sample_document):
    """
    Chaque chunk doit hériter du nom de fichier source du document parent.
    """
    chunks = chunk_document(sample_document)
    for chunk in chunks:
        assert chunk.source == sample_document.source


@pytest.mark.fast
def test_chunk_document_indices_are_sequential(long_document):
    """
    Les chunk_index doivent être 0, 1, 2, ... dans l'ordre.
    """
    chunks = chunk_document(long_document, chunk_size=512, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


@pytest.mark.fast
def test_chunk_document_respects_max_size(long_document):
    """
    Aucun chunk (sauf éventuellement le dernier) ne doit dépasser chunk_size.
    """
    chunk_size = 200
    chunks = chunk_document(long_document, chunk_size=chunk_size, chunk_overlap=20)
    # Tous les chunks sauf le dernier doivent être <= chunk_size
    for chunk in chunks[:-1]:
        assert len(chunk.text) <= chunk_size


@pytest.mark.fast
def test_chunk_document_start_end_chars_consistent(sample_document):
    """
    end_char doit être > start_char pour chaque chunk.
    """
    chunks = chunk_document(sample_document)
    for chunk in chunks:
        assert chunk.end_char > chunk.start_char


@pytest.mark.fast
def test_chunk_document_overlap(long_document):
    """
    Avec un overlap, le début du chunk N+1 doit se trouver
    AVANT la fin du chunk N (les deux chunks partagent du texte).
    """
    chunks = chunk_document(long_document, chunk_size=512, chunk_overlap=50)
    if len(chunks) >= 2:
        assert chunks[1].start_char < chunks[0].end_char


# ---------------------------------------------------------------------------
# Tests de chunk_documents()
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_chunk_documents_aggregates_all(sample_document, long_document):
    """
    chunk_documents doit retourner les chunks de TOUS les documents combinés.
    """
    chunks = chunk_documents(
        [sample_document, long_document],
        chunk_size=512,
        chunk_overlap=50,
    )
    # Minimum 1 chunk par document
    assert len(chunks) >= 2


@pytest.mark.fast
def test_chunk_documents_empty_list():
    """
    Une liste vide doit retourner une liste vide.
    """
    result = chunk_documents([], chunk_size=512, chunk_overlap=50)
    assert result == []
