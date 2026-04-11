"""
Tests unitaires pour src/processing/text_cleaner.py

Ces tests sont PURS : aucun modèle ML chargé, aucun fichier lu.
Ils ne testent que la logique des fonctions regex.
Ils doivent s'exécuter en < 1 seconde.
"""

import pytest
from src.processing.text_cleaner import clean_text, clean_document, clean_documents
from src.ingestion.models import Document


# ---------------------------------------------------------------------------
# Tests de clean_text()
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_clean_text_removes_newlines():
    """
    Les sauts de ligne simples doivent être remplacés par des espaces.
    """
    # Arrange
    text = "Erste Zeile\nZweite Zeile"
    # Act
    result = clean_text(text)
    # Assert
    assert "\n" not in result
    assert result == "Erste Zeile Zweite Zeile"


@pytest.mark.fast
def test_clean_text_removes_multiple_spaces():
    """
    Les espaces multiples consécutifs doivent être réduits à un seul.
    """
    text = "Auto   Versicherung    Schutz"
    result = clean_text(text)
    assert result == "Auto Versicherung Schutz"


@pytest.mark.fast
def test_clean_text_repairs_hyphenated_words():
    """
    Les mots coupés par un tiret en fin de ligne doivent être reconstitués.
    "Versi-\ncherung" → "Versicherung"
    C'est typique des PDFs qui découpent les longs mots allemands.
    """
    text = "Die Versi-\ncherung übernimmt die Kosten."
    result = clean_text(text)
    assert "Versicherung" in result
    assert "Versi-" not in result


@pytest.mark.fast
def test_clean_text_strips_leading_trailing_spaces():
    """
    Les espaces en début et fin de texte doivent être supprimés.
    """
    text = "  Text mit Leerzeichen  "
    result = clean_text(text)
    assert result == "Text mit Leerzeichen"


@pytest.mark.fast
def test_clean_text_empty_string():
    """
    Une chaîne vide doit rester vide après nettoyage.
    """
    result = clean_text("")
    assert result == ""


# ---------------------------------------------------------------------------
# Tests de clean_document()
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_clean_document_returns_new_document(sample_document):
    """
    clean_document doit retourner un NOUVEAU Document (immutabilité).
    Le document original ne doit pas être modifié.

    Principe d'immutabilité : les fonctions de transformation ne doivent
    jamais modifier leurs entrées — elles retournent de nouveaux objets.
    """
    original_content = sample_document.content
    cleaned = clean_document(sample_document)

    # L'original n'est pas modifié
    assert sample_document.content == original_content
    # On a bien un nouvel objet
    assert cleaned is not sample_document


@pytest.mark.fast
def test_clean_document_preserves_metadata(sample_document):
    """
    Le nettoyage ne doit pas perdre les métadonnées du document
    (source, file_type, num_pages).
    """
    cleaned = clean_document(sample_document)

    assert cleaned.source == sample_document.source
    assert cleaned.file_type == sample_document.file_type
    assert cleaned.num_pages == sample_document.num_pages


# ---------------------------------------------------------------------------
# Tests de clean_documents()
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_clean_documents_returns_same_count(sample_document):
    """
    La liste retournée doit avoir le même nombre d'éléments.
    """
    documents = [sample_document, sample_document]
    cleaned = clean_documents(documents)
    assert len(cleaned) == 2


@pytest.mark.fast
def test_clean_documents_empty_list():
    """
    Une liste vide en entrée doit produire une liste vide en sortie.
    """
    result = clean_documents([])
    assert result == []
