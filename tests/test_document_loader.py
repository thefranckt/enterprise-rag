"""
Tests unitaires pour src/ingestion/document_loader.py

Utilise `tmp_path` (fixture pytest built-in) pour créer des fichiers
temporaires sans toucher aux vraies données du projet.
`tmp_path` est automatiquement nettoyé après chaque test.
"""

import pytest
from src.ingestion.document_loader import load_document, load_documents_from_dir
from src.ingestion.models import Document


# ---------------------------------------------------------------------------
# Tests de load_document()
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_load_document_txt_returns_document(tmp_path):
    """
    load_document() doit retourner un objet Document valide
    quand on lui passe un fichier .txt existant.
    """
    # Arrange
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Contenu de test pour le loader.", encoding="utf-8")

    # Act
    doc = load_document(str(txt_file))

    # Assert
    assert isinstance(doc, Document)


@pytest.mark.fast
def test_load_document_txt_content_is_correct(tmp_path):
    """
    Le texte extrait doit contenir le contenu du fichier .txt.
    """
    content = "Hier ist ein Versicherungstext."
    txt_file = tmp_path / "versicherung.txt"
    txt_file.write_text(content, encoding="utf-8")

    doc = load_document(str(txt_file))

    assert content in doc.content


@pytest.mark.fast
def test_load_document_sets_source(tmp_path):
    """
    Le champ `source` du Document doit correspondre au chemin du fichier.
    """
    txt_file = tmp_path / "kontakt.txt"
    txt_file.write_text("Some text here.", encoding="utf-8")

    doc = load_document(str(txt_file))

    assert "kontakt.txt" in doc.source


@pytest.mark.fast
def test_load_document_unsupported_extension_raises(tmp_path):
    """
    Un fichier avec une extension non supportée (ex: .docx) doit lever
    une ValueError ou NotImplementedError.
    """
    unsupported = tmp_path / "document.docx"
    unsupported.write_bytes(b"fake docx content")

    with pytest.raises((ValueError, NotImplementedError)):
        load_document(str(unsupported))


@pytest.mark.fast
def test_load_document_nonexistent_file_raises():
    """
    Un chemin vers un fichier inexistant doit lever une exception
    (FileNotFoundError ou autre).
    """
    with pytest.raises(Exception):
        load_document("/chemin/qui/nexiste/pas.txt")


# ---------------------------------------------------------------------------
# Tests de load_documents_from_dir()
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_load_documents_from_dir_counts_files(tmp_path):
    """
    load_documents_from_dir() doit retourner autant de Documents
    qu'il y a de fichiers supportés dans le répertoire.
    """
    (tmp_path / "a.txt").write_text("Texte A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Texte B", encoding="utf-8")
    (tmp_path / "c.txt").write_text("Texte C", encoding="utf-8")

    docs = load_documents_from_dir(str(tmp_path))

    assert len(docs) == 3


@pytest.mark.fast
def test_load_documents_from_dir_all_are_documents(tmp_path):
    """
    Chaque élément retourné doit être une instance de Document.
    """
    (tmp_path / "d.txt").write_text("Texte D", encoding="utf-8")

    docs = load_documents_from_dir(str(tmp_path))

    for doc in docs:
        assert isinstance(doc, Document)


@pytest.mark.fast
def test_load_documents_from_dir_empty_dir(tmp_path):
    """
    Un répertoire vide doit retourner une liste vide.
    """
    docs = load_documents_from_dir(str(tmp_path))
    assert docs == []
