"""
Tests d'intégration pour src/embeddings/embedder.py

Marqués @pytest.mark.slow car ils chargent le modèle
sentence-transformers/all-MiniLM-L6-v2 (~90 Mo).

Lancez uniquement ces tests avec :
    pytest -m slow
Excluez-les des runs rapides avec :
    pytest -m "not slow"
"""

import numpy as np
import pytest
from src.embeddings.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    """
    Charge le modèle une seule fois pour tous les tests du module.
    `scope="module"` évite de re-charger le modèle à chaque test.
    """
    return Embedder()


# ---------------------------------------------------------------------------
# Tests de embed_text()
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_embed_text_returns_numpy_array(embedder):
    """
    embed_text() doit retourner un numpy array.
    """
    result = embedder.embed_text("Hallo, das ist ein Test.")
    assert isinstance(result, np.ndarray)


@pytest.mark.slow
def test_embed_text_has_correct_shape(embedder):
    """
    Le vecteur d'embedding doit avoir exactement 384 dimensions
    (taille de sortie du modèle all-MiniLM-L6-v2).
    """
    result = embedder.embed_text("Test sentence.")
    assert result.shape == (384,)


@pytest.mark.slow
def test_embed_text_is_normalized(embedder):
    """
    Les embeddings de all-MiniLM-L6-v2 sont normalisés (norme ≈ 1.0).
    Vérifie que la norme L2 est proche de 1.
    """
    result = embedder.embed_text("Versicherung und Haftpflicht.")
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 0.01


@pytest.mark.slow
def test_embed_text_identical_sentences_are_equal(embedder):
    """
    Deux embeddings du même texte doivent être identiques
    (le modèle est déterministe).
    """
    text = "Gleicher Text ergibt gleichen Vektor."
    v1 = embedder.embed_text(text)
    v2 = embedder.embed_text(text)
    np.testing.assert_array_almost_equal(v1, v2)


@pytest.mark.slow
def test_embed_text_different_sentences_differ(embedder):
    """
    Deux textes distincts doivent produire des vecteurs différents.
    """
    v1 = embedder.embed_text("Haftpflichtversicherung Bedingungen")
    v2 = embedder.embed_text("Fahrradversicherung Tarifblatt")
    assert not np.allclose(v1, v2)


# ---------------------------------------------------------------------------
# Tests de embed_chunks()
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_embed_chunks_returns_matrix(embedder, sample_chunk):
    """
    embed_chunks() doit retourner une matrice 2D numpy.
    """
    result = embedder.embed_chunks([sample_chunk, sample_chunk])
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


@pytest.mark.slow
def test_embed_chunks_shape(embedder, sample_chunk):
    """
    La forme doit être (nombre_de_chunks, 384).
    """
    chunks = [sample_chunk] * 3
    result = embedder.embed_chunks(chunks)
    assert result.shape == (3, 384)


@pytest.mark.slow
def test_embed_chunks_cosine_similarity_same_text(embedder, sample_chunk):
    """
    La similarité cosinus entre deux embeddings du même texte doit être 1.0.
    La similarité cosinus = produit des vecteurs normalisés.
    """
    result = embedder.embed_chunks([sample_chunk, sample_chunk])
    v1, v2 = result[0], result[1]
    # Avec normalisation L2, dot product = cosine similarity
    similarity = float(np.dot(v1, v2))
    assert abs(similarity - 1.0) < 0.01
