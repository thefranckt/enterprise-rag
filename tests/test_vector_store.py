"""
Tests d'intégration pour src/vector_store/store.py

Marqués @pytest.mark.slow car ils utilisent l'Embedder (chargement modèle).
Utilisent `tmp_path` pour save/load sans toucher au vrai index FAISS du projet.
"""

import numpy as np
import pytest
from src.retrieval.vector_store import VectorStore, SearchResult
from src.embeddings.embedder import Embedder
from src.processing.chunker import Chunk


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def sample_chunks():
    """Trois chunks distincts pour construire un index de test."""
    return [
        Chunk(
            text="Die Haftpflichtversicherung schützt vor Schadensersatzansprüchen.",
            source="haftpflicht.pdf",
            chunk_index=0,
            start_char=0,
            end_char=65,
        ),
        Chunk(
            text="Das Fahrrad ist gegen Diebstahl versichert.",
            source="fahrrad.pdf",
            chunk_index=0,
            start_char=0,
            end_char=43,
        ),
        Chunk(
            text="Der Versicherungsnehmer hat eine Jahresprämie zu zahlen.",
            source="vertrag.pdf",
            chunk_index=0,
            start_char=0,
            end_char=55,
        ),
    ]


@pytest.fixture
def built_store(embedder, sample_chunks):
    """Retourne un VectorStore déjà construit avec les sample_chunks."""
    embeddings = embedder.embed_chunks(sample_chunks)
    store = VectorStore()
    store.build(sample_chunks, embeddings)
    return store


# ---------------------------------------------------------------------------
# Tests de build()
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_build_creates_non_empty_store(built_store):
    """
    Après build(), le store doit contenir des vecteurs.
    """
    assert built_store.index is not None
    assert built_store.index.ntotal == 3


@pytest.mark.slow
def test_build_stores_chunks(built_store, sample_chunks):
    """
    Les chunks doivent être accessibles après build().
    """
    assert len(built_store.chunks) == len(sample_chunks)


# ---------------------------------------------------------------------------
# Tests de search()
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_search_returns_correct_count(embedder, built_store):
    """
    search() avec top_k=2 doit retourner exactement 2 résultats.
    """
    query_vec = embedder.embed_text("Versicherung")
    results = built_store.search(query_vec, top_k=2)
    assert len(results) == 2


@pytest.mark.slow
def test_search_returns_search_results(embedder, built_store):
    """
    Chaque élément retourné doit être un SearchResult
    avec les attributs chunk et score.
    """
    query_vec = embedder.embed_text("Haftpflicht")
    results = built_store.search(query_vec, top_k=1)
    result = results[0]
    assert isinstance(result, SearchResult)
    assert isinstance(result.chunk, Chunk)
    assert isinstance(result.score, float)


@pytest.mark.slow
def test_search_scores_are_non_negative(embedder, built_store):
    """
    FAISS IndexFlatL2 retourne des distances L2 (≥ 0).
    """
    query_vec = embedder.embed_text("Jahresprämie")
    results = built_store.search(query_vec, top_k=3)
    for result in results:
        assert result.score >= 0.0


@pytest.mark.slow
def test_search_most_relevant_chunk(embedder, built_store):
    """
    La requête la plus proche sémantiquement du chunk 0
    doit retourner ce chunk en premier résultat.
    Haftpflicht est très proche du premier chunk.
    """
    query_vec = embedder.embed_text("Haftpflichtversicherung Schadensersatz")
    results = built_store.search(query_vec, top_k=1)
    assert "Haftpflicht" in results[0].chunk.text


# ---------------------------------------------------------------------------
# Tests de save() / load()
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_save_and_load_preserves_index_size(embedder, built_store, tmp_path):
    """
    Après save() puis load(), le nombre de vecteurs dans l'index
    doit être identique à l'original.
    `save(directory)` crée faiss_index.bin + chunks.pkl dans ce dossier.
    `VectorStore.load(directory)` est une classmethod.
    """
    save_dir = tmp_path / "store1"

    built_store.save(save_dir)
    new_store = VectorStore.load(save_dir)

    assert new_store.index.ntotal == built_store.index.ntotal


@pytest.mark.slow
def test_save_and_load_preserves_chunks(embedder, built_store, tmp_path):
    """
    Les chunks chargés doivent avoir les mêmes sources que les originaux.
    """
    save_dir = tmp_path / "store2"

    built_store.save(save_dir)
    new_store = VectorStore.load(save_dir)

    original_sources = {c.source for c in built_store.chunks}
    loaded_sources = {c.source for c in new_store.chunks}
    assert original_sources == loaded_sources


@pytest.mark.slow
def test_loaded_store_can_search(embedder, built_store, tmp_path):
    """
    Un store chargé depuis disque doit pouvoir effectuer des recherches.
    """
    save_dir = tmp_path / "store3"

    built_store.save(save_dir)
    new_store = VectorStore.load(save_dir)

    query_vec = embedder.embed_text("Versicherung")
    results = new_store.search(query_vec, top_k=1)
    assert len(results) == 1
