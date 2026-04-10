"""
Index vectoriel basé sur FAISS.

Responsabilités :
- Construire l'index à partir des embeddings
- Sauvegarder et charger l'index depuis le disque
- Rechercher les chunks les plus similaires à une requête
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from configs.settings import settings
from src.processing.chunker import Chunk


@dataclass
class SearchResult:
    """Résultat d'une recherche sémantique."""
    chunk: Chunk          # Le chunk retrouvé
    score: float          # Score de similarité (plus proche de 0 = plus similaire en L2)
    rank: int             # Position dans les résultats (0 = meilleur)


class VectorStore:
    """
    Gère l'index FAISS et la recherche sémantique.

    Usage :
        store = VectorStore()
        store.build(chunks, embeddings)
        store.save()

        # Plus tard :
        store = VectorStore.load()
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self):
        self.index = None           # L'index FAISS
        self.chunks: list[Chunk] = []  # Les chunks associés aux vecteurs

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Construit l'index FAISS à partir des embeddings.

        Args:
            chunks     : Liste des Chunks (même ordre que embeddings)
            embeddings : Matrice numpy (nb_chunks, dimension)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Nombre de chunks ({len(chunks)}) "
                f"!= nombre d'embeddings ({len(embeddings)})"
            )

        dimension = embeddings.shape[1]

        # Créer un index FAISS de type "Flat L2" (comparaison exacte, distance euclidienne)
        self.index = faiss.IndexFlatL2(dimension)

        # FAISS requiert des vecteurs en float32
        vectors = embeddings.astype(np.float32)

        # Ajouter tous les vecteurs à l'index
        self.index.add(vectors)
        self.chunks = chunks

        print(f"Index construit : {self.index.ntotal} vecteurs de dimension {dimension}")

    def search(self, query_vector: np.ndarray, top_k: int | None = None) -> list[SearchResult]:
        """
        Recherche les chunks les plus similaires au vecteur requête.

        Args:
            query_vector : Vecteur de la question (dimension 384)
            top_k        : Nombre de résultats à retourner

        Returns:
            Liste de SearchResult triés par similarité (meilleur en premier)
        """
        if self.index is None:
            raise RuntimeError("L'index n'est pas construit. Appelle build() d'abord.")

        top_k = top_k or settings.retrieval_top_k

        # FAISS attend une matrice 2D : (1, dimension)
        query = query_vector.astype(np.float32).reshape(1, -1)

        # Recherche : retourne distances et indices des top_k voisins
        distances, indices = self.index.search(query, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS retourne -1 si pas assez de vecteurs
                continue
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(dist),
                rank=rank,
            ))

        return results

    def save(self, directory: Path | None = None) -> None:
        """Sauvegarde l'index et les chunks sur le disque."""
        save_dir = directory or settings.data_processed_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder l'index FAISS
        faiss_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_path))

        # Sauvegarder les chunks avec pickle
        chunks_path = save_dir / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Index sauvegardé dans : {save_dir}")
        print(f"  → {faiss_path.name} ({faiss_path.stat().st_size / 1024:.1f} KB)")
        print(f"  → {chunks_path.name} ({chunks_path.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load(cls, directory: Path | None = None) -> "VectorStore":
        """
        Charge un index sauvegardé depuis le disque.

        C'est une classmethod : on l'appelle sur la classe, pas sur une instance.
        Exemple : store = VectorStore.load()
        """
        load_dir = directory or settings.data_processed_dir
        load_dir = Path(load_dir)

        faiss_path = load_dir / "faiss_index.bin"
        chunks_path = load_dir / "chunks.pkl"

        if not faiss_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index introuvable dans {load_dir}. "
                "Lance d'abord le script d'indexation."
            )

        store = cls()
        store.index = faiss.read_index(str(faiss_path))

        with open(chunks_path, "rb") as f:
            store.chunks = pickle.load(f)

        print(f"Index chargé : {store.index.ntotal} vecteurs")
        return store