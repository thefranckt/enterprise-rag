"""
Génération d'embeddings pour les chunks de texte.

Transforme chaque chunk en vecteur numérique de dimension 384
en utilisant le modèle sentence-transformers/all-MiniLM-L6-v2.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from configs.settings import settings
from src.processing.chunker import Chunk


class Embedder:
    """
    Encapsule le modèle d'embeddings et expose une interface simple.

    Usage :
        embedder = Embedder()
        vectors = embedder.embed_chunks(chunks)
    """

    def __init__(self, model_name: str | None = None):
        """
        Charge le modèle d'embeddings.

        Le modèle est téléchargé depuis Hugging Face au premier appel,
        puis mis en cache localement (~90MB).
        """
        model_name = model_name or settings.embedding_model_name
        print(f"Chargement du modèle : {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = settings.embedding_dimension
        print(f"Modèle prêt. Dimension des vecteurs : {self.dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Transforme un texte en vecteur.

        Args:
            text: Texte à transformer

        Returns:
            Vecteur numpy de dimension 384
        """
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Transforme une liste de Chunks en matrice de vecteurs.

        Args:
            chunks      : Liste de Chunks à encoder
            batch_size  : Nombre de chunks traités en même temps
            show_progress: Afficher une barre de progression

        Returns:
            Matrice numpy de shape (nb_chunks, 384)
            La ligne i correspond au vecteur du chunk i.
        """
        # Extraire uniquement le texte de chaque chunk
        texts = [chunk.text for chunk in chunks]

        print(f"Encodage de {len(texts)} chunks (batch_size={batch_size})...")

        # Le modèle encode tous les textes en une seule passe optimisée
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        print(f"Embeddings générés : shape={embeddings.shape}")
        return embeddings
    