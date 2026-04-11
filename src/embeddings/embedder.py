"""
Génération d'embeddings pour les chunks de texte.

Transforme chaque chunk en vecteur numérique de dimension 384
en utilisant le modèle sentence-transformers/all-MiniLM-L6-v2.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from configs.settings import settings
from src.logger import get_logger
from src.processing.chunker import Chunk

logger = get_logger(__name__)


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
        logger.info("Chargement du modèle : %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = settings.embedding_dimension
        logger.info("Modèle prêt. Dimension des vecteurs : %d", self.dimension)

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

        logger.info("Encodage de %d chunks (batch_size=%d)...", len(texts), batch_size)

        # Le modèle encode tous les textes en une seule passe optimisée
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        logger.info("Embeddings générés : shape=%s", embeddings.shape)
        return embeddings
    