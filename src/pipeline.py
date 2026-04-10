"""
Pipeline RAG complet.

Orchestre toutes les étapes :
1. Chargement de l'index vectoriel
2. Embedding de la question
3. Recherche sémantique
4. Génération de la réponse

C'est ce module qui sera utilisé par l'API FastAPI et l'interface Gradio.
"""

from dataclasses import dataclass

from configs.settings import settings
from src.embeddings.embedder import Embedder
from src.generation.generator import Generator
from src.retrieval.vector_store import SearchResult, VectorStore


@dataclass
class RAGResponse:
    """La réponse complète du pipeline RAG."""
    question: str           # La question posée
    answer: str             # La réponse générée
    sources: list[str]      # Les fichiers sources utilisés
    results: list[SearchResult]  # Les chunks récupérés (pour debug/transparence)


class RAGPipeline:
    """
    Pipeline RAG complet et réutilisable.

    Charge une fois le modèle et l'index, répond ensuite
    à autant de questions qu'on veut sans rechargement.
    """

    def __init__(self):
        print("=== Initialisation du pipeline RAG ===\n")

        # Charger l'index vectoriel (déjà construit et sauvegardé)
        print("1/3 Chargement de l'index vectoriel...")
        self.store = VectorStore.load()

        # Charger le modèle d'embeddings
        print("\n2/3 Chargement du modèle d'embeddings...")
        self.embedder = Embedder()

        # Charger le générateur
        print("\n3/3 Chargement du générateur...")
        self.generator = Generator()

        print("\n=== Pipeline prêt ===\n")

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """
        Répond à une question en utilisant les documents indexés.

        Args:
            question : La question en langage naturel
            top_k    : Nombre de chunks à récupérer (défaut : settings.retrieval_top_k)

        Returns:
            RAGResponse avec la réponse et les sources
        """
        top_k = top_k or settings.retrieval_top_k

        # Étape 1 : transformer la question en vecteur
        query_vector = self.embedder.embed_text(question)

        # Étape 2 : rechercher les chunks les plus pertinents
        results = self.store.search(query_vector, top_k=top_k)

        # Étape 3 : générer une réponse à partir des chunks
        answer = self.generator.generate(question, results)

        # Extraire les sources uniques (sans doublons)
        sources = list(dict.fromkeys(r.chunk.source for r in results))

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            results=results,
        )