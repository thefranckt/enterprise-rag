"""
Configuration centralisée du projet Enterprise RAG.

Toutes les valeurs configurables du projet sont définies ici.
Pour modifier un paramètre, on ne touche qu'à ce fichier (ou au .env).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Chemin absolu vers la racine du projet
# __file__ = chemin vers ce fichier (configs/settings.py)
# .parent = dossier configs/
# .parent.parent = dossier racine du projet
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Paramètres de l'application.
    Les valeurs peuvent être surchargées par des variables d'environnement
    ou par le fichier .env à la racine du projet.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Informations du projet ---
    project_name: str = "Enterprise RAG"
    version: str = "0.1.0"
    debug: bool = False

    # --- Chemins des dossiers ---
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # --- Modèle d'embeddings ---
    # all-MiniLM-L6-v2 : modèle léger (~90MB), très performant, tourne sur CPU
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # taille des vecteurs produits par ce modèle

    # --- Chunking (découpage des documents) ---
    chunk_size: int = 512        # nombre de tokens par chunk
    chunk_overlap: int = 50      # chevauchement entre chunks consécutifs

    # --- Retrieval (recherche de documents) ---
    retrieval_top_k: int = 5     # nombre de chunks retournés par la recherche

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Instance unique réutilisable dans tout le projet
# On importe "settings" directement, pas la classe Settings
settings = Settings()
