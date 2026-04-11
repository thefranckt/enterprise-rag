"""
Configuration centralisée des logs.

Pourquoi structurer les logs ?
  - print() : invisible en production, pas de niveau, pas d'horodatage
  - logging : niveaux hiérarchiques, rotation de fichiers, sourcé par module

Architecture :
  - Un logger racine "rag" pour toute l'application
  - Chaque module crée son logger fils via get_logger(__name__)
  - Les logs remontent vers le logger racine qui porte les handlers

Niveaux (du plus verbeux au plus silencieux) :
  DEBUG < INFO < WARNING < ERROR < CRITICAL

  debug=False (prod)  : INFO  en console, DEBUG dans le fichier
  debug=True  (dev)   : DEBUG en console et dans le fichier

Usage dans chaque module :
    from src.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Pipeline chargé : %s", model_name)
    logger.warning("Index introuvable, démarrage dégradé")
    logger.error("Échec du chargement : %s", e)
    logger.debug("Détail interne (visible si debug=True)")
"""

import logging
import logging.handlers

from configs.settings import settings

_ROOT_LOGGER = "rag"


def setup_logging() -> None:
    """
    Configure les handlers du logger racine "rag".

    Idempotent : n'ajoute pas de handlers en double si appelé plusieurs fois.
    C'est important pour pytest qui importe les modules plusieurs fois.

    Handlers créés :
      Console  — niveau INFO (ou DEBUG si settings.debug=True)
                 format : [HH:MM:SS] LEVEL  nom — message
      Fichier  — niveau DEBUG, rotation 5 × 5 MB
                 → logs/app.log  (archives : app.log.1 … app.log.5)
    """
    root_logger = logging.getLogger(_ROOT_LOGGER)

    # Déjà configuré → ne rien ajouter (idempotence)
    if root_logger.handlers:
        return

    level = logging.DEBUG if settings.debug else logging.INFO
    root_logger.setLevel(level)

    # ── Console ──────────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    # ── Fichier rotatif ──────────────────────────────────────────────────────
    # RotatingFileHandler : quand app.log dépasse maxBytes il devient app.log.1
    # backupCount=5 → on garde 5 archives, soit 30 MB d'historique au total
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logs_dir / "app.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB par fichier
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger fils du logger racine "rag".

    Exemple :
        get_logger("src.embeddings.embedder")
        → logging.getLogger("rag.src.embeddings.embedder")

    Les logs remontent automatiquement vers le logger racine "rag"
    qui porte les handlers console + fichier.

    Args:
        name : Identifiant du module. Utiliser __name__ systématiquement.

    Returns:
        Un logging.Logger configuré et prêt à l'emploi.
    """
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")


# Configuration automatique à l'import de ce module
setup_logging()
