# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Enterprise RAG API
# ─────────────────────────────────────────────────────────────────────────────
#
# Image de base : python:3.13-slim
#   - "slim" = Debian minimal, sans compilateurs ni headers inutiles
#   - Réduit la surface d'attaque et le poids final de l'image
#
# Architecture :
#   - Copie requirements.txt EN PREMIER pour bénéficier du cache de couches
#     Docker : si requirements.txt n'a pas changé, pip install n'est pas
#     ré-exécuté au build suivant.
#   - HF_HOME=/app/models_cache : les modèles HuggingFace sont téléchargés
#     dans ce dossier, monté en volume dans docker-compose pour persister
#     entre les redémarrages (évite de retélécharger ~600 MB à chaque fois).
#
# Construction :
#   docker build -t enterprise-rag .
#
# Lancement :
#   docker compose up
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim

# ── Métadonnées ───────────────────────────────────────────────────────────────
LABEL maintainer="thefranckt"
LABEL description="Enterprise RAG — FastAPI + gelectra-base-germanquad"
LABEL version="0.1.0"

# ── Répertoire de travail ────────────────────────────────────────────────────
WORKDIR /app

# ── Dépendances système ──────────────────────────────────────────────────────
# libgomp1 : implémentation OpenMP requise par faiss-cpu pour la parallélisation
# Les autres outils (wget, curl) sont délibérément exclus (minimalisme).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ───────────────────────────────────────────────────────
# Copie requirements.txt AVANT le code source pour maximiser le cache Docker.
# Si seul le code change (pas les dépendances), cette couche est réutilisée.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Cache des modèles HuggingFace ────────────────────────────────────────────
# HF_HOME : dossier où huggingface_hub télécharge tous les modèles/tokenizers.
# Monter /app/models_cache en volume persistant dans docker-compose évite
# de retélécharger sentence-transformers (~90 MB) + gelectra (~440 MB)
# à chaque docker compose up.
ENV HF_HOME=/app/models_cache
ENV TRANSFORMERS_CACHE=/app/models_cache

# ── Code source ──────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/

# ── Dossiers de données ───────────────────────────────────────────────────────
# Créés ici pour éviter les erreurs si les volumes ne sont pas encore montés.
RUN mkdir -p data/raw data/processed logs models_cache

# ── Port exposé ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Docker surveille l'état de l'API toutes les 30s.
# start-period=90s : délai avant le 1er check (chargement des modèles ~60s).
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ── Commande de démarrage ────────────────────────────────────────────────────
# --host 0.0.0.0 : écoute sur toutes les interfaces (requis en conteneur)
# --port 8000    : port exposé ci-dessus
# --workers 1    : 1 seul worker (les modèles ML ne supportent pas le fork)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
