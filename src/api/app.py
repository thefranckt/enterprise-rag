"""
API FastAPI pour le pipeline RAG.

Endpoints :
  GET  /         → info sur l'API
  GET  /health   → statut du serveur
  POST /query    → poser une question au RAG
  POST /index    → réindexer les documents
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from configs.settings import settings
from src.embeddings.embedder import Embedder
from src.generation.generator import Generator
from src.ingestion import load_documents_from_dir
from src.processing import clean_documents, chunk_documents
from src.retrieval.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Modèles de données pour l'API (requêtes et réponses)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Corps de la requête POST /query."""
    question: str = Field(
        ...,                          # ... = champ obligatoire
        min_length=3,
        max_length=500,
        description="La question à poser au système RAG",
        examples=["Was ist bei einem Autounfall zu tun?"]
    )
    top_k: int = Field(
        default=3,
        ge=1,                         # ge = greater or equal
        le=10,
        description="Nombre de chunks à récupérer"
    )


class SourceInfo(BaseModel):
    """Informations sur un chunk source retourné."""
    source: str
    chunk_index: int
    score: float
    preview: str


class QueryResponse(BaseModel):
    """Corps de la réponse POST /query."""
    question: str
    answer: str
    sources: list[str]
    source_details: list[SourceInfo]


class HealthResponse(BaseModel):
    """Corps de la réponse GET /health."""
    status: str
    pipeline_loaded: bool
    documents_indexed: int
    version: str


# ---------------------------------------------------------------------------
# État global de l'application (chargé une seule fois au démarrage)
# ---------------------------------------------------------------------------

# Dictionnaire qui stocke les composants lourds en mémoire
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.

    Le code AVANT yield s'exécute au démarrage du serveur.
    Le code APRÈS yield s'exécute à l'arrêt.
    """
    # --- DÉMARRAGE ---
    print("Démarrage de l'API — chargement des composants...")

    try:
        app_state["store"] = VectorStore.load()
        app_state["embedder"] = Embedder()
        app_state["generator"] = Generator()
        app_state["ready"] = True
        print("API prête.")
    except FileNotFoundError:
        print("Index introuvable — lance d'abord le script d'indexation.")
        app_state["ready"] = False

    yield  # L'application tourne ici

    # --- ARRÊT ---
    print("Arrêt de l'API.")
    app_state.clear()


# ---------------------------------------------------------------------------
# Création de l'application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Enterprise RAG API — Document Intelligence System",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    """Point d'entrée — informations sur l'API."""
    return {
        "name": settings.project_name,
        "version": settings.version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Vérifie l'état du serveur et du pipeline."""
    store = app_state.get("store")
    return HealthResponse(
        status="ok" if app_state.get("ready") else "degraded",
        pipeline_loaded=app_state.get("ready", False),
        documents_indexed=store.index.ntotal if store else 0,
        version=settings.version,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Pose une question au système RAG.

    Le système recherche les chunks pertinents et génère une réponse.
    """
    if not app_state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="Pipeline non disponible. L'index n'est pas chargé."
        )

    store: VectorStore = app_state["store"]
    embedder: Embedder = app_state["embedder"]
    generator: Generator = app_state["generator"]

    # Recherche sémantique
    query_vector = embedder.embed_text(request.question)
    results = store.search(query_vector, top_k=request.top_k)

    # Génération de la réponse
    answer = generator.generate(request.question, results)

    # Sources uniques (ordre préservé)
    sources = list(dict.fromkeys(r.chunk.source for r in results))

    # Détails des sources pour la transparence
    source_details = [
        SourceInfo(
            source=r.chunk.source,
            chunk_index=r.chunk.chunk_index,
            score=round(r.score, 4),
            preview=r.chunk.text[:150],
        )
        for r in results
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        source_details=source_details,
    )


@app.post("/index", tags=["Admin"])
def reindex():
    """
    Réindexe tous les documents dans data/raw/.

    À appeler après avoir ajouté de nouveaux documents.
    """
    if not app_state.get("embedder"):
        raise HTTPException(status_code=503, detail="Embedder non disponible.")

    embedder: Embedder = app_state["embedder"]

    # Pipeline d'indexation complet
    raw_docs = load_documents_from_dir(settings.data_raw_dir)
    clean_docs = clean_documents(raw_docs)
    chunks = chunk_documents(clean_docs, settings.chunk_size, settings.chunk_overlap)
    embeddings = embedder.embed_chunks(chunks, show_progress=False)

    store = VectorStore()
    store.build(chunks, embeddings)
    store.save()

    # Mettre à jour l'index en mémoire
    app_state["store"] = store

    return {
        "status": "ok",
        "documents": len(raw_docs),
        "chunks": len(chunks),
    }