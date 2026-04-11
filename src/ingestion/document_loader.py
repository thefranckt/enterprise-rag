"""
Chargement de documents depuis le disque.
Supporte : PDF (.pdf), Texte (.txt)
"""

from pathlib import Path
from pypdf import PdfReader

from src.ingestion.models import Document
from src.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file_path: str | Path) -> Document:
    """Charge un document depuis un chemin de fichier."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Format '{path.suffix}' non supporté. "
            f"Formats acceptés : {SUPPORTED_EXTENSIONS}"
        )

    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    else:
        return _load_txt(path)


def _load_pdf(path: Path) -> Document:
    """Extrait le texte d'un fichier PDF."""
    reader = PdfReader(str(path))

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n\n".join(pages_text)

    metadata = {}
    if reader.metadata:
        metadata = {
            "author": reader.metadata.get("/Author", ""),
            "title": reader.metadata.get("/Title", ""),
            "creation_date": reader.metadata.get("/CreationDate", ""),
        }

    return Document(
        content=full_text,
        source=path.name,
        file_type="pdf",
        num_pages=len(reader.pages),
        metadata=metadata,
    )


def _load_txt(path: Path) -> Document:
    """Lit un fichier texte."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")

    return Document(
        content=content,
        source=path.name,
        file_type="txt",
        num_pages=1,
        metadata={},
    )


def load_documents_from_dir(directory: str | Path) -> list[Document]:
    """Charge tous les documents supportés depuis un dossier."""
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {dir_path}")

    documents = []
    errors = []

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                doc = load_document(file_path)
                documents.append(doc)
                logger.info("Chargé : %s", doc)
            except Exception as e:
                errors.append((file_path.name, str(e)))
                logger.warning("Erreur chargement %s : %s", file_path.name, e)

    if errors:
        logger.warning(
            "%d fichier(s) en erreur sur %d total",
            len(errors),
            len(documents) + len(errors),
        )

    return documents
