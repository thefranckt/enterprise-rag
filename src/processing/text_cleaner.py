"""
Nettoyage du texte extrait des documents.

Cette étape transforme le texte brut (souvent bruité)
en texte propre et exploitable par le pipeline RAG.
"""

import re
from src.ingestion.models import Document


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut extrait d'un PDF ou TXT.

    Opérations appliquées dans l'ordre :
    1. Reconstitue les mots coupés par un tiret en fin de ligne
    2. Remplace les sauts de ligne par des espaces
    3. Supprime les espaces multiples
    4. Supprime les espaces en début et fin de texte
    """

    # 1. Reconstituer les mots coupés par tiret en fin de ligne
    # Exemple : "Versi-\ncherung" → "Versicherung"
    text = re.sub(r"-\n", "", text)

    # 2. Remplacer les sauts de ligne multiples par un seul espace
    # Exemple : "Leistungen:\n\n\n-Vollkasko" → "Leistungen: -Vollkasko"
    text = re.sub(r"\n+", " ", text)

    # 3. Supprimer les espaces multiples
    # Exemple : "Auto  Versicherung" → "Auto Versicherung"
    text = re.sub(r" {2,}", " ", text)

    # 4. Supprimer les espaces en début et fin
    text = text.strip()

    return text


def clean_document(document: Document) -> Document:
    """
    Applique le nettoyage sur un objet Document.

    Retourne un NOUVEAU Document avec le contenu nettoyé.
    Le document original n'est jamais modifié.
    """
    cleaned_content = clean_text(document.content)

    return Document(
        content=cleaned_content,
        source=document.source,
        file_type=document.file_type,
        num_pages=document.num_pages,
        metadata=document.metadata,
    )


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Nettoie une liste de Documents.

    Args:
        documents: Liste de Documents bruts

    Returns:
        Liste de Documents nettoyés
    """
    cleaned = []
    for doc in documents:
        cleaned_doc = clean_document(doc)
        original_len = len(doc.content)
        cleaned_len = len(cleaned_doc.content)
        reduction = round((1 - cleaned_len / original_len) * 100, 1)
        print(f"  ✓ Nettoyé : {doc.source} "
              f"({original_len} → {cleaned_len} chars, -{reduction}%)")
        cleaned.append(cleaned_doc)

    return cleaned