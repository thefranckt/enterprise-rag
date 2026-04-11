"""
Extraction de réponses à partir des chunks récupérés.

Approche : QA extractif (pas génératif).
Modèle   : deepset/gelectra-base-germanquad (~440MB, CPU-compatible)

Différence fondamentale avec flan-t5 :
  - Génératif  : le modèle FABRIQUE une réponse mot par mot
  - Extractif  : le modèle SURLIGNE la span de réponse dans le texte source

Le modèle calcule deux probabilités pour chaque token du contexte :
  - P(token est le début de la réponse)
  - P(token est la fin de la réponse)
Il retourne la span [start, end] avec le score de confiance le plus élevé.

Exemple :
  Contexte  : "Die Versicherung kostet 68,75 Euro pro Jahr."
  Question  : "Was kostet die Versicherung?"
  Réponse   : "68,75 Euro pro Jahr"  ← extrait directement du texte
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.logger import get_logger
from src.retrieval.vector_store import SearchResult

logger = get_logger(__name__)

# Seuil de confiance minimum : en dessous, on considère que la réponse est absente
CONFIDENCE_THRESHOLD = 0.01


class Generator:
    """
    Encapsule le modèle QA extractif pour l'extraction de réponses.

    Usage :
        generator = Generator()
        answer = generator.generate(question, search_results)
    """

    def __init__(self, model_name: str = "deepset/gelectra-base-germanquad"):
        """
        Charge le modèle QA extractif directement (sans pipeline()).

        gelectra-base-germanquad :
          - Basé sur ELECTRA (variante de BERT)
          - Fine-tuné sur GermanQuAD (dataset QA en allemand)
          - ~440MB, CPU-compatible
          - Retourne une span extraite du contexte + score de confiance

        Pourquoi pas pipeline("question-answering") ?
          La version de transformers installée a retiré ce nom de tâche.
          On utilise directement AutoTokenizer + AutoModelForQuestionAnswering,
          ce qui est équivalent mais indépendant du registre de tâches.
        """
        logger.info("Chargement du générateur : %s", model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        logger.info("Générateur prêt.")

    def build_context(self, results: list[SearchResult]) -> str:
        """
        Assemble les chunks récupérés en un seul bloc de texte.

        Note : gelectra est limité à 512 tokens (BERT-like).
        On concatène les chunks — les plus pertinents sont en premier
        (FAISS les retourne déjà triés par score de similarité).
        """
        return " ".join(result.chunk.text for result in results)

    def generate(self, question: str, results: list[SearchResult]) -> str:
        """
        Extrait la réponse à la question depuis les chunks récupérés.

        Architecture extractive :
          Le modèle calcule pour chaque token du contexte :
            - P(token = début de la réponse)  → start_logits
            - P(token = fin de la réponse)    → end_logits
          On prend les positions argmax et on extrait la span [start, end].

        Args:
            question : La question en langage naturel (allemand)
            results  : Les chunks récupérés par la recherche FAISS

        Returns:
            La span de texte extraite du contexte, ou un message d'absence.
        """
        if not results:
            return "Aucun document pertinent trouvé pour répondre à cette question."

        context = self.build_context(results)

        # Tokenisation : question + contexte séparés par [SEP]
        # truncation=True coupe si > 512 tokens (limite BERT)
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        input_ids = inputs["input_ids"][0]

        # Structure : [CLS] question [SEP] contexte [SEP]
        # On doit chercher la réponse UNIQUEMENT dans les tokens du contexte.
        # Sans cela, le modèle peut pointer sur les tokens de la question
        # et retourner la question elle-même comme réponse.
        sep_token_id = self.tokenizer.sep_token_id
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        context_start = sep_positions[0].item() + 1  # premier token après le 1er [SEP]

        # On masque les logits hors contexte avec -inf
        start_logits = outputs.start_logits[0].clone()
        end_logits = outputs.end_logits[0].clone()
        start_logits[:context_start] = float("-inf")
        end_logits[:context_start] = float("-inf")

        # On masque aussi les tokens spéciaux ([CLS], [SEP], [PAD])
        # car ils decodent en vide avec skip_special_tokens=True
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        for idx, token_id in enumerate(input_ids):
            if token_id.item() in special_ids:
                start_logits[idx] = float("-inf")
                end_logits[idx] = float("-inf")

        start_idx = start_logits.argmax().item()
        end_idx = end_logits.argmax().item() + 1  # end est exclusif

        # Limite la longueur de la span pour éviter les extractions trop longues
        MAX_ANSWER_TOKENS = 80

        # Problème du double-argmax indépendant :
        # argmax(start) et argmax(end) pris séparément peuvent produire une
        # span incohérente (end < start) ou un token unique qui décode en vide.
        #
        # Solution : chercher la meilleure paire (s, e) qui maximise
        # start_score[s] + end_score[e] avec la contrainte s <= e <= s + MAX.
        # On vectorise ça avec des opérations tensorielles (pas de double boucle).
        start_probs = torch.softmax(start_logits, dim=-1)  # (seq_len,)
        end_probs = torch.softmax(end_logits, dim=-1)      # (seq_len,)

        # Matrice des scores : score[i,j] = start_probs[i] + end_probs[j]
        # On masque les cases où j < i ou j - i > MAX_ANSWER_TOKENS
        n = start_probs.shape[0]
        scores = start_probs.unsqueeze(1) + end_probs.unsqueeze(0)  # (n, n)
        # Masque triangulaire inférieure (end < start) + distance max
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)
        mask |= torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=MAX_ANSWER_TOKENS)
        scores[mask] = 0.0

        best = scores.argmax()
        start_idx = (best // n).item()
        end_idx = (best % n).item() + 1

        confidence = scores[start_idx, end_idx - 1].item() / 2  # ramène entre 0 et 1

        if confidence < CONFIDENCE_THRESHOLD or end_idx <= start_idx:
            return "Diese Information konnte in den Dokumenten nicht gefunden werden."

        # Reconvertir les ids de tokens en texte
        answer_ids = input_ids[start_idx:end_idx]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        # Fallback si le décodage produit une chaîne vide (token spécial résiduel)
        if not answer:
            return "Diese Information konnte in den Dokumenten nicht gefunden werden."

        return answer




