"""
Génération de réponses à partir des chunks récupérés.

Utilise flan-t5-base de Google — modèle léger (~250MB), CPU-compatible,
entraîné pour suivre des instructions.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from configs.settings import settings
from src.retrieval.vector_store import SearchResult


# Prompt template : structure standard d'un RAG
PROMPT_TEMPLATE = """Answer the following question based only on the provided context.
If the answer is not in the context, say "I cannot find this information in the documents."

Context:
{context}

Question: {question}

Answer:"""


class Generator:
    """
    Encapsule le LLM pour la génération de réponses.

    Usage :
        generator = Generator()
        answer = generator.generate(question, search_results)
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Charge le modèle de génération.

        flan-t5-base : ~250MB, CPU-friendly, entraîné pour suivre des instructions.
        """
        print(f"Chargement du générateur : {model_name}")
        self.model_name = model_name

        # T5Tokenizer utilise sentencepiece directement
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()  # mode inférence, pas d'entraînement
        print("Générateur prêt.")

    def build_prompt(self, question: str, results: list[SearchResult]) -> str:
        """
        Construit le prompt à envoyer au LLM.

        Assemble les chunks récupérés en un bloc de contexte,
        puis l'insère dans le template.
        """
        # Construire le contexte à partir des chunks récupérés
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(
                f"[Source: {result.chunk.source}, chunk {result.chunk.chunk_index}]\n"
                f"{result.chunk.text}"
            )

        context = "\n\n".join(context_parts)

        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, question: str, results: list[SearchResult]) -> str:
        """
        Génère une réponse à partir de la question et des chunks récupérés.

        Args:
            question : La question de l'utilisateur
            results  : Les chunks récupérés par la recherche

        Returns:
            La réponse générée sous forme de string
        """
        if not results:
            return "Aucun document pertinent trouvé pour répondre à cette question."

        prompt = self.build_prompt(question, results)

        # Tokenizer : convertit le texte en ids numériques pour le modèle
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",   # format PyTorch tensors
            max_length=1024,
            truncation=True,
        )

        # Génération sans calcul de gradient (économise la mémoire)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                do_sample=False,   # réponse déterministe
            )

        # Décoder les ids en texte
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()
    