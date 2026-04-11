"""
Interface Gradio pour le pipeline RAG.

Gradio permet de créer une UI web interactive en quelques lignes Python.
Contrairement à FastAPI (interface développeur/JSON), Gradio cible
les utilisateurs finaux : champ texte, bouton, résultat visuel.

Architecture :
  - Le pipeline RAG est chargé UNE SEULE FOIS au démarrage du module
  - Chaque requête utilisateur appelle la fonction `answer_question()`
  - Gradio gère automatiquement le threading et le serveur HTTP
"""

import gradio as gr

from src.logger import get_logger
from src.pipeline import RAGPipeline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chargement du pipeline (une seule fois, au démarrage)
# ---------------------------------------------------------------------------

logger.info("Chargement du pipeline RAG...")
rag = RAGPipeline()
logger.info("Pipeline prêt.")


# ---------------------------------------------------------------------------
# Fonction principale appelée par Gradio à chaque soumission
# ---------------------------------------------------------------------------

def answer_question(question: str, top_k: int) -> tuple[str, str, str]:
    """
    Reçoit la question de l'utilisateur, interroge le pipeline RAG,
    retourne trois valeurs que Gradio affichera dans trois composants.

    Args:
        question : La question posée par l'utilisateur
        top_k    : Nombre de chunks à récupérer (contrôlé par le slider)

    Returns:
        tuple (answer, sources, details_markdown)
    """
    if not question.strip():
        return "Bitte eine Frage eingeben.", "", ""

    response = rag.query(question, top_k=top_k)

    # Sources : liste simple
    sources_text = "\n".join(f"• {s}" for s in response.sources)

    # Détails : tableau Markdown des chunks récupérés
    # Gradio supporte le Markdown nativement dans les composants Textbox/Markdown
    details_lines = ["| # | Source | Chunk | Score | Aperçu |",
                     "|---|--------|-------|-------|--------|"]
    for r in response.results:
        preview = r.chunk.text[:80].replace("|", "/").replace("\n", " ")
        details_lines.append(
            f"| {r.rank} | {r.chunk.source} | {r.chunk.chunk_index} "
            f"| {r.score:.3f} | {preview}... |"
        )
    details_md = "\n".join(details_lines)

    return response.answer, sources_text, details_md


# ---------------------------------------------------------------------------
# Construction de l'interface avec gr.Blocks
# ---------------------------------------------------------------------------
# gr.Blocks permet de contrôler précisément la mise en page.
# Alternative : gr.Interface (plus simple mais moins flexible).

def build_interface() -> gr.Blocks:
    """Construit et retourne l'interface Gradio."""

    with gr.Blocks(
        title="Enterprise RAG — Insurance Assistant",
        theme=gr.themes.Soft(),   # thème visuel propre
    ) as demo:

        # --- En-tête ---
        gr.Markdown("""
        # Enterprise RAG — Insurance Assistant
        Posez une question en **allemand** sur les documents d'assurance indexés.
        Le système retrouve les passages pertinents et extrait la réponse directement du texte.
        """)

        # --- Inputs ---
        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Frage (Question)",
                    placeholder="z.B. Was ist bei einem Autounfall zu tun?",
                    lines=2,
                )
            with gr.Column(scale=1):
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Top-K chunks",
                    # Top-K = nombre de passages récupérés avant génération
                    # Plus de chunks = plus de contexte, mais aussi plus de bruit
                )

        submit_btn = gr.Button("Antwort suchen", variant="primary")

        # --- Outputs ---
        gr.Markdown("### Ergebnis (Résultat)")

        answer_output = gr.Textbox(
            label="Antwort (Réponse extraite)",
            lines=3,
            interactive=False,
        )

        with gr.Row():
            sources_output = gr.Textbox(
                label="Quellen (Sources)",
                lines=4,
                interactive=False,
            )
            details_output = gr.Markdown(
                label="Chunk-Details",
                value="*Résultats de la recherche sémantique apparaîtront ici.*"
            )

        # --- Exemples cliquables ---
        gr.Examples(
            examples=[
                ["Was ist bei einem Autounfall zu tun?", 3],
                ["Ist Diebstahl des Fahrrads versichert?", 3],
                ["Was passiert wenn mein Auto im Ausland liegen bleibt?", 5],
                ["Welche Schäden deckt die Cyber-Versicherung ab?", 3],
            ],
            inputs=[question_input, topk_slider],
        )

        # --- Connexion bouton → fonction ---
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, topk_slider],
            outputs=[answer_output, sources_output, details_output],
        )

        # Aussi déclenché par Entrée dans le champ texte
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, topk_slider],
            outputs=[answer_output, sources_output, details_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,   # True = génère un lien public temporaire (Hugging Face tunnel)
    )
