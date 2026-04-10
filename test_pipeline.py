from src.pipeline import RAGPipeline

# Initialisation (charge tout une seule fois)
rag = RAGPipeline()

# Questions de test
questions = [
    "Was ist bei einem Autounfall zu tun?",
    "Welche Leistungen sind im Fahrradschutz enthalten?",
    "Was kostet die Versicherung?",
]

print("\n" + "="*60)
for question in questions:
    print(f"\nFRAGE : {question}")
    response = rag.query(question, top_k=3)
    print(f"ANTWORT : {response.answer}")
    print(f"QUELLEN : {', '.join(response.sources)}")
    print("-"*60)