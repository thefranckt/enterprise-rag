from src.pipeline import RAGPipeline

# Initialisation (charge tout une seule fois)
rag = RAGPipeline()

# Questions de test — couvrant les 5 documents
questions = [
    # auto.pdf
    "Was ist bei einem Autounfall zu tun?",
    "Welche Dokumente brauche ich nach einem Unfall?",
    # fahrrad.pdf
    "Welche Leistungen sind im Fahrradschutz enthalten?",
    "Ist Diebstahl des Fahrrads versichert?",
    # homeassist.pdf
    "Was macht der Haushaltsassistenz-Service?",
    "Hilft der Service bei einem Rohrbruch zu Hause?",
    # multiassist.pdf
    "Welche Länder sind durch den Multiassist abgedeckt?",
    "Was passiert wenn mein Auto im Ausland liegen bleibt?",
    # websecure_gewerbe.pdf
    "Was ist eine Cyber-Versicherung für Unternehmen?",
    "Welche Schäden deckt die Cyber-Versicherung ab?",
]

print("\n" + "="*60)
for question in questions:
    print(f"\nFRAGE : {question}")
    response = rag.query(question, top_k=3)
    print(f"ANTWORT : {response.answer}")
    print(f"QUELLEN : {', '.join(response.sources)}")
    print("-"*60)