from app import retrieval, generation
from app.config import *
from sentence_transformers import SentenceTransformer


retriever = retrieval.VectorRetriever(dim=384)
retriever.load("../index/")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("\nFrance RAG System (CLI Mode)")
print("Type your query below. Type 'exit' to quit.\n")

while True:
    query = input("Ask: ").strip()
    if query.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    try:
        query_embedding = embedder.encode([query])[0]
        chunks = retriever.search(query_embedding, TOP_K)
        prompt = generation.build_prompt(query, chunks)
        answer = generation.query_llm(prompt)

        print("\nAnswer:")
        print(answer)


    except Exception as e:
        print(f"Error: {e}")