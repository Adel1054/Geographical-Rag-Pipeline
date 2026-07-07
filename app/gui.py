import gradio as gr
from app import retrieval, generation
from app.config import *
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    raise ValueError("API Key missing! Please set TOGETHER_API_KEY in your .env file.")

embedder = SentenceTransformer(EMBEDDING_MODEL)
retriever = retrieval.VectorRetriever(dim=384)
retriever.load("../index/")


def ask_rag(query, top_k=5):
    if not query.strip():
        return "Please enter a question.", ""

    try:
        query_embedding = embedder.encode([query])[0]
        chunks = retriever.search(query_embedding, top_k)
        prompt = generation.build_prompt(query, chunks)
        answer = generation.query_llm(prompt)

        references_text = "\n\n".join(
            f"[{c['metadata']['title']} - chunk {c['metadata']['chunk_id']}] {c['text'][:150]}..."
            for c in chunks
        )

        return answer, references_text

    except Exception as e:
        return f"Error: {e}", ""


demo = gr.Interface(
    fn=ask_rag,
    inputs=[
        gr.Textbox(label="Ask a Question", placeholder="e.g., What is the climate like in France?"),
        gr.Slider(1, 10, value=5, step=1, label="Top-K Chunks")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Top Retrieved Chunks")
    ],
    title="RAG Pipeline",
    description="Retrieval-Augmented Generation over Britannica data using Deepseek R1 and FAISS."
)

if __name__ == "__main__":
    demo.launch()
