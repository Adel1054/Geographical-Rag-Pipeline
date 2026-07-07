import os
from app import data_utils, retrieval
from app.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

from sentence_transformers import SentenceTransformer
import glob

DATA_DIR = "../data/raw"
INDEX_DIR = "../index/"
os.makedirs(INDEX_DIR, exist_ok=True)

all_chunks = []
for filepath in glob.glob(f"{DATA_DIR}/*.txt"):
    title = os.path.basename(filepath).replace(".txt", "")
    chunks = data_utils.process_and_chunk_file(filepath, title)
    all_chunks.extend(chunks)

model = SentenceTransformer(EMBEDDING_MODEL)
embeddings, chunks = data_utils.embed_chunks(all_chunks, EMBEDDING_MODEL)

retriever = retrieval.VectorRetriever(dim=len(embeddings[0]))
retriever.add_embeddings(embeddings, chunks)
retriever.save(INDEX_DIR)

print(f"FAISS index and chunks saved to {INDEX_DIR}")