import os, re
from bs4 import BeautifulSoup
import json
from sentence_transformers import SentenceTransformer


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def process_and_chunk_file(filepath, title):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    text = normalize_text(raw)
    chunks = chunk_text(text)
    return [{"text": chunk, "metadata": {"title": title, "chunk_id": i}} for i, chunk in enumerate(chunks)]

def embed_chunks(chunks, model_name):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings, chunks