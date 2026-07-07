import numpy as np
import faiss
import pickle
import os

class VectorRetriever:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.embeddings = []
        self.chunks = []

    def add_embeddings(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype('float32'))
        self.embeddings.extend(embeddings)
        self.chunks.extend(chunks)

        # --- REPLACE THESE METHODS IN retrieval.py ---

        def save(self, path):
            faiss.write_index(self.index, os.path.join(path, "faiss.index"))
            # Save both chunks and raw embeddings together
            with open(os.path.join(path, "rag_data.pkl"), "wb") as f:
                pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

        def load(self, path):
            self.index = faiss.read_index(os.path.join(path, "faiss.index"))
            with open(os.path.join(path, "rag_data.pkl"), "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.embeddings = data["embeddings"]

        def search(self, query_embedding, top_k, filter_title=None):
            if filter_title:
                filtered_chunks = [
                    (i, c) for i, c in enumerate(self.chunks)
                    if c["metadata"]["title"].lower() == filter_title.lower()
                ]
                if not filtered_chunks:
                    return []

                filtered_embeddings = [self.embeddings[i] for i, _ in filtered_chunks]
                faiss_index = faiss.IndexFlatL2(len(query_embedding))
                faiss_index.add(np.array(filtered_embeddings).astype('float32'))

                D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), top_k)
                return [filtered_chunks[i][1] for i in I[0] if i != -1]
            else:
                D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
                # Prevent out-of-bounds mapping if FAISS returns -1 padding
                return [self.chunks[i] for i in I[0] if i != -1]