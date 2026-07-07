from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app import data_utils, retrieval, generation
from app.config import *
from sentence_transformers import SentenceTransformer
import os

app = FastAPI(title="RAG API")

embedder = SentenceTransformer(EMBEDDING_MODEL)
retriever = retrieval.VectorRetriever(dim=384)
retriever.load("../index/")


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    filter_title: str | None = None

class ChunkMetadata(BaseModel):
    title: str
    chunk_id: int

class RetrievedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class RetrieveResponse(BaseModel):
    chunks: List[RetrievedChunk]

class GenerateResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(req: QueryRequest):
    try:
        query_embedding = embedder.encode([req.query])[0]
        chunks = retriever.search(query_embedding, req.top_k, req.filter_title)

        response_chunks = [
            RetrievedChunk(
                text=chunk["text"],
                metadata=ChunkMetadata(
                    title=chunk["metadata"]["title"],
                    chunk_id=chunk["metadata"]["chunk_id"]
                )
            ) for chunk in chunks
        ]
        return RetrieveResponse(chunks=response_chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
def generate_answer(req: QueryRequest):
    try:
        query_embedding = embedder.encode([req.query])[0]
        chunks = retriever.search(query_embedding, req.top_k, req.filter_title)
        prompt = generation.build_prompt(req.query, chunks)
        answer = generation.query_llm(prompt)

        response_chunks = [
            RetrievedChunk(
                text=chunk["text"],
                metadata=ChunkMetadata(
                    title=chunk["metadata"]["title"],
                    chunk_id=chunk["metadata"]["chunk_id"]
                )
            ) for chunk in chunks
        ]
        return GenerateResponse(query=req.query, answer=answer, retrieved_chunks=response_chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")