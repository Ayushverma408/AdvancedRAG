"""
FastAPI backend — exposes the RAG pipeline as an HTTP API.
Run: uvicorn src.api:app --reload --port 8000
"""

import sys
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

# Ensure src/ is on path regardless of where uvicorn is invoked from
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# RAG chain is loaded once at startup and reused
_chain = None
_retriever = None
PIPELINE_MODE = "rerank"  # change to switch pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever
    print(f"Loading RAG pipeline: {PIPELINE_MODE}")
    from chain import build_chain
    _chain, _retriever = build_chain(mode=PIPELINE_MODE)
    print("RAG pipeline ready")
    yield


app = FastAPI(title="Fischer Surgery RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    pipeline: Optional[str] = None  # future: allow per-request pipeline switch


class ChunkResponse(BaseModel):
    page: int | str
    source: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkResponse]
    pages: list[int | str]
    pipeline: str
    latency_retrieval_s: float
    latency_llm_s: float
    latency_total_s: float


@app.get("/health")
def health():
    return {"status": "ok", "pipeline": PIPELINE_MODE}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieval
    t0 = time.perf_counter()
    docs = _retriever.invoke(question)
    t_retrieval = time.perf_counter() - t0

    # LLM generation
    t1 = time.perf_counter()
    answer = _chain.invoke(question)
    t_llm = time.perf_counter() - t1

    chunks = [
        ChunkResponse(
            page=doc.metadata.get("page", "?"),
            source=doc.metadata.get("source", "Fischer's Mastery of Surgery"),
            content=doc.page_content.strip(),
        )
        for doc in docs
    ]

    pages = sorted(set(c.page for c in chunks))

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        pages=pages,
        pipeline=PIPELINE_MODE,
        latency_retrieval_s=round(t_retrieval, 3),
        latency_llm_s=round(t_llm, 3),
        latency_total_s=round(t_retrieval + t_llm, 3),
    )
