"""
FastAPI backend — exposes the RAG pipeline as an HTTP API.
Run: uvicorn src.api:app --reload --port 8000
"""

import sys
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from logger import get_logger
from rate_limiter import limiter, RateLimitExceeded

load_dotenv()

log = get_logger()

_chain = None
_retriever = None
PIPELINE_MODE = "rerank"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever
    log.info(f"Starting up — loading pipeline: {PIPELINE_MODE}", extra={"event": "startup", "pipeline": PIPELINE_MODE})
    from chain import build_chain
    _chain, _retriever = build_chain(mode=PIPELINE_MODE)
    log.info("RAG pipeline ready", extra={"event": "startup_complete", "pipeline": PIPELINE_MODE})
    yield
    log.info("Shutting down", extra={"event": "shutdown"})


app = FastAPI(title="Fischer Surgery RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    pipeline: Optional[str] = None


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
    status = limiter.status()
    log.debug("Health check", extra={"event": "health_check", **status})
    return {"status": "ok", "pipeline": PIPELINE_MODE, "rate_limit": status}


@app.get("/rate-limit")
def rate_limit_status():
    return limiter.status()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Rate limit check — raises 429 if exceeded
    try:
        limiter.check_and_record()
    except RateLimitExceeded as e:
        log.warning(
            f"Request blocked: {e.reason}",
            extra={"event": "request_blocked", "question_preview": question[:60]},
        )
        raise HTTPException(status_code=429, detail=e.reason)

    log.info(
        f"Query received: {question[:80]}...",
        extra={
            "event": "query_start",
            "question_preview": question[:80],
            "pipeline": PIPELINE_MODE,
            **{f"rl_{k}": v for k, v in limiter.status().items()},
        },
    )

    # Retrieval
    t0 = time.perf_counter()
    try:
        docs = _retriever.invoke(question)
    except Exception as e:
        log.error(f"Retrieval failed: {e}", extra={"event": "retrieval_error"}, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval failed")
    t_retrieval = round(time.perf_counter() - t0, 3)

    # LLM generation
    t1 = time.perf_counter()
    try:
        answer = _chain.invoke(question)
    except Exception as e:
        log.error(f"LLM generation failed: {e}", extra={"event": "llm_error"}, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed")
    t_llm = round(time.perf_counter() - t1, 3)
    t_total = round(t_retrieval + t_llm, 3)

    chunks = [
        ChunkResponse(
            page=doc.metadata.get("page", "?"),
            source=doc.metadata.get("source", "Fischer's Mastery of Surgery"),
            content=doc.page_content.strip(),
        )
        for doc in docs
    ]
    pages = sorted(set(c.page for c in chunks))

    log.info(
        f"Query complete in {t_total}s",
        extra={
            "event": "query_complete",
            "question_preview": question[:80],
            "pipeline": PIPELINE_MODE,
            "latency_total_s": t_total,
            "latency_retrieval_s": t_retrieval,
            "latency_llm_s": t_llm,
            "chunks_returned": len(chunks),
            "pages": pages,
        },
    )

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        pages=pages,
        pipeline=PIPELINE_MODE,
        latency_retrieval_s=t_retrieval,
        latency_llm_s=t_llm,
        latency_total_s=t_total,
    )
