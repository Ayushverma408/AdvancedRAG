"""
FastAPI backend — exposes the RAG pipeline as an HTTP API.
Run: uvicorn src.api:app --reload --port 8000
"""

import sys
import os
import time
import asyncio
import json
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from logger import get_logger
from rate_limiter import limiter, RateLimitExceeded
from books import get_book, all_books

load_dotenv()

log = get_logger()

PIPELINE_MODE = "hyde"

# Approximate cost constants (GPT-4o pricing)
GPT4O_INPUT_PER_TOKEN  = 2.50  / 1_000_000
GPT4O_OUTPUT_PER_TOKEN = 10.00 / 1_000_000
EMBED_PER_TOKEN        = 0.02  / 1_000_000
AVG_TOKENS_PER_CHAR    = 0.25

# Lazy chain cache: book_key -> (chain, retriever)
_chains: dict[str, tuple] = {}
_generators: dict[str, object] = {}
_chain_lock = threading.Lock()


def _estimate_cost(question: str, context_chunks: list[dict], answer: str) -> float:
    """Rough cost estimate for a single query in USD."""
    ctx_chars   = sum(len(c.get("content", "")) for c in context_chunks)
    input_tok   = (200 + len(question) * AVG_TOKENS_PER_CHAR + ctx_chars * AVG_TOKENS_PER_CHAR)
    output_tok  = len(answer) * AVG_TOKENS_PER_CHAR
    embed_tok   = len(question) * AVG_TOKENS_PER_CHAR
    return round(
        input_tok * GPT4O_INPUT_PER_TOKEN
        + output_tok * GPT4O_OUTPUT_PER_TOKEN
        + embed_tok * EMBED_PER_TOKEN,
        6,
    )


def get_or_build_chain(book_key: str) -> tuple:
    if book_key not in _chains:
        with _chain_lock:
            if book_key not in _chains:
                book = get_book(book_key)
                log.info(
                    f"Building chain for '{book_key}'",
                    extra={"event": "chain_build", "book_key": book_key, "pipeline": PIPELINE_MODE},
                )
                from chain import build_chain
                chain, retriever = build_chain(
                    mode=PIPELINE_MODE,
                    collection=book["collection"],
                    book_name=book["display_name"],
                )
                _chains[book_key] = (chain, retriever)
                log.info(
                    f"Chain ready for '{book_key}'",
                    extra={"event": "chain_ready", "book_key": book_key, "pipeline": PIPELINE_MODE},
                )
    return _chains[book_key]


def get_or_build_generator(book_key: str):
    if book_key not in _generators:
        with _chain_lock:
            if book_key not in _generators:
                book = get_book(book_key)
                from chain import build_generator
                _generators[book_key] = build_generator(
                    collection=book["collection"],
                    book_name=book["display_name"],
                )
    return _generators[book_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "API starting up",
        extra={"event": "startup", "pipeline": PIPELINE_MODE, "books_registered": len(all_books())},
    )
    yield
    log.info("API shutting down", extra={"event": "shutdown"})


app = FastAPI(title="Book RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    book_key: str = "fischer_surgery"
    pipeline: Optional[str] = None


class ChunkResponse(BaseModel):
    page: int | str
    source: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkResponse]
    pages: list[int | str]
    book_key: str
    pipeline: str
    latency_retrieval_s: float
    latency_llm_s: float
    latency_total_s: float


@app.get("/health")
def health():
    status = limiter.status()
    log.debug("Health check", extra={"event": "health_check", **{f"rl_{k}": v for k, v in status.items()}})
    return {"status": "ok", "pipeline": PIPELINE_MODE, "rate_limit": status}


@app.get("/books")
def list_books():
    return all_books()


@app.get("/rate-limit")
def rate_limit_status():
    return limiter.status()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        book = get_book(req.book_key)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        limiter.check_and_record()
    except RateLimitExceeded as e:
        log.warning(
            "Request rate-limited",
            extra={
                "event": "rate_limit_blocked",
                "question": question,
                "book_key": req.book_key,
                "reason": e.reason,
                **{f"rl_{k}": v for k, v in limiter.status().items()},
            },
        )
        raise HTTPException(status_code=429, detail=e.reason)

    request_id = str(uuid.uuid4())[:8]
    rl = limiter.status()

    log.info(
        f"Query received",
        extra={
            "event":            "query_start",
            "request_id":       request_id,
            "book_key":         req.book_key,
            "book_name":        book["display_name"],
            "pipeline":         PIPELINE_MODE,
            "question":         question,
            "question_len":     len(question),
            "rl_daily_count":   rl["daily_count"],
            "rl_daily_remaining": rl["daily_remaining"],
            "rl_minute_count":  rl["minute_count"],
        },
    )

    async def event_gen():
        loop = asyncio.get_event_loop()

        # ── Phase 1: Retrieval ────────────────────────────────────────────
        yield _sse({"phase": "retrieving", "pipeline": PIPELINE_MODE})

        t0 = time.perf_counter()
        try:
            _, retriever = get_or_build_chain(req.book_key)
            docs = await loop.run_in_executor(None, retriever.invoke, question)
        except Exception as e:
            log.error(
                "Retrieval failed",
                extra={"event": "retrieval_error", "request_id": request_id,
                       "book_key": req.book_key, "question": question},
                exc_info=True,
            )
            yield _sse({"phase": "error", "msg": "Retrieval failed"})
            return
        t_retrieval = round(time.perf_counter() - t0, 3)

        chunks = [
            {
                "page":    doc.metadata.get("page", "?"),
                "source":  doc.metadata.get("source", book["display_name"]),
                "content": doc.page_content.strip(),
            }
            for doc in docs
        ]
        pages = sorted(set(c["page"] for c in chunks))

        log.info(
            f"Retrieval done — {len(chunks)} chunks in {t_retrieval}s",
            extra={
                "event":             "retrieval_complete",
                "request_id":        request_id,
                "book_key":          req.book_key,
                "pipeline":          PIPELINE_MODE,
                "question":          question,
                "chunks_returned":   len(chunks),
                "pages":             pages,
                "latency_retrieval_s": t_retrieval,
            },
        )
        yield _sse({"phase": "retrieved", "chunks": chunks, "latency_retrieval_s": t_retrieval})

        # ── Phase 2: Generation ───────────────────────────────────────────
        yield _sse({"phase": "generating"})

        from chain import format_docs
        context   = format_docs(docs)
        generator = get_or_build_generator(req.book_key)

        t1 = time.perf_counter()
        try:
            answer = await loop.run_in_executor(
                None, generator.invoke, {"context": context, "question": question}
            )
        except Exception as e:
            log.error(
                "Generation failed",
                extra={"event": "llm_error", "request_id": request_id,
                       "book_key": req.book_key, "question": question},
                exc_info=True,
            )
            yield _sse({"phase": "error", "msg": "Generation failed"})
            return
        t_llm   = round(time.perf_counter() - t1, 3)
        t_total = round(t_retrieval + t_llm, 3)
        cost    = _estimate_cost(question, chunks, answer)

        log.info(
            f"Query complete",
            extra={
                "event":               "query_complete",
                "request_id":          request_id,
                "book_key":            req.book_key,
                "book_name":           book["display_name"],
                "pipeline":            PIPELINE_MODE,
                "question":            question,
                "answer_preview":      answer[:200].replace("\n", " "),
                "answer_len":          len(answer),
                "chunks_returned":     len(chunks),
                "pages":               pages,
                "latency_retrieval_s": t_retrieval,
                "latency_llm_s":       t_llm,
                "latency_total_s":     t_total,
                "cost_usd":            cost,
                "rl_daily_count":      rl["daily_count"],
                "rl_daily_remaining":  rl["daily_remaining"],
            },
        )

        yield _sse({
            "phase":               "done",
            "answer":              answer,
            "chunks":              chunks,
            "pages":               pages,
            "book_key":            req.book_key,
            "pipeline":            PIPELINE_MODE,
            "latency_retrieval_s": t_retrieval,
            "latency_llm_s":       t_llm,
            "latency_total_s":     t_total,
        })

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        book = get_book(req.book_key)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        limiter.check_and_record()
    except RateLimitExceeded as e:
        log.warning(
            "Request rate-limited",
            extra={"event": "rate_limit_blocked", "question": question,
                   "book_key": req.book_key, "reason": e.reason},
        )
        raise HTTPException(status_code=429, detail=e.reason)

    request_id = str(uuid.uuid4())[:8]
    rl = limiter.status()

    log.info(
        "Query received",
        extra={
            "event":            "query_start",
            "request_id":       request_id,
            "book_key":         req.book_key,
            "book_name":        book["display_name"],
            "pipeline":         PIPELINE_MODE,
            "question":         question,
            "question_len":     len(question),
            "rl_daily_count":   rl["daily_count"],
            "rl_daily_remaining": rl["daily_remaining"],
            "rl_minute_count":  rl["minute_count"],
        },
    )

    try:
        chain, retriever = get_or_build_chain(req.book_key)
    except Exception as e:
        log.error("Chain build failed", extra={"event": "chain_error", "request_id": request_id}, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load book pipeline")

    t0 = time.perf_counter()
    try:
        docs = retriever.invoke(question)
    except Exception as e:
        log.error("Retrieval failed", extra={"event": "retrieval_error", "request_id": request_id,
                  "question": question}, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval failed")
    t_retrieval = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
    try:
        answer = chain.invoke(question)
    except Exception as e:
        log.error("LLM generation failed", extra={"event": "llm_error", "request_id": request_id,
                  "question": question}, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed")
    t_llm   = round(time.perf_counter() - t1, 3)
    t_total = round(t_retrieval + t_llm, 3)

    chunks = [
        ChunkResponse(
            page=doc.metadata.get("page", "?"),
            source=doc.metadata.get("source", book["display_name"]),
            content=doc.page_content.strip(),
        )
        for doc in docs
    ]
    pages = sorted(set(c.page for c in chunks))
    cost  = _estimate_cost(question, [{"content": c.content} for c in chunks], answer)

    log.info(
        "Query complete",
        extra={
            "event":               "query_complete",
            "request_id":          request_id,
            "book_key":            req.book_key,
            "book_name":           book["display_name"],
            "pipeline":            PIPELINE_MODE,
            "question":            question,
            "answer_preview":      answer[:200].replace("\n", " "),
            "answer_len":          len(answer),
            "chunks_returned":     len(chunks),
            "pages":               pages,
            "latency_retrieval_s": t_retrieval,
            "latency_llm_s":       t_llm,
            "latency_total_s":     t_total,
            "cost_usd":            cost,
        },
    )

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        pages=pages,
        book_key=req.book_key,
        pipeline=PIPELINE_MODE,
        latency_retrieval_s=t_retrieval,
        latency_llm_s=t_llm,
        latency_total_s=t_total,
    )
