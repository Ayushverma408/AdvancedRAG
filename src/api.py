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
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

from logger import get_logger
from rate_limiter import limiter, RateLimitExceeded
from books import medical_books, get_book_by_collection
from image_index import lookup_images

load_dotenv()

log = get_logger()

PIPELINE_MODE = "multi-book-hyde"

# Approximate cost constants (GPT-4o pricing)
GPT4O_INPUT_PER_TOKEN  = 2.50  / 1_000_000
GPT4O_OUTPUT_PER_TOKEN = 10.00 / 1_000_000
EMBED_PER_TOKEN        = 0.02  / 1_000_000
AVG_TOKENS_PER_CHAR    = 0.25

# Single multi-book pipeline — built once on first request
_multi_retriever = None
_multi_generator = None
_pipeline_lock   = threading.Lock()


def _estimate_cost(question: str, context_chunks: list[dict], answer: str) -> float:
    ctx_chars  = sum(len(c.get("content", "")) for c in context_chunks)
    input_tok  = 200 + len(question) * AVG_TOKENS_PER_CHAR + ctx_chars * AVG_TOKENS_PER_CHAR
    output_tok = len(answer) * AVG_TOKENS_PER_CHAR
    embed_tok  = len(question) * AVG_TOKENS_PER_CHAR
    return round(
        input_tok * GPT4O_INPUT_PER_TOKEN
        + output_tok * GPT4O_OUTPUT_PER_TOKEN
        + embed_tok * EMBED_PER_TOKEN,
        6,
    )


def get_or_build_multi_pipeline():
    global _multi_retriever, _multi_generator
    if _multi_retriever is None:
        with _pipeline_lock:
            if _multi_retriever is None:
                log.info("Building multi-book pipeline", extra={"event": "pipeline_build"})
                from retriever_multi import build_multi_book_retriever
                from chain import build_multi_book_generator
                books = medical_books()
                _multi_retriever = build_multi_book_retriever(books)
                _multi_generator = build_multi_book_generator()
                log.info("Multi-book pipeline ready", extra={"event": "pipeline_ready",
                         "books": [b["collection"] for b in books]})
    return _multi_retriever, _multi_generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "API starting up",
        extra={"event": "startup", "pipeline": PIPELINE_MODE,
               "books": [b["collection"] for b in medical_books()]},
    )
    # Warm up pipeline at startup so first user query isn't slow
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_or_build_multi_pipeline)
    yield
    log.info("API shutting down", extra={"event": "shutdown"})


app = FastAPI(title="Medical RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    book_key: Optional[str] = None   # ignored — multi-book mode searches all books
    pipeline: Optional[str] = None   # ignored — pipeline is fixed to multi-book-hyde
    free_mode: bool = False           # skip RAG; answer directly from GPT-4o knowledge


class ChunkResponse(BaseModel):
    page: int | str
    source: str        # book display name
    collection: str    # ChromaDB collection name (used for page preview)
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
    log.debug("Health check", extra={"event": "health_check", **{f"rl_{k}": v for k, v in status.items()}})
    return {"status": "ok", "pipeline": PIPELINE_MODE, "rate_limit": status}


@app.get("/books")
def list_books():
    return medical_books()


@app.get("/rate-limit")
def rate_limit_status():
    return limiter.status()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _images_for_docs(docs: list) -> list[dict]:
    """Look up images per collection from retrieved docs."""
    from collections import defaultdict
    pages_by_col: dict[str, set] = defaultdict(set)
    for doc in docs:
        col  = doc.metadata.get("collection", "")
        page = doc.metadata.get("page")
        if col and page is not None:
            pages_by_col[col].add(page)

    images = []
    for col, pages in pages_by_col.items():
        for e in lookup_images(sorted(pages), col):
            images.append({"path": os.path.abspath(e["path"]), "caption": e["caption"]})
    return images


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        limiter.check_and_record()
    except RateLimitExceeded as e:
        log.warning("Request rate-limited",
                    extra={"event": "rate_limit_blocked", "question": question,
                           "reason": e.reason, **{f"rl_{k}": v for k, v in limiter.status().items()}})
        raise HTTPException(status_code=429, detail=e.reason)

    request_id = str(uuid.uuid4())[:8]
    rl = limiter.status()

    log.info("Query received",
             extra={"event": "query_start", "request_id": request_id,
                    "pipeline": PIPELINE_MODE, "question": question,
                    "question_len": len(question),
                    "rl_daily_count": rl["daily_count"],
                    "rl_daily_remaining": rl["daily_remaining"],
                    "rl_minute_count": rl["minute_count"]})

    async def event_gen():
        loop = asyncio.get_event_loop()

        # ── Free mode: bypass RAG, answer directly from GPT-4o ───────────
        if req.free_mode:
            yield _sse({"phase": "generating"})
            t0 = time.perf_counter()
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                FREE_PROMPT = (
                    "You are an expert medical assistant with broad clinical knowledge. "
                    "Answer the following question clearly and accurately. "
                    "If relevant, mention uncertainty or recommend consulting primary sources.\n\n"
                    f"Question: {question}"
                )
                answer = await loop.run_in_executor(
                    None, lambda: llm.invoke(FREE_PROMPT).content
                )
            except Exception:
                yield _sse({"phase": "error", "msg": "Generation failed"})
                return
            t_total = round(time.perf_counter() - t0, 3)
            yield _sse({
                "phase": "done", "answer": answer, "chunks": [], "pages": [],
                "images": [], "pipeline": "free", "latency_retrieval_s": 0,
                "latency_llm_s": t_total, "latency_total_s": t_total,
            })
            return

        # ── Phase 1: Retrieval ────────────────────────────────────────────
        yield _sse({"phase": "retrieving", "pipeline": PIPELINE_MODE})

        t0 = time.perf_counter()
        try:
            retriever, _ = get_or_build_multi_pipeline()
            docs = await loop.run_in_executor(None, retriever.invoke, question)
        except Exception:
            log.error("Retrieval failed",
                      extra={"event": "retrieval_error", "request_id": request_id,
                             "question": question}, exc_info=True)
            yield _sse({"phase": "error", "msg": "Retrieval failed"})
            return
        t_retrieval = round(time.perf_counter() - t0, 3)

        chunks = [
            {
                "page":       doc.metadata.get("page", "?"),
                "source":     doc.metadata.get("source", ""),
                "collection": doc.metadata.get("collection", ""),
                "content":    doc.page_content.strip(),
            }
            for doc in docs
        ]
        pages  = sorted(set(c["page"] for c in chunks))
        images = _images_for_docs(docs)

        log.info(f"Retrieval done — {len(chunks)} chunks in {t_retrieval}s",
                 extra={"event": "retrieval_complete", "request_id": request_id,
                        "pipeline": PIPELINE_MODE, "question": question,
                        "chunks_returned": len(chunks), "pages": pages,
                        "latency_retrieval_s": t_retrieval})
        yield _sse({"phase": "retrieved", "chunks": chunks, "latency_retrieval_s": t_retrieval})

        # ── Phase 2: Generation ───────────────────────────────────────────
        yield _sse({"phase": "generating"})

        from chain import format_docs
        context   = format_docs(docs)
        _, generator = get_or_build_multi_pipeline()

        t1 = time.perf_counter()
        try:
            answer = await loop.run_in_executor(
                None, generator.invoke, {"context": context, "question": question}
            )
        except Exception:
            log.error("Generation failed",
                      extra={"event": "llm_error", "request_id": request_id,
                             "question": question}, exc_info=True)
            yield _sse({"phase": "error", "msg": "Generation failed"})
            return
        t_llm   = round(time.perf_counter() - t1, 3)
        t_total = round(t_retrieval + t_llm, 3)
        cost    = _estimate_cost(question, chunks, answer)

        log.info("Query complete",
                 extra={"event": "query_complete", "request_id": request_id,
                        "pipeline": PIPELINE_MODE, "question": question,
                        "answer_preview": answer[:200].replace("\n", " "),
                        "answer_len": len(answer), "chunks_returned": len(chunks),
                        "pages": pages, "latency_retrieval_s": t_retrieval,
                        "latency_llm_s": t_llm, "latency_total_s": t_total,
                        "cost_usd": cost, "rl_daily_count": rl["daily_count"],
                        "rl_daily_remaining": rl["daily_remaining"]})

        yield _sse({
            "phase":               "done",
            "answer":              answer,
            "chunks":              chunks,
            "pages":               pages,
            "images":              images,
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
        limiter.check_and_record()
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=e.reason)

    request_id = str(uuid.uuid4())[:8]

    try:
        retriever, generator = get_or_build_multi_pipeline()
    except Exception:
        log.error("Pipeline build failed", extra={"event": "pipeline_error",
                  "request_id": request_id}, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load pipeline")

    t0 = time.perf_counter()
    try:
        docs = retriever.invoke(question)
    except Exception:
        raise HTTPException(status_code=500, detail="Retrieval failed")
    t_retrieval = round(time.perf_counter() - t0, 3)

    from chain import format_docs
    t1 = time.perf_counter()
    try:
        answer = generator.invoke({"context": format_docs(docs), "question": question})
    except Exception:
        raise HTTPException(status_code=500, detail="Generation failed")
    t_llm   = round(time.perf_counter() - t1, 3)
    t_total = round(t_retrieval + t_llm, 3)

    chunks = [
        ChunkResponse(
            page=doc.metadata.get("page", "?"),
            source=doc.metadata.get("source", ""),
            collection=doc.metadata.get("collection", ""),
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
        latency_retrieval_s=t_retrieval,
        latency_llm_s=t_llm,
        latency_total_s=t_total,
    )


# ── PDF page preview ──────────────────────────────────────────────────────────

@app.get("/page/{collection}/{page_num}")
def get_page_image(collection: str, page_num: int, dpi: int = 150):
    """
    Render a single PDF page as a PNG and return it.
    page_num is 1-indexed (matches the page metadata stored at ingest time).
    Used by the Chainlit UI for the page-preview feature.
    """
    try:
        book = get_book_by_collection(collection)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    pdf_path = book["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    import fitz
    doc = fitz.open(pdf_path)
    total = len(doc)

    # page_num is 1-indexed; clamp to valid range
    idx = max(0, min(page_num - 1, total - 1))

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[idx].get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()

    return Response(content=png_bytes, media_type="image/png")
