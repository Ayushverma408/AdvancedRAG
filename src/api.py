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
    # Warm up pipeline + cross-encoder at startup so first query pays no cold-start cost
    import asyncio
    from retriever_rerank import get_cross_encoder
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_or_build_multi_pipeline)
    await loop.run_in_executor(None, get_cross_encoder)
    log.info("Cross-encoder warmed up", extra={"event": "cross_encoder_ready"})
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
    use_hyde: bool = True             # False = fast mode: skip HyDE, embed query only (~4s faster)
    profile_prompt: str = ""          # optional user profile context injected into the system prompt
    answer_depth: str = "balanced"           # "concise" | "balanced" | "comprehensive"
    answer_tone: str = "teaching"            # "textbook" | "teaching"
    answer_restrictiveness: str = "guided"   # "strict" | "guided" | "open"


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
                profile_section = f"\nUser profile:\n{req.profile_prompt.strip()}\n" if req.profile_prompt.strip() else ""
                FREE_PROMPT = (
                    "You are ScrubRef, a surgical reference assistant with broad surgical and medical knowledge. "
                    "Answer the following question drawing freely on your full knowledge."
                    f"{profile_section}\n\n"
                    f"Question: {question}"
                )
                answer = await loop.run_in_executor(
                    None, lambda: llm.invoke(FREE_PROMPT).content
                )
            except Exception:
                log.error("Free mode generation failed",
                          extra={"event": "free_mode_error", "question": question,
                                 "request_id": request_id}, exc_info=True)
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
        use_hyde      = req.use_hyde
        pipeline_used = "multi-book-hyde" if use_hyde else "multi-book-fast"
        yield _sse({"phase": "retrieving", "pipeline": pipeline_used})

        t0 = time.perf_counter()
        try:
            retriever, _ = get_or_build_multi_pipeline()

            # Thread-safe queue: retriever calls status_cb from a worker thread,
            # we drain it here in the async generator to yield real-time sub-phase events.
            status_q: asyncio.Queue = asyncio.Queue()

            def _status_cb(label: str) -> None:
                loop.call_soon_threadsafe(status_q.put_nowait, label)

            retrieve_future = loop.run_in_executor(
                None,
                lambda: retriever.retrieve(question, use_hyde=use_hyde, status_cb=_status_cb),
            )

            # Poll for sub-phase events while the retriever runs in its thread.
            # Avoid asyncio.wait_for(queue.get()) — on Python 3.13 the timeout/put
            # race can silently drop items. Plain sleep + get_nowait is simpler.
            while not retrieve_future.done():
                await asyncio.sleep(0.05)
                while not status_q.empty():
                    yield _sse({"phase": "sub_phase", "label": status_q.get_nowait()})

            # Let any in-flight call_soon_threadsafe callbacks land, then drain
            await asyncio.sleep(0)
            while not status_q.empty():
                yield _sse({"phase": "sub_phase", "label": status_q.get_nowait()})

            docs, ret_timings = await retrieve_future

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

        from chain import format_docs, build_answer_system_prompt
        context   = format_docs(docs)
        _, generator = get_or_build_multi_pipeline()

        system_prompt = build_answer_system_prompt(
            depth=req.answer_depth,
            tone=req.answer_tone,
            restrictiveness=req.answer_restrictiveness,
            profile_prompt=req.profile_prompt,
            context=context,
        )

        t1 = time.perf_counter()
        answer_tokens = []
        try:
            async for token in generator.astream({"system_prompt": system_prompt, "question": question}):
                answer_tokens.append(token)
                yield _sse({"phase": "token", "delta": token})
            answer = "".join(answer_tokens)
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
            "pipeline":            pipeline_used,
            "latency_retrieval_s": t_retrieval,
            "latency_llm_s":       t_llm,
            "latency_total_s":     t_total,
            # sub-phase breakdown from retriever
            "latency_hyde_s":      ret_timings.get("latency_hyde_s", 0),
            "latency_embed_s":     ret_timings.get("latency_embed_s", 0),
            "latency_search_s":    ret_timings.get("latency_retrieval_s", 0),
            "latency_rerank_s":    ret_timings.get("latency_rerank_s", 0),
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
        log.error("Retrieval failed", extra={"event": "retrieval_error",
                  "request_id": request_id, "question": question}, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval failed")
    t_retrieval = round(time.perf_counter() - t0, 3)

    from chain import format_docs
    t1 = time.perf_counter()
    try:
        answer = generator.invoke({"context": format_docs(docs), "question": question})
    except Exception:
        log.error("Generation failed", extra={"event": "llm_error",
                  "request_id": request_id, "question": question}, exc_info=True)
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
def get_page_image(collection: str, page_num: int, dpi: int = 150, highlight: str = ""):
    """
    Render a single PDF page as a PNG and return it.
    page_num is 1-indexed (matches the page metadata stored at ingest time).
    highlight: optional chunk text snippet — matched on the page and highlighted in yellow.
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
    page = doc[idx]

    # Yellow highlight — find start + end of chunk on the page, highlight all lines between
    if highlight:
        import re
        normalized = re.sub(r"\s+", " ", highlight).strip()

        # Locate where the chunk starts on the page
        start_rects = None
        for length in [60, 45, 35, 25]:
            rects = page.search_for(normalized[:length].strip())
            if rects:
                start_rects = rects
                break

        # Locate where the chunk ends (only worth searching if chunk is long)
        end_rects = None
        if start_rects and len(normalized) > 80:
            for length in [60, 45, 35, 25]:
                rects = page.search_for(normalized[-length:].strip())
                if rects:
                    end_rects = rects
                    break

        if start_rects:
            y_top = start_rects[0].y0
            y_bot = end_rects[-1].y1 if end_rects else start_rects[-1].y1

            # Highlight every text line whose bbox falls within [y_top, y_bot]
            for block in page.get_text("dict", flags=0).get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    lr = fitz.Rect(line["bbox"])
                    if lr.y1 >= y_top - 2 and lr.y0 <= y_bot + 2:
                        annot = page.add_highlight_annot(lr)
                        annot.set_colors(stroke=[1, 1, 0])
                        annot.update()

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()

    return Response(content=png_bytes, media_type="image/png")
