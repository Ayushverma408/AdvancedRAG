"""
Multi-Book HyDE Retriever.

Searches across all ingested medical book collections simultaneously:
1. Generate ONE hypothetical document (HyDE) for the query — shared across all books.
2. Embed query + hyp_doc ONCE (2 API calls total, not 8).
3. For each book collection: retrieve via dense(hyp_emb) + dense(query_emb) + BM25(query),
   merge per-book results via RRF.  All 4 books run in parallel.
4. Globally merge all per-book result lists via RRF.
5. Cross-encoder rerank the top-MAX_RERANK candidates and return top K.

Each returned Document has:
  doc.metadata["source"]     — book display name  (set at ingest time)
  doc.metadata["page"]       — page number        (set at ingest time)
  doc.metadata["collection"] — ChromaDB collection name (tagged here at retrieval time)

Key performance design:
  - HyDE + embedding API calls happen BEFORE threads start (sequential but minimal: 2 calls)
  - Threads do only local work: ChromaDB vector search (disk) + BM25 (CPU in-memory)
  - No OpenAI API calls inside threads → latency is max(4 books) not sum(8 API calls)
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from retriever_hybrid import load_all_chunks, reciprocal_rank_fusion
from retriever_rerank import get_cross_encoder

log = logging.getLogger("book_rag")

FETCH_K_PER_BOOK = 10      # candidates fetched per source (dense-hyp / dense-orig / bm25) per book
FINAL_K = 4                # final docs returned after global rerank
MAX_RERANK = 30            # cap global RRF pool before cross-encoder (keeps rerank fast)
RERANK_SCORE_THRESHOLD = -1.0  # cross-encoder logit floor — chunks below this are filtered as noise
MIN_FINAL_K = 2            # minimum chunks always returned even if all fall below threshold

HYDE_PROMPT = """\
You are an expert medical author. Given the question below, write a short \
passage (3–5 sentences) that a medical textbook would contain to directly answer it. \
Use precise clinical language and terminology. Do not say "the book says" — \
just write the passage itself as if it appeared in the textbook.

Question: {question}

Passage:"""


def _retrieve_one_book(
    br: dict,
    hyp_emb: list,
    query_emb: list,
    query: str,
    fetch_k: int,
) -> List[Document]:
    """
    Retrieve from a single book using 3 signals: dense(hyp_emb), dense(query_emb), BM25(query).
    Uses pre-computed embeddings — no OpenAI API calls here. Pure local I/O + CPU.
    Runs inside a thread pool — all books execute concurrently.
    """
    t0 = time.perf_counter()
    vectorstore = br["vectorstore"]
    bm25        = br["bm25"]
    collection  = br["collection"]

    bm25.k = fetch_k

    # similarity_search_by_vector skips embedding — uses pre-computed vector directly
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix=f"sig_{collection[:6]}") as sig_pool:
        f_hyp  = sig_pool.submit(vectorstore.similarity_search_by_vector, hyp_emb,   k=fetch_k)
        f_orig = sig_pool.submit(vectorstore.similarity_search_by_vector, query_emb, k=fetch_k)
        f_bm25 = sig_pool.submit(bm25.invoke, query)
        hyp_results  = f_hyp.result()
        orig_results = f_orig.result()
        bm25_results = f_bm25.result()

    t_book = time.perf_counter() - t0

    for doc in hyp_results + orig_results + bm25_results:
        doc.metadata.setdefault("collection", collection)

    merged = reciprocal_rank_fusion([hyp_results, orig_results, bm25_results])

    log.debug(
        f"Book retrieval done: {collection}",
        extra={
            "event": "book_retrieval_done",
            "collection": collection,
            "hyp_hits": len(hyp_results),
            "orig_hits": len(orig_results),
            "bm25_hits": len(bm25_results),
            "rrf_merged": len(merged),
            "latency_book_s": round(t_book, 3),
        },
    )
    return merged


class MultiBookHyDERetriever(BaseRetriever):
    """
    HyDE retriever that searches across multiple ChromaDB book collections,
    merges results globally via RRF, and reranks with a cross-encoder.

    Embeddings are computed once upfront; all book retrievals run in parallel
    using only pre-computed vectors — zero OpenAI API calls inside threads.
    """

    # Each entry: {"collection": str, "bm25": BM25Retriever, "vectorstore": Chroma}
    book_retrievers: List[Any]
    llm: Any
    embeddings: Any   # shared OpenAIEmbeddings instance for query + hyp_doc
    fetch_k: int = FETCH_K_PER_BOOK
    final_k: int = FINAL_K
    max_rerank: int = MAX_RERANK
    score_threshold: float = RERANK_SCORE_THRESHOLD
    min_final_k: int = MIN_FINAL_K

    class Config:
        arbitrary_types_allowed = True

    def _generate_hypothetical_doc(self, query: str) -> str:
        return self.llm.invoke(HYDE_PROMPT.format(question=query)).content.strip()

    def retrieve(self, query: str, use_hyde: bool = True) -> List[Document]:
        """
        Main retrieval entry point. Call this directly from the API.

        use_hyde=True  (default) — generates a hypothetical passage first, embeds both
                                   query + hyp_doc, uses both as dense signals. Best quality.
        use_hyde=False (fast)    — skips HyDE generation, embeds only the query.
                                   Saves ~4s per query, small quality drop.
        """
        t_start = time.perf_counter()

        # ── Step 1: HyDE (optional) + embed ───────────────────────────────────
        if use_hyde:
            t_hyde_0 = time.perf_counter()
            hyp_doc = self._generate_hypothetical_doc(query)
            t_hyde  = time.perf_counter() - t_hyde_0
            if t_hyde > 5.0:
                log.warning("HyDE generation slow", extra={
                    "event": "hyde_slow", "latency_hyde_s": round(t_hyde, 3), "query": query,
                })

            # Embed query + hyp_doc concurrently — 2 API calls in parallel
            t_embed_0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed") as ep:
                f_q = ep.submit(self.embeddings.embed_query, query)
                f_h = ep.submit(self.embeddings.embed_query, hyp_doc)
                query_emb = f_q.result()
                hyp_emb   = f_h.result()
            t_embed = time.perf_counter() - t_embed_0
            log.debug("HyDE + embed done", extra={
                "event": "hyde_embed_done", "hyp_doc_preview": hyp_doc[:120],
                "latency_hyde_s": round(t_hyde, 3), "latency_embed_s": round(t_embed, 3),
            })
        else:
            # Fast mode: embed query once, reuse for both dense signals
            t_hyde  = 0.0
            t_embed_0 = time.perf_counter()
            query_emb = self.embeddings.embed_query(query)
            hyp_emb   = query_emb
            t_embed   = time.perf_counter() - t_embed_0
            log.debug("Fast embed done (no HyDE)", extra={
                "event": "fast_embed_done", "latency_embed_s": round(t_embed, 3),
            })

        # ── Step 2: Retrieve from ALL books in parallel — pure local, no API ──
        t_retrieval_0 = time.perf_counter()
        all_per_book_results: List[List[Document]] = []

        with ThreadPoolExecutor(
            max_workers=len(self.book_retrievers),
            thread_name_prefix="book_ret",
        ) as pool:
            futures = {
                pool.submit(
                    _retrieve_one_book, br, hyp_emb, query_emb, query, self.fetch_k
                ): br["collection"]
                for br in self.book_retrievers
            }
            for future in as_completed(futures):
                collection = futures[future]
                try:
                    book_merged = future.result()
                    if book_merged:
                        all_per_book_results.append(book_merged)
                except Exception as exc:
                    log.warning(f"Book retrieval failed: {collection}", extra={
                        "event": "book_retrieval_error", "collection": collection, "error": str(exc),
                    }, exc_info=exc)

        t_retrieval = time.perf_counter() - t_retrieval_0
        log.debug("Parallel book retrieval done", extra={
            "event": "parallel_retrieval_done",
            "books_retrieved": len(all_per_book_results),
            "total_candidates": sum(len(r) for r in all_per_book_results),
            "latency_retrieval_s": round(t_retrieval, 3),
        })

        if not all_per_book_results:
            log.error("All book retrievals failed or returned empty",
                      extra={"event": "all_books_empty", "query": query})
            return []

        # ── Step 3: Global RRF ─────────────────────────────────────────────────
        t_rrf_0 = time.perf_counter()
        merged  = reciprocal_rank_fusion(all_per_book_results)
        t_rrf   = time.perf_counter() - t_rrf_0

        # ── Step 4: Cross-encoder rerank — capped at MAX_RERANK ───────────────
        t_rerank_0  = time.perf_counter()
        model       = get_cross_encoder()
        rerank_pool = merged[: self.max_rerank]
        pairs       = [(query, doc.page_content) for doc in rerank_pool]
        scores      = model.predict(pairs)
        scored      = sorted(zip(scores, rerank_pool), key=lambda x: x[0], reverse=True)
        # Filter by score threshold — drop noise; guarantee at least min_final_k chunks
        qualifying = [(s, doc) for s, doc in scored if float(s) >= self.score_threshold]
        if len(qualifying) < self.min_final_k:
            qualifying = list(scored[: self.min_final_k])
        top_docs    = [doc for _, doc in qualifying[: self.final_k]]
        t_rerank    = time.perf_counter() - t_rerank_0
        top_score   = round(float(scored[0][0]), 4) if scored else None
        log.info("Cross-encoder rerank complete", extra={
            "event": "rerank_complete",
            "candidates_in": len(rerank_pool),
            "candidates_out": len(top_docs),
            "filtered_by_threshold": len(scored) - len(qualifying),
            "top_score": top_score,
            "latency_rerank_s": round(t_rerank, 3),
        })

        t_total = time.perf_counter() - t_start
        timings = {
            "latency_hyde_s":      round(t_hyde, 3),
            "latency_embed_s":     round(t_embed, 3),
            "latency_retrieval_s": round(t_retrieval, 3),
            "latency_rrf_s":       round(t_rrf, 4),
            "latency_rerank_s":    round(t_rerank, 3),
            "latency_total_s":     round(t_total, 3),
        }
        log.info("Retrieval pipeline complete", extra={
            "event": "retrieval_pipeline_complete",
            "use_hyde": use_hyde,
            "chunks_returned": len(top_docs),
            **timings,
        })
        return top_docs, timings

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        docs, _ = self.retrieve(query, use_hyde=True)
        return docs


def build_multi_book_retriever(
    books_info: List[dict],
    fetch_k: int = FETCH_K_PER_BOOK,
    final_k: int = FINAL_K,
) -> MultiBookHyDERetriever:
    """
    Build a retriever that searches across all given collections.

    books_info: list of dicts — each must have "collection" key.
                (pass the output of medical_books() from books.py)
    Skips collections that have no vectors yet (not yet ingested).
    """
    from langchain_community.retrievers import BM25Retriever

    # One shared embeddings instance — used to pre-compute query + hyp_doc vectors
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    book_retrievers = []
    for info in books_info:
        collection = info["collection"]
        docs, vectorstore = load_all_chunks(collection)
        if not docs:
            print(f"  [multi] Skipping '{collection}' — no vectors found (not yet ingested?)")
            continue

        bm25 = BM25Retriever.from_documents(docs, k=fetch_k)
        book_retrievers.append({
            "collection": collection,
            "bm25":        bm25,
            "vectorstore": vectorstore,   # stored for similarity_search_by_vector
        })
        print(f"  [multi] Loaded '{collection}' ({len(docs)} chunks)")

    if not book_retrievers:
        raise RuntimeError("No ingested book collections found. Run: python src/ingest.py --book <key>")

    # gpt-4o-mini: 5x faster and ~10x cheaper for HyDE — just writes a 3-sentence passage
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return MultiBookHyDERetriever(
        book_retrievers=book_retrievers,
        llm=llm,
        embeddings=embeddings,
        fetch_k=fetch_k,
        final_k=final_k,
    )
