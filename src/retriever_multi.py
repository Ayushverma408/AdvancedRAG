"""
Multi-Book HyDE Retriever.

Searches across all ingested medical book collections simultaneously:
1. Generate ONE hypothetical document (HyDE) for the query — shared across all books.
2. For each book collection: retrieve via dense(hyp_doc) + dense(query) + BM25(query),
   merge per-book results via RRF.
3. Globally merge all per-book result lists via RRF.
4. Cross-encoder rerank the global pool and return top K.

Each returned Document has:
  doc.metadata["source"]     — book display name  (set at ingest time)
  doc.metadata["page"]       — page number        (set at ingest time)
  doc.metadata["collection"] — ChromaDB collection name (tagged here at retrieval time)
"""

from typing import List, Any

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from retriever_hybrid import load_all_chunks, reciprocal_rank_fusion
from retriever_rerank import get_cross_encoder

FETCH_K_PER_BOOK = 10  # candidates fetched per source (dense-hyp / dense-orig / bm25) per book
FINAL_K = 6            # final docs returned after global rerank

HYDE_PROMPT = """\
You are an expert medical author. Given the question below, write a short \
passage (3–5 sentences) that a medical textbook would contain to directly answer it. \
Use precise clinical language and terminology. Do not say "the book says" — \
just write the passage itself as if it appeared in the textbook.

Question: {question}

Passage:"""


class MultiBookHyDERetriever(BaseRetriever):
    """
    HyDE retriever that searches across multiple ChromaDB book collections,
    merges results globally via RRF, and reranks with a cross-encoder.
    """

    # Each entry: {"collection": str, "bm25": BM25Retriever, "dense": VectorStoreRetriever}
    book_retrievers: List[Any]
    llm: Any
    fetch_k: int = FETCH_K_PER_BOOK
    final_k: int = FINAL_K

    class Config:
        arbitrary_types_allowed = True

    def _generate_hypothetical_doc(self, query: str) -> str:
        return self.llm.invoke(HYDE_PROMPT.format(question=query)).content.strip()

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Step 1: One hypothetical doc shared across all books
        hyp_doc = self._generate_hypothetical_doc(query)
        print(f"    HyDE doc: {hyp_doc[:120]}...")

        all_per_book_results: List[List[Document]] = []

        for br in self.book_retrievers:
            dense = br["dense"]
            bm25 = br["bm25"]
            collection = br["collection"]

            dense.search_kwargs["k"] = self.fetch_k

            # Three retrieval signals per book
            hyp_results  = dense.invoke(hyp_doc)
            orig_results = dense.invoke(query)
            bm25.k = self.fetch_k
            bm25_results = bm25.invoke(query)

            # Tag every doc with its collection so app.py can look up images later
            for doc in hyp_results + orig_results + bm25_results:
                doc.metadata.setdefault("collection", collection)

            # Per-book RRF: normalise within the book before global merge
            book_merged = reciprocal_rank_fusion([hyp_results, orig_results, bm25_results])
            if book_merged:
                all_per_book_results.append(book_merged)

        if not all_per_book_results:
            return []

        # Step 2: Global RRF across all books
        merged = reciprocal_rank_fusion(all_per_book_results)

        # Step 3: Cross-encoder rerank on the global pool
        model = get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in merged]
        scores = model.predict(pairs)
        scored = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.final_k]]


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

    book_retrievers = []
    for info in books_info:
        collection = info["collection"]
        docs, vectorstore = load_all_chunks(collection)
        if not docs:
            print(f"  [multi] Skipping '{collection}' — no vectors found (not yet ingested?)")
            continue

        bm25  = BM25Retriever.from_documents(docs, k=fetch_k)
        dense = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": fetch_k},
        )
        book_retrievers.append({
            "collection": collection,
            "bm25":        bm25,
            "dense":       dense,
        })
        print(f"  [multi] Loaded '{collection}' ({len(docs)} chunks)")

    if not book_retrievers:
        raise RuntimeError("No ingested book collections found. Run: python src/ingest.py --book <key>")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    return MultiBookHyDERetriever(
        book_retrievers=book_retrievers,
        llm=llm,
        fetch_k=fetch_k,
        final_k=final_k,
    )
