"""
Reranking Retriever: Hybrid retrieval + Cross-encoder reranking.

Why reranking?
- Hybrid retrieval gives high recall (we get the right chunks) but lower precision
  (some noise from BM25 keyword matches that aren't truly relevant)
- A cross-encoder reads (query, chunk) together and scores relevance deeply
- Much more accurate than embedding similarity, but too slow to run on all 12k chunks
- So: hybrid fetches top-20 candidates, cross-encoder rescores, we keep top-6

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking (factual Q&A)
- Small (~80MB), runs on CPU, ~50ms per query on 20 candidates
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from typing import List

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FETCH_K = 20   # fetch this many from hybrid before reranking
FINAL_K = 6    # keep this many after reranking

_model_cache = None


def get_cross_encoder() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder model."""
    global _model_cache
    if _model_cache is None:
        print(f"Loading cross-encoder: {RERANK_MODEL}")
        _model_cache = CrossEncoder(RERANK_MODEL)
        print("Cross-encoder loaded")
    return _model_cache


class RerankRetriever(BaseRetriever):
    """Hybrid retrieval + cross-encoder reranking."""

    hybrid_retriever: object
    fetch_k: int = FETCH_K
    final_k: int = FINAL_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Step 1: Hybrid retrieval — get broad set of candidates
        # Temporarily override k for wider net
        self.hybrid_retriever.k = self.fetch_k
        candidates = self.hybrid_retriever.invoke(query)
        self.hybrid_retriever.k = self.final_k  # restore

        if not candidates:
            return []

        # Step 2: Cross-encoder reranking
        model = get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = model.predict(pairs)

        # Step 3: Sort by score descending, return top final_k
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.final_k]]


def build_rerank_retriever(fetch_k: int = FETCH_K, final_k: int = FINAL_K) -> RerankRetriever:
    from retriever_hybrid import build_hybrid_retriever
    hybrid = build_hybrid_retriever(k=fetch_k)
    return RerankRetriever(hybrid_retriever=hybrid, fetch_k=fetch_k, final_k=final_k)
