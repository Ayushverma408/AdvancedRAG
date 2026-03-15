"""
Hybrid Retriever: Dense (Chroma) + Sparse (BM25) combined via Reciprocal Rank Fusion.

Why hybrid?
- Dense (embeddings): captures semantic meaning, handles paraphrasing
- BM25: exact keyword matches — critical for medical terms, drug names, procedure names
  e.g. "pancreaticoduodenectomy", "anastomosis", "Roux-en-Y"

Reciprocal Rank Fusion (RRF):
  score(doc) = sum(1 / (k + rank_i)) for each retriever i
  k=60 is standard. Naturally down-weights low-ranked results from each list.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION = "fischer_surgery"
TOP_K = 6
RRF_K = 60  # standard RRF constant


def load_all_chunks(collection: str = DEFAULT_COLLECTION):
    """Pull all stored chunks from Chroma (no re-embedding needed)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    raw = vs.get()
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    print(f"Loaded {len(docs)} chunks for BM25 index")
    return docs, vs


def reciprocal_rank_fusion(results_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    Higher score = more relevant. Deduplicates by page_content.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content  # use content as unique key
            if key not in scores:
                scores[key] = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (k + rank + 1)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


class HybridRetriever(BaseRetriever):
    """Combines BM25 and dense vector retrieval using Reciprocal Rank Fusion."""

    bm25_retriever: BM25Retriever
    dense_retriever: object
    k: int = TOP_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Fetch more than k from each so RRF has enough to work with
        fetch_k = self.k * 3

        self.bm25_retriever.k = fetch_k
        bm25_results = self.bm25_retriever.invoke(query)

        dense_results = self.dense_retriever.invoke(query)

        # RRF merge
        merged = reciprocal_rank_fusion([bm25_results, dense_results])
        return merged[: self.k]


def build_hybrid_retriever(k: int = TOP_K, collection: str = DEFAULT_COLLECTION) -> HybridRetriever:
    docs, vectorstore = load_all_chunks(collection)

    bm25 = BM25Retriever.from_documents(docs, k=k * 3)

    dense = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k * 3},
    )

    return HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, k=k)
