"""
Shared retrieval utilities used by the multi-book pipeline.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION = "fischer_surgery"
RRF_K = 60  # standard RRF constant


def load_vectorstore(collection: str = DEFAULT_COLLECTION):
    """Return a Chroma vectorstore for the given collection. Does NOT load all chunks into memory."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    count = vs._collection.count()
    return count, vs


def load_all_chunks(collection: str = DEFAULT_COLLECTION):
    """Pull all stored chunks from Chroma in batches (avoids SQLite variable limit).
    NOTE: only needed if building a BM25 index. For dense-only retrieval use load_vectorstore()."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    total = vs._collection.count()
    batch_size = 5000
    docs = []
    for offset in range(0, total, batch_size):
        raw = vs.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        for text, meta in zip(raw["documents"], raw["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta))
    print(f"Loaded {len(docs)} chunks")
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
            key = doc.page_content
            if key not in scores:
                scores[key] = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (k + rank + 1)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]
