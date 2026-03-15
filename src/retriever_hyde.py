"""
HyDE (Hypothetical Document Embeddings) Retriever.

Problem: Generic questions like "how do I communicate better?" have no semantic
match in the book because the question lives in a different embedding space than
the answer text.

Solution:
1. Ask GPT to write a SHORT hypothetical passage that would answer the question
2. Embed THAT passage — it's in the same embedding space as real book content
3. Retrieve using the hypothetical embedding (dense)
4. Also retrieve using original question (BM25 + dense)
5. RRF merge all three result lists
6. Cross-encoder rerank the merged pool

Example:
  Q: "how do I communicate better?"
  Hypothetical: "Effective communication requires active listening, clear articulation
                 of ideas, and the ability to read non-verbal cues from your audience..."
  That hypothetical matches real book paragraphs on communication far better than
  the original question would.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from typing import List

from retriever_hybrid import load_all_chunks, reciprocal_rank_fusion
from retriever_rerank import get_cross_encoder

CHROMA_DIR = "chroma_db"
FETCH_K = 15   # per retrieval source before merging
FINAL_K = 6

HYDE_PROMPT = """\
You are an expert at understanding books. Given the question below, write a short \
passage (3–5 sentences) that a book would contain to directly answer it. \
Use the kind of language and depth an author would — do not say "the book says", \
just write the passage itself.

Question: {question}

Passage:"""


class HyDERetriever(BaseRetriever):
    """HyDE: retrieves using a GPT-generated hypothetical answer embedding."""

    bm25_retriever: BM25Retriever
    dense_retriever: object
    llm: object
    fetch_k: int = FETCH_K
    final_k: int = FINAL_K

    class Config:
        arbitrary_types_allowed = True

    def _generate_hypothetical_doc(self, query: str) -> str:
        prompt = HYDE_PROMPT.format(question=query)
        return self.llm.invoke(prompt).content.strip()

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Step 1: Generate hypothetical document
        hyp_doc = self._generate_hypothetical_doc(query)
        print(f"    HyDE doc: {hyp_doc[:120]}...")

        # Step 2: Dense retrieval on hypothetical doc (main HyDE signal)
        self.dense_retriever.search_kwargs["k"] = self.fetch_k
        hyp_results = self.dense_retriever.invoke(hyp_doc)

        # Step 3: Dense retrieval on original query
        orig_dense = self.dense_retriever.invoke(query)

        # Step 4: BM25 on original query (keyword anchor)
        self.bm25_retriever.k = self.fetch_k
        bm25_results = self.bm25_retriever.invoke(query)

        # Step 5: RRF merge — hypothetical dense weighted first (most signal)
        merged = reciprocal_rank_fusion([hyp_results, orig_dense, bm25_results])

        # Step 6: Cross-encoder rerank
        if not merged:
            return []
        model = get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in merged]
        scores = model.predict(pairs)
        scored = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.final_k]]


def build_hyde_retriever(
    fetch_k: int = FETCH_K,
    final_k: int = FINAL_K,
    collection: str = "fischer_surgery",
) -> HyDERetriever:
    docs, vectorstore = load_all_chunks(collection)

    bm25 = BM25Retriever.from_documents(docs, k=fetch_k)
    dense = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": fetch_k})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    return HyDERetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        llm=llm,
        fetch_k=fetch_k,
        final_k=final_k,
    )
