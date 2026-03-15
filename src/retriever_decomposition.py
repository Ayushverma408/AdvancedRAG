"""
Query Decomposition + HyDE Retriever.

Problem: Broad questions like "how do I become more confident?" span multiple
topics — body language, mindset, habits, social proof, etc. A single retrieval
pass picks up one aspect at most.

Solution:
1. Decompose the broad question into N specific sub-questions
2. For each sub-question, run HyDE (hypothetical doc → dense retrieval)
3. Also run BM25 on the original question as a keyword anchor
4. RRF merge ALL result lists
5. Cross-encoder rerank the merged pool against the ORIGINAL question

This gives broad coverage (decomposition) + semantic precision (HyDE) +
keyword matching (BM25) — in one pipeline.

Cost: ~(N+1) LLM calls per query (decomposition + N HyDE generations)
      Typically N=3 → 4 LLM calls total + 1 for the final answer = 5
"""

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

from retriever_hybrid import load_all_chunks, reciprocal_rank_fusion
from retriever_rerank import get_cross_encoder

FETCH_K_PER_SUB = 8   # retrieved per sub-question
NUM_SUB_QUESTIONS = 3
FINAL_K = 6

DECOMPOSE_PROMPT = """\
You are an expert at breaking down questions into precise search queries.

Given the question below, generate {n} specific sub-questions that together \
fully cover what the user is asking. Each sub-question should target a distinct \
aspect and be concrete enough to retrieve a specific passage from a book.

Rules:
- Each sub-question must be self-contained and searchable
- Cover different angles: definitions, techniques, examples, causes, effects
- Do not repeat the same idea with different words
- Return ONLY the sub-questions, one per line, no numbering or explanation

Question: {question}

Sub-questions:"""

HYDE_PROMPT = """\
Write a short passage (3–5 sentences) that a book would contain to directly \
answer the question below. Write it as a book author would — no preamble.

Question: {question}

Passage:"""


class DecompositionRetriever(BaseRetriever):
    """Decomposes question → HyDE per sub-question → RRF merge → rerank."""

    bm25_retriever: BM25Retriever
    dense_retriever: object
    llm: object
    num_sub_questions: int = NUM_SUB_QUESTIONS
    fetch_k_per_sub: int = FETCH_K_PER_SUB
    final_k: int = FINAL_K

    class Config:
        arbitrary_types_allowed = True

    def _decompose(self, query: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT.format(n=self.num_sub_questions, question=query)
        response = self.llm.invoke(prompt).content.strip()
        subs = [
            line.strip()
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 8
        ]
        return subs[: self.num_sub_questions]

    def _hyde_doc(self, question: str) -> str:
        return self.llm.invoke(HYDE_PROMPT.format(question=question)).content.strip()

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Step 1: Decompose
        sub_questions = self._decompose(query)
        print(f"    Decomposed into {len(sub_questions)} sub-questions:")
        for sq in sub_questions:
            print(f"      · {sq[:80]}")

        all_result_lists = []

        # Step 2: For each sub-question → HyDE → dense retrieval
        self.dense_retriever.search_kwargs["k"] = self.fetch_k_per_sub
        for sq in sub_questions:
            hyp_doc = self._hyde_doc(sq)
            results = self.dense_retriever.invoke(hyp_doc)
            all_result_lists.append(results)
            # Also dense on the sub-question itself
            all_result_lists.append(self.dense_retriever.invoke(sq))

        # Step 3: BM25 on original query as keyword anchor
        self.bm25_retriever.k = self.fetch_k_per_sub
        all_result_lists.append(self.bm25_retriever.invoke(query))

        # Step 4: RRF merge all result lists
        merged = reciprocal_rank_fusion(all_result_lists)

        if not merged:
            return []

        # Step 5: Cross-encoder rerank against ORIGINAL query
        model = get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in merged]
        scores = model.predict(pairs)
        scored = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.final_k]]


def build_decomposition_retriever(
    num_sub_questions: int = NUM_SUB_QUESTIONS,
    fetch_k_per_sub: int = FETCH_K_PER_SUB,
    final_k: int = FINAL_K,
    collection: str = "fischer_surgery",
) -> DecompositionRetriever:
    docs, vectorstore = load_all_chunks(collection)

    bm25 = BM25Retriever.from_documents(docs, k=fetch_k_per_sub)
    dense = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": fetch_k_per_sub},
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    return DecompositionRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        llm=llm,
        num_sub_questions=num_sub_questions,
        fetch_k_per_sub=fetch_k_per_sub,
        final_k=final_k,
    )
