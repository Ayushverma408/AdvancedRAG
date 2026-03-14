"""
Query Expansion Retriever: generates multiple query variants, retrieves for each,
merges via RRF, then reranks with cross-encoder.

Why this helps for medical text:
- Users ask "Whipple procedure" but book says "pancreaticoduodenectomy"
- Users ask "weight loss surgery" but book says "bariatric surgery"
- Users ask about "side effects" but book says "complications" or "morbidity"
- Cross-encoder then picks the best chunks from the merged pool
"""

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from typing import List

from retriever_hybrid import build_hybrid_retriever, reciprocal_rank_fusion
from retriever_rerank import get_cross_encoder, RERANK_MODEL

EXPANSION_PROMPT = """You are a medical search expert. Given a surgical/medical question, generate {n} alternative search queries that capture the same intent using different terminology.

Rules:
- Use clinical/anatomical synonyms (e.g. "Whipple" → "pancreaticoduodenectomy")
- Vary specificity (broader and narrower versions)
- Use both lay and technical terms where applicable
- Each query should be a standalone search query, not a sentence
- Return ONLY the queries, one per line, no numbering or explanation

Original question: {question}

Alternative queries:"""

FETCH_K_PER_QUERY = 8   # candidates per query variant
NUM_VARIANTS = 3        # number of extra query variants to generate
FINAL_K = 6             # final chunks after reranking


class QueryExpansionRetriever(BaseRetriever):
    """Expands query into variants, retrieves for all, merges via RRF, reranks."""

    hybrid_retriever: object
    llm: object
    num_variants: int = NUM_VARIANTS
    fetch_k_per_query: int = FETCH_K_PER_QUERY
    final_k: int = FINAL_K

    class Config:
        arbitrary_types_allowed = True

    def _generate_variants(self, query: str) -> list[str]:
        """Use GPT-4o to generate alternative phrasings of the query."""
        prompt = EXPANSION_PROMPT.format(n=self.num_variants, question=query)
        response = self.llm.invoke(prompt)
        variants = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return variants[: self.num_variants]

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Step 1: Generate query variants
        variants = self._generate_variants(query)
        all_queries = [query] + variants
        print(f"    Expanded to {len(all_queries)} queries: {[q[:50] for q in all_queries]}")

        # Step 2: Retrieve for each query variant
        self.hybrid_retriever.k = self.fetch_k_per_query
        all_result_lists = []
        for q in all_queries:
            results = self.hybrid_retriever.invoke(q)
            all_result_lists.append(results)

        # Step 3: Merge all result lists via RRF
        merged = reciprocal_rank_fusion(all_result_lists)

        # Step 4: Cross-encoder rerank on the merged pool
        model = get_cross_encoder()
        if not merged:
            return []

        pairs = [(query, doc.page_content) for doc in merged]
        scores = model.predict(pairs)
        scored = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored[: self.final_k]]


def build_query_expansion_retriever(
    num_variants: int = NUM_VARIANTS,
    fetch_k_per_query: int = FETCH_K_PER_QUERY,
    final_k: int = FINAL_K,
) -> QueryExpansionRetriever:
    hybrid = build_hybrid_retriever(k=fetch_k_per_query)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)  # slight temp for variant diversity

    return QueryExpansionRetriever(
        hybrid_retriever=hybrid,
        llm=llm,
        num_variants=num_variants,
        fetch_k_per_query=fetch_k_per_query,
        final_k=final_k,
    )
