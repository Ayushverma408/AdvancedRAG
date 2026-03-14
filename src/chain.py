"""
RAG chain with strict medical grounding.
Supports multiple retriever modes: naive, hybrid.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "fischer_surgery"
TOP_K = 6

SYSTEM_PROMPT = """You are a precise medical reference assistant for Fischer's Mastery of Surgery (8th Edition).

Your job is to answer surgical questions using ONLY the context provided below from the book.

RULES:
1. Use ONLY information explicitly stated in the provided context. Do not use prior knowledge.
2. Synthesize and organize the relevant information from the context into a clear, clinical answer.
3. Cite page numbers inline, e.g. (Page 2581), wherever you reference information.
4. If the context contains partial information, provide what is available and note what is incomplete.
5. ONLY if there is truly NO relevant information in the context at all, say:
   "This specific information is not found in the retrieved sections. Try rephrasing your question."
6. Do not speculate, infer, or add information beyond what is written in the context.

Context from the book:
{context}
"""

HUMAN_PROMPT = "{question}"


def format_docs(docs):
    """Format retrieved docs with page citations."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_retriever(mode: str = "naive") -> object:
    """
    Build a retriever based on mode.
    mode: "naive" | "hybrid"
    """
    if mode == "hybrid":
        from retriever_hybrid import build_hybrid_retriever
        return build_hybrid_retriever(k=TOP_K)

    if mode == "rerank":
        from retriever_rerank import build_rerank_retriever
        return build_rerank_retriever()

    if mode == "query_expansion":
        from retriever_query_expansion import build_query_expansion_retriever
        return build_query_expansion_retriever()

    # Default: naive dense retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def build_chain(mode: str = "naive"):
    """
    Build the full RAG chain.
    mode: "naive" | "hybrid"
    Returns (chain, retriever)
    """
    retriever = build_retriever(mode)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query_with_sources(question: str, chain, retriever):
    """Run query and also return source pages for transparency."""
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    sources = sorted(set(doc.metadata.get("page", "?") for doc in source_docs))
    return answer, sources
