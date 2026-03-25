"""
RAG chain — generic, works for any book.
Supports multiple retriever modes: naive, hybrid, rerank, query_expansion.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from image_index import lookup_images

load_dotenv()

CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION = "fischer_surgery"
DEFAULT_BOOK_NAME = "Fischer's Mastery of Surgery"
TOP_K = 6

SYSTEM_PROMPT = """You are a precise reference assistant for "{book_name}".

Your job is to answer questions using ONLY the context provided below from the book.

RULES:
1. Use ONLY information explicitly stated in the provided context. Do not use prior knowledge.
2. Synthesize and organize the relevant information into a clear, well-structured answer.
3. Cite locations inline, e.g. (Page 42) or (Chapter 3), wherever you reference information.
4. If the context contains partial information, provide what is available and note what is incomplete.
5. ONLY if there is truly NO relevant information in the context at all, say:
   "This specific information is not found in the retrieved sections. Try rephrasing your question."
6. Do not speculate, infer, or add information beyond what is written in the context.

Context from the book:
{{context}}
"""

MULTI_BOOK_SYSTEM_PROMPT = """You are a precise medical reference assistant with access to multiple surgical textbooks.

Your job is to answer questions using ONLY the context provided below.

RULES:
1. Use ONLY information explicitly stated in the provided context. Do not use prior knowledge.
2. Synthesize and organize the relevant information into a clear, well-structured answer.
3. Cite sources inline with both book name and page, e.g. (Fischer's Mastery of Surgery, Page 42) \
or (Sabiston Textbook of Surgery, Page 157), wherever you reference information.
4. If multiple books cover the same topic, synthesize them and note where they agree or differ.
5. If the context contains partial information, provide what is available and note what is incomplete.
6. ONLY if there is truly NO relevant information in the context at all, say:
   "This specific information is not found in the retrieved sections. Try rephrasing your question."
7. Do not speculate, infer, or add information beyond what is written in the context.

STRUCTURE:
Use bold section headers when the answer is complex enough to warrant them — adapt to what the question actually asks:
- Operative/procedural questions: **Indication** · **Key Steps** · **Technical Considerations** · **Complications**
- Pathophysiology/mechanism questions: **Definition** · **Mechanism** · **Clinical Features** · **Management**
- Complication questions: **Causes** · **Recognition** · **Management** · **Prevention**
- Diagnostic/workup questions: **Presentation** · **Investigations** · **Interpretation**
- Comparison questions: a structured side-by-side or clearly labelled paragraphs per option
- Simple factual lookups: a single concise paragraph — do not force headers onto a one-line answer
Only use the sections that apply. Never include a section heading with no content under it.

Context from the books:
{context}
"""

HUMAN_PROMPT = "{question}"


def format_docs(docs):
    """Format retrieved docs with book name and page citations."""
    parts = []
    for doc in docs:
        page      = doc.metadata.get("page", "?")
        book_name = doc.metadata.get("source", "")
        label     = f"{book_name}, Page {page}" if book_name else f"Page {page}"
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_retriever(mode: str = "naive", collection: str = DEFAULT_COLLECTION) -> object:
    """
    Build a retriever based on mode.
    mode: "naive" | "hybrid" | "rerank" | "query_expansion"
    """
    if mode == "hybrid":
        from retriever_hybrid import build_hybrid_retriever
        return build_hybrid_retriever(k=TOP_K, collection=collection)

    if mode == "rerank":
        from retriever_rerank import build_rerank_retriever
        return build_rerank_retriever(collection=collection)

    if mode == "query_expansion":
        from retriever_query_expansion import build_query_expansion_retriever
        return build_query_expansion_retriever(collection=collection)

    if mode == "hyde":
        from retriever_hyde import build_hyde_retriever
        return build_hyde_retriever(collection=collection)

    if mode == "decomposition":
        from retriever_decomposition import build_decomposition_retriever
        return build_decomposition_retriever(collection=collection)

    # Default: naive dense retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def build_chain(
    mode: str = "naive",
    collection: str = DEFAULT_COLLECTION,
    book_name: str = DEFAULT_BOOK_NAME,
):
    """
    Build the full RAG chain for a given book collection.
    Returns (chain, retriever)
    """
    retriever = build_retriever(mode, collection)

    # Inject book name into the system prompt
    system_with_book = SYSTEM_PROMPT.format(book_name=book_name)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_with_book),
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


def build_generator(
    collection: str = DEFAULT_COLLECTION,
    book_name: str = DEFAULT_BOOK_NAME,
):
    """
    Build just the LLM generation part — no retriever.
    Expects input dict: {"context": formatted_str, "question": str}
    Used by the streaming endpoint to avoid running retrieval twice.
    """
    system_with_book = SYSTEM_PROMPT.format(book_name=book_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_with_book),
        ("human", HUMAN_PROMPT),
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return prompt | llm | StrOutputParser()


def build_multi_book_generator():
    """
    Build the LLM generation step for multi-book mode.
    Expects input dict: {"context": formatted_str, "question": str}
    Uses MULTI_BOOK_SYSTEM_PROMPT — citations include book name + page.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", MULTI_BOOK_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return prompt | llm | StrOutputParser()


def query_with_sources(question: str, chain, retriever, collection: str = DEFAULT_COLLECTION):
    """Run query and return source pages + any images from those pages."""
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    sources = sorted(set(doc.metadata.get("page", "?") for doc in source_docs))
    images = lookup_images(sources, collection)
    return answer, sources, images
