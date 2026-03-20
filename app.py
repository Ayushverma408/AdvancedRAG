"""
Chainlit UI for the Medical RAG system.
Run: chainlit run app.py

Searches across ALL ingested medical book collections simultaneously.
Results are merged via RRF and reranked globally — no book selector needed.

Auth: users sign up/log in once via the Next.js web app (web/).
      The JWT cookie from port 3000 is automatically sent to Chainlit on port 8000
      (localhost cookies are not port-scoped per RFC 6265). Single login, no prompt.
Sessions: stored in PostgreSQL via PostgreSQLDataLayer (src/chainlit_data_layer.py).
"""

import os
import sys
import asyncio
from collections import defaultdict

import chainlit as cl
import chainlit.data as cl_data
import jwt as pyjwt
from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is on path and cwd is project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from chain import build_multi_book_generator, format_docs
from image_index import lookup_images
from books import medical_books
from retriever_multi import build_multi_book_retriever
from chainlit_data_layer import PostgreSQLDataLayer

# ── Data layer ────────────────────────────────────────────────────────────────
_DB_URL = os.environ.get("DATABASE_URL", "")
if _DB_URL:
    cl_data.chainlit_data_layer = PostgreSQLDataLayer(_DB_URL)


# ── Auth ──────────────────────────────────────────────────────────────────────
@cl.header_auth_callback
def header_auth_callback(headers: dict) -> cl.User | None:
    """
    Read the JWT set by the Next.js web app from the cookie header.
    Cookies on localhost are not port-scoped, so the auth_token cookie
    from port 3000 is sent automatically to Chainlit on port 8000.
    """
    jwt_secret = os.environ.get("JWT_SECRET", "")
    if not jwt_secret:
        return cl.User(identifier="dev", metadata={"name": "Dev User"})

    cookie_str = headers.get("cookie", "")
    cookies = dict(
        part.strip().split("=", 1)
        for part in cookie_str.split(";")
        if "=" in part
    )
    token = cookies.get("auth_token", "").strip()
    if not token:
        return None

    try:
        payload = pyjwt.decode(token, jwt_secret, algorithms=["HS256"])
        return cl.User(
            identifier=payload["email"],
            metadata={"id": payload["userId"], "name": payload["name"]},
        )
    except Exception:
        return None


MAX_IMAGES = 6  # cap images shown per response


@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    name = (user.metadata.get("name") if user else None) or "there"
    loading = await cl.Message(content="⚙️ Loading pipeline, please wait…").send()

    loop = asyncio.get_event_loop()
    retriever, generator = await loop.run_in_executor(None, _build_pipeline)

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("generator", generator)

    books = medical_books()
    book_names = [b["display_name"] for b in books if _collection_loaded(b["collection"])]

    loading.content = (
        f"Hi {name}! Ready to answer questions across your medical library.\n\n"
        f"**Books loaded:** {', '.join(book_names)}\n\n"
        f"Pipeline: `MULTI-BOOK HyDE`"
    )
    await loading.update()


def _collection_loaded(collection: str) -> bool:
    """Check if a ChromaDB collection has any vectors (was ingested)."""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        vs = Chroma(
            collection_name=collection,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="chroma_db",
        )
        return vs._collection.count() > 0
    except Exception:
        return False


def _build_pipeline():
    """Blocking — runs in executor."""
    books = medical_books()
    retriever = build_multi_book_retriever(books)
    generator = build_multi_book_generator()
    return retriever, generator


@cl.on_message
async def on_message(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    generator = cl.user_session.get("generator")
    question  = message.content.strip()

    if not question:
        return

    loop = asyncio.get_event_loop()

    # ── Retrieve ─────────────────────────────────────────────────────────────
    async with cl.Step(name="Retrieving context") as step:
        docs = await loop.run_in_executor(None, retriever.invoke, question)

        # Summarise: which pages from which books
        source_summary = []
        for doc in docs:
            book_name = doc.metadata.get("source", "Unknown")
            page      = doc.metadata.get("page", "?")
            source_summary.append(f"{book_name} p.{page}")

        step.output = f"Retrieved {len(docs)} chunks: {', '.join(source_summary)}"

    # ── Generate ──────────────────────────────────────────────────────────────
    async with cl.Step(name="Generating answer") as step:
        context = format_docs(docs)
        answer  = await loop.run_in_executor(
            None,
            lambda: generator.invoke({"context": context, "question": question}),
        )
        step.output = f"{len(answer)} characters generated"

    # ── Image lookup — per collection ─────────────────────────────────────────
    pages_by_collection: dict[str, set] = defaultdict(set)
    for doc in docs:
        col  = doc.metadata.get("collection", "")
        page = doc.metadata.get("page")
        if col and page is not None:
            pages_by_collection[col].add(page)

    image_paths = []
    for col, pages in pages_by_collection.items():
        image_paths.extend(lookup_images(sorted(pages), col))

    elements = []
    for path in image_paths[:MAX_IMAGES]:
        abs_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        if os.path.exists(abs_path):
            elements.append(
                cl.Image(path=abs_path, name=os.path.basename(abs_path), display="inline")
            )

    # ── Reply ─────────────────────────────────────────────────────────────────
    # Build deduplicated citation list: "BookName, p.N"
    seen_citations: set[str] = set()
    citations = []
    for doc in docs:
        book_name = doc.metadata.get("source", "Unknown")
        page      = doc.metadata.get("page", "?")
        cite      = f"{book_name}, p.{page}"
        if cite not in seen_citations:
            seen_citations.add(cite)
            citations.append(cite)

    img_note = f" · {len(elements)} image(s) attached" if elements else ""
    footer   = f"\n\n---\n📄 **Sources:** {' | '.join(citations)}{img_note}"

    await cl.Message(content=answer + footer, elements=elements).send()
