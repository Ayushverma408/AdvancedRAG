"""
Ingest books into ChromaDB.

Usage:
    # Ingest a single book by key
    python src/ingest.py --book fischer_surgery

    # Ingest all registered books
    python src/ingest.py --all

    # List available books
    python src/ingest.py --list

Supported formats: PDF, EPUB, LIT (LIT requires Calibre's ebook-convert in PATH)

Dependencies:
    pip install ebooklib beautifulsoup4
"""

import os
import sys
import argparse
import subprocess
import tempfile

import fitz  # pymupdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

sys.path.insert(0, os.path.dirname(__file__))
from books import get_book, all_books

load_dotenv()

CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_pdf(path: str, book_name: str) -> list[Document]:
    """Load PDF page by page, preserving page number metadata."""
    print(f"  Loading PDF: {path}")
    doc = fitz.open(path)
    documents = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if len(text) < 50:
            continue
        documents.append(Document(
            page_content=text,
            metadata={"source": book_name, "page": page_num + 1},
        ))

    doc.close()
    print(f"  Loaded {len(documents)} non-empty pages out of {page_num + 1} total")
    return documents


def load_epub(path: str, book_name: str) -> list[Document]:
    """Load EPUB chapter by chapter. Requires: pip install ebooklib beautifulsoup4"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "EPUB support requires: pip install ebooklib beautifulsoup4"
        )

    print(f"  Loading EPUB: {path}")
    book = epub.read_epub(path)
    documents = []
    chapter_idx = 0

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if len(text) < 50:
            continue
        chapter_idx += 1
        documents.append(Document(
            page_content=text,
            metadata={"source": book_name, "page": chapter_idx},
        ))

    print(f"  Loaded {len(documents)} chapters")
    return documents


def load_lit(path: str, book_name: str) -> list[Document]:
    """
    Load a .lit file by converting it to EPUB via Calibre, then loading the EPUB.
    Requires Calibre: https://calibre-ebook.com/download
    """
    calibre = _calibre_bin()
    if not calibre:
        raise RuntimeError(
            "LIT files require Calibre for conversion.\n"
            "Install from https://calibre-ebook.com/download\n"
            "then ensure 'ebook-convert' is in your PATH."
        )

    print(f"  Converting LIT → EPUB via Calibre: {path}")
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp_epub = tmp.name

    try:
        subprocess.run(
            [calibre, path, tmp_epub],
            check=True,
            capture_output=True,
        )
        docs = load_epub(tmp_epub, book_name)
    finally:
        if os.path.exists(tmp_epub):
            os.unlink(tmp_epub)

    return docs


_CALIBRE_PATHS = [
    "ebook-convert",
    "/Applications/calibre.app/Contents/MacOS/ebook-convert",
    "/usr/local/bin/ebook-convert",
]


def _calibre_bin() -> str | None:
    """Return path to ebook-convert if available, else None."""
    for path in _CALIBRE_PATHS:
        try:
            subprocess.run([path, "--version"], capture_output=True, check=True)
            return path
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def load_file(path: str, book_name: str) -> list[Document]:
    """Dispatch to the right loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path, book_name)
    elif ext == ".epub":
        return load_epub(path, book_name)
    elif ext == ".lit":
        return load_lit(path, book_name)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .epub, .lit")


# ── Chunking & embedding ──────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split into smaller chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks from {len(documents)} pages/chapters")
    return chunks


def build_vectorstore(chunks: list[Document], collection: str) -> Chroma:
    """Embed chunks and store in ChromaDB under the given collection name."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"  Embedding {len(chunks)} chunks into collection '{collection}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors in '{collection}'")
    return vectorstore


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest_book(key: str):
    book = get_book(key)
    path = book["pdf_path"]
    name = book["display_name"]
    collection = book["collection"]

    print(f"\n{'=' * 60}")
    print(f"Ingesting: {name}")
    print(f"  File:       {path}")
    print(f"  Collection: {collection}")
    print(f"{'=' * 60}")

    if not os.path.exists(path):
        print(f"  ERROR: File not found at '{path}'. Skipping.")
        return

    docs = load_file(path, name)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks, collection)
    print(f"  ✓ {key} ingested successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest books into ChromaDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--book", metavar="KEY", help="Ingest a single book by key")
    group.add_argument("--all", action="store_true", help="Ingest all registered books")
    group.add_argument("--list", action="store_true", help="List all available book keys")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable books:")
        for b in all_books():
            status = "✓" if os.path.exists(b["pdf_path"]) else "✗ (file missing)"
            print(f"  {b['key']:<25} {b['icon']}  {b['display_name']}  [{status}]")
        sys.exit(0)

    if args.book:
        ingest_book(args.book)
    elif args.all:
        books = all_books()
        print(f"Ingesting all {len(books)} books...")
        for b in books:
            ingest_book(b["key"])
        print("\nAll done.")
