"""
Naive RAG - Step 1: Ingest PDF into ChromaDB
Loads Fischer's Mastery of Surgery, chunks it, embeds with OpenAI, stores in Chroma.
"""

import os
import fitz  # pymupdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = "data/raw/splitted_E_Christopher.pdf"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "fischer_surgery"

# Chunk config — medical text is dense, use moderate size with good overlap
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdf(path: str) -> list[Document]:
    """Load PDF page by page, preserving page number metadata."""
    print(f"Loading PDF: {path}")
    doc = fitz.open(path)
    documents = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if len(text) < 50:  # skip nearly empty pages (figures, blank pages)
            continue

        documents.append(Document(
            page_content=text,
            metadata={
                "source": "Fischer's Mastery of Surgery, 8th Ed",
                "page": page_num + 1,  # 1-indexed
            }
        ))

    doc.close()
    print(f"Loaded {len(documents)} non-empty pages out of {page_num + 1} total")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split pages into smaller chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Embedding {len(chunks)} chunks and storing in ChromaDB...")
    print("This will take a few minutes and cost ~$0.05 in OpenAI API calls.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. Vector store saved to: {CHROMA_DIR}/")
    return vectorstore


if __name__ == "__main__":
    docs = load_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    print(f"\nIngestion complete. {vectorstore._collection.count()} vectors stored.")
