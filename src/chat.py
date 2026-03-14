"""
Terminal chat interface for the RAG system.
Usage:
    python src/chat.py              # naive RAG (default)
    python src/chat.py --mode hybrid
"""

import sys
import argparse
from chain import build_chain, query_with_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="naive", choices=["naive", "hybrid", "rerank"],
                        help="Retrieval mode")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Fischer's Mastery of Surgery — RAG Assistant [{args.mode.upper()}]")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Commands: 'quit'/'exit' to stop, 'sources' to toggle source display\n")

    try:
        chain, retriever = build_chain(mode=args.mode)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Have you run ingest.py yet? Run: python src/ingest.py")
        sys.exit(1)

    show_sources = True

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if question.lower() == "sources":
            show_sources = not show_sources
            print(f"Source display: {'ON' if show_sources else 'OFF'}")
            continue

        print("\nThinking...\n")
        try:
            answer, source_pages = query_with_sources(question, chain, retriever)
            print(f"Answer:\n{answer}")
            if show_sources:
                print(f"\nRetrieved from pages: {source_pages}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
