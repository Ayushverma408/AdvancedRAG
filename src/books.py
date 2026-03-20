"""
Central registry of medical books.
Add a new entry here whenever you ingest a new PDF.

Each book needs:
  display_name  : shown in citations and UI
  description   : subtitle / edition info
  category      : medical
  pdf_path      : relative to project root (data/raw/medical/)
  collection    : ChromaDB collection name (must be unique, no spaces)

Ingest a new book:
  python src/ingest.py --book <key>
"""

BOOKS: dict[str, dict] = {

    "fischer_surgery": {
        "display_name": "Fischer's Mastery of Surgery",
        "description":  "8th Edition — Vol 1 & 2. Comprehensive surgical reference.",
        "category":     "medical",
        "pdf_path":     "data/raw/medical/splitted_E_Christopher.pdf",
        "collection":   "fischer_surgery",
    },

    "sabiston_surgery": {
        "display_name": "Sabiston Textbook of Surgery",
        "description":  "22nd Edition. The Biological Basis of Modern Surgical Practice.",
        "category":     "medical",
        "pdf_path":     "data/raw/medical/Sabiston 22nd Edition (3).pdf",
        "collection":   "sabiston_surgery",
    },

    "shackelford_alimentary": {
        "display_name": "Shackelford's Surgery of the Alimentary Tract",
        "description":  "9th Edition. Comprehensive reference for GI surgery.",
        "category":     "medical",
        "pdf_path":     "data/raw/medical/Shackelford_s surgery of the alimentary tract ninth edition.pdf",
        "collection":   "shackelford_alimentary",
    },

    "blumgart_liver": {
        "display_name": "Blumgart's Surgery of the Liver, Biliary Tract and Pancreas",
        "description":  "Jarnagin edition. Definitive reference for HPB surgery.",
        "category":     "medical",
        "pdf_path":     "data/raw/medical/William_R_Jarnagin_Blumgart's_Surgery_of_the_Liver,_Biliary_Tract.pdf",
        "collection":   "blumgart_liver",
    },

}


def get_book(key: str) -> dict:
    if key not in BOOKS:
        raise KeyError(f"Book '{key}' not found. Available: {list(BOOKS.keys())}")
    return BOOKS[key]


def all_books() -> list[dict]:
    """Returns list of book dicts with key included."""
    return [{"key": k, **v} for k, v in BOOKS.items()]


def medical_books() -> list[dict]:
    """Returns all medical books as list of dicts with key included."""
    return [{"key": k, **v} for k, v in BOOKS.items() if v["category"] == "medical"]


def get_book_by_collection(collection: str) -> dict:
    """Look up a book by its ChromaDB collection name."""
    for book in BOOKS.values():
        if book["collection"] == collection:
            return book
    raise KeyError(f"No book with collection '{collection}'")
