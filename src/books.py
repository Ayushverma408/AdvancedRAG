"""
Central registry of all available books.
Add a new entry here whenever you ingest a new PDF/EPUB.

Each book needs:
  display_name  : shown in the UI picker
  description   : subtitle shown under the book name
  category      : medical / cs / personal_dev / etc.
  pdf_path      : relative to project root
  collection    : ChromaDB collection name (must be unique, no spaces)
  icon          : emoji shown in the picker

Directory layout expected:
  data/raw/medical/        ← PDF/EPUB medical books
  data/raw/cs/             ← PDF/EPUB CS/tech books
  data/raw/personal_dev/   ← EPUB/LIT personal development books
"""

BOOKS: dict[str, dict] = {

    # ── Medical ────────────────────────────────────────────────────────────
    "fischer_surgery": {
        "display_name": "Fischer's Mastery of Surgery",
        "description":  "8th Edition — Vol 1 & 2. Comprehensive surgical reference.",
        "category":     "medical",
        "pdf_path":     "data/raw/medical/splitted_E_Christopher.pdf",
        "collection":   "fischer_surgery",
        "icon":         "🏥",
    },

    # ── CS / Tech ──────────────────────────────────────────────────────────
    "ace_programming": {
        "display_name": "Ace the Programming Interview",
        "description":  "160 Questions and Answers for Success.",
        "category":     "cs",
        "pdf_path":     "data/raw/cs/Ace the Programming Interview - 160 Questions and Answers for Success.pdf",
        "collection":   "cs_ace_programming",
        "icon":         "💻",
    },

    "clean_code_leetcode": {
        "display_name": "Clean Code Handbook — LeetCode",
        "description":  "50 Common Interview Questions with solutions.",
        "category":     "cs",
        "pdf_path":     "data/raw/cs/Clean Code Handbook - LeetCode 50 Common Interview Questions.pdf",
        "collection":   "cs_clean_code",
        "icon":         "🧹",
    },

    "grokking_algorithms": {
        "display_name": "Grokking Algorithms",
        "description":  "Illustrated guide to algorithms for programmers.",
        "category":     "cs",
        "pdf_path":     "data/raw/cs/Grokking Algorithms - Grokking Algorithms - An illustrated guide for programmers and other curious people.pdf",
        "collection":   "cs_grokking",
        "icon":         "📊",
    },

    "system_design": {
        "display_name": "System Design Interview",
        "description":  "An insider's guide to system design interviews.",
        "category":     "cs",
        "pdf_path":     "data/raw/cs/SystemDesignInterview.pdf",
        "collection":   "cs_system_design",
        "icon":         "🏗️",
    },

    # ── Personal Development ───────────────────────────────────────────────
    "how_to_win_friends": {
        "display_name": "How to Win Friends and Influence People",
        "description":  "Dale Carnegie. The timeless classic on human relations.",
        "category":     "personal_dev",
        "pdf_path":     "data/raw/personal_dev/How To Win Friends and Influence People -- Carnegie, Dale -- 2009 -- Gallery Books -- 582fcb4a691a4c3b0d394083d9748edf -- Anna\u2019s Archive.epub",
        "collection":   "pd_how_to_win_friends",
        "icon":         "🤝",
    },

    "models_manson": {
        "display_name": "Models: Attract Women Through Honesty",
        "description":  "Mark Manson. Relationships built on authenticity.",
        "category":     "personal_dev",
        "pdf_path":     "data/raw/personal_dev/Models_ Attract Women Through Honesty -- Manson, Mark -- 2011 -- 9b66f70b5214ceef08120b98d8dddd47 -- Anna\u2019s Archive.epub",
        "collection":   "pd_models_manson",
        "icon":         "❤️",
    },

    "no_mr_nice_guy": {
        "display_name": "No More Mr. Nice Guy",
        "description":  "Robert Glover. A proven plan for getting what you want.",
        "category":     "personal_dev",
        "pdf_path":     "data/raw/personal_dev/No More Mr_ Nice Guy!_ A Proven Plan For Getting What You -- Robert Glover,Robert A_ Glover -- 2001 -- Running Press -- 9781401400019 -- 03fc3aa5b3faa6d4a162289b7b30e949 -- Anna\u2019s Archive.epub",
        "collection":   "pd_no_mr_nice_guy",
        "icon":         "💪",
    },

    "effective_communication": {
        "display_name": "The Science of Effective Communication",
        "description":  "Ian Tuhovsky. Improve your social skills.",
        "category":     "personal_dev",
        # NOTE: .lit file — auto-converted to EPUB during ingestion (requires Calibre)
        "pdf_path":     "data/raw/personal_dev/The Science of Effective Communication_ Improve Your Social -- Ian Tuhovsky [Tuhovsky, Ian] -- Positive Psychology Coaching Series 15, 2017 -- 64345f691c9ea864890b083362cfdc94 -- Anna\u2019s Archive.lit",
        "collection":   "pd_effective_communication",
        "icon":         "🗣️",
    },
}


def get_book(key: str) -> dict:
    if key not in BOOKS:
        raise KeyError(f"Book '{key}' not found. Available: {list(BOOKS.keys())}")
    return BOOKS[key]


def all_books() -> list[dict]:
    """Returns list of book dicts with key included."""
    return [{"key": k, **v} for k, v in BOOKS.items()]
