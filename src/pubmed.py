"""
PubMed E-utilities helper.
Uses only stdlib (urllib) — no new pip dependencies required.
"""
import json
import urllib.parse
import urllib.request

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_COMMON = {"tool": "scrubref", "email": "noreply@scrubref.com"}


def search_pubmed(query: str, max_results: int = 3) -> list[dict]:
    """
    Search PubMed and return up to max_results article summaries.
    Returns [] on any error (network, parse, timeout).
    Each item: {pmid, title, authors, year, journal, url}
    """
    # Step 1: esearch — get PMIDs ranked by relevance
    params = urllib.parse.urlencode({
        "db": "pubmed", "term": query,
        "retmax": max_results, "retmode": "json", "sort": "relevance",
        **_COMMON,
    })
    try:
        with urllib.request.urlopen(f"{EUTILS}/esearch.fcgi?{params}", timeout=8) as r:
            pmids = json.loads(r.read()).get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []
    if not pmids:
        return []

    # Step 2: esummary — get metadata (title, authors, journal, date)
    params = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids),
        "retmode": "json", **_COMMON,
    })
    try:
        with urllib.request.urlopen(f"{EUTILS}/esummary.fcgi?{params}", timeout=8) as r:
            result_set = json.loads(r.read()).get("result", {})
    except Exception:
        return []

    out = []
    for pmid in pmids:
        item = result_set.get(pmid)
        if not item or not isinstance(item, dict):
            continue
        title = item.get("title", "").rstrip(".")
        raw_authors = item.get("authors", [])
        if len(raw_authors) == 0:
            authors = ""
        elif len(raw_authors) == 1:
            authors = raw_authors[0].get("name", "")
        elif len(raw_authors) == 2:
            authors = f"{raw_authors[0].get('name', '')}, {raw_authors[1].get('name', '')}"
        else:
            authors = f"{raw_authors[0].get('name', '')} et al."
        year = (item.get("pubdate") or "")[:4]
        journal = item.get("source", "")
        out.append({
            "pmid":    pmid,
            "title":   title,
            "authors": authors,
            "year":    year,
            "journal": journal,
            "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return out
