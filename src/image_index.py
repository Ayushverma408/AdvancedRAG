"""
Image extraction and lookup for PDF books.

Index format (per entry):
  { "path": "data/images/{collection}/page_N_img_I.ext", "caption": "Figure 3-2. ..." }

During ingest  : extract_images() pulls images + captions, writes index.json.
During query   : lookup_images() returns list of {"path", "caption"} dicts.
Standalone CLI : python src/image_index.py --reindex <collection> <pdf_path>
                 Rebuilds captions without re-extracting image files.

Only PDFs are supported (fitz/pymupdf).
"""

import os
import sys
import json
import logging
import argparse

log = logging.getLogger(__name__)

import fitz  # pymupdf

IMAGES_DIR = "data/images"
MIN_WIDTH  = 100   # pixels — filter out tiny icons/decorations
MIN_HEIGHT = 100
CAPTION_BELOW_PX = 100   # how far below image rect to search for caption text
CAPTION_ABOVE_PX = 80    # how far above image rect to search (fallback)
CAPTION_MAX_LEN  = 300


def _extract_caption(page: fitz.Page, xref: int) -> str:
    """
    Try to find a caption for an image by looking at text directly below
    (or above) the image's bounding rect on the page.
    Returns empty string if nothing useful is found.
    """
    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return ""

    if not rects:
        return ""

    r = rects[0]

    # Check below first (most medical textbooks put captions under figures)
    below = fitz.Rect(r.x0 - 10, r.y1, r.x1 + 10, r.y1 + CAPTION_BELOW_PX)
    caption = page.get_text("text", clip=below).strip()

    # Fall back to checking above
    if len(caption) < 10:
        above = fitz.Rect(r.x0 - 10, r.y0 - CAPTION_ABOVE_PX, r.x1 + 10, r.y0)
        caption = page.get_text("text", clip=above).strip()

    if len(caption) > CAPTION_MAX_LEN:
        caption = caption[:CAPTION_MAX_LEN] + "…"

    return caption


def extract_images(pdf_path: str, collection: str) -> dict:
    """
    Extract images + captions from a PDF, save to disk, write index.json.

    Index format: {page_str: [{"path": str, "caption": str}, ...]}
    Skips image files already on disk. Always re-extracts captions.
    """
    out_dir    = os.path.join(IMAGES_DIR, collection)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.json")

    doc   = fitz.open(pdf_path)
    index: dict[str, list[dict]] = {}
    total = 0

    for page_num in range(len(doc)):
        page       = doc[page_num]
        raw_images = page.get_images(full=True)
        page_entries: list[dict] = []

        for img_idx, img in enumerate(raw_images):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue

            if base["width"] < MIN_WIDTH or base["height"] < MIN_HEIGHT:
                continue

            ext      = base["ext"]
            filename = f"page_{page_num + 1}_img_{img_idx}.{ext}"
            filepath = os.path.join(out_dir, filename)

            # Only write to disk if not already there
            if not os.path.exists(filepath):
                with open(filepath, "wb") as f:
                    f.write(base["image"])

            caption = _extract_caption(page, xref)
            page_entries.append({"path": filepath, "caption": caption})
            total += 1

        if page_entries:
            index[str(page_num + 1)] = page_entries

    doc.close()

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Extracted {total} images across {len(index)} pages → {out_dir}")
    return index


def reindex_captions(pdf_path: str, collection: str) -> dict:
    """
    Rebuild captions for an already-extracted image set without touching image files.
    Use this after changing caption extraction logic.
    """
    out_dir    = os.path.join(IMAGES_DIR, collection)
    index_path = os.path.join(out_dir, "index.json")

    if not os.path.exists(index_path):
        print(f"  No existing index at {index_path}. Run extract_images first.")
        return {}

    with open(index_path) as f:
        old_index = json.load(f)

    doc          = fitz.open(pdf_path)
    new_index: dict[str, list[dict]] = {}
    updated      = 0

    for page_str, entries in old_index.items():
        page_num = int(page_str) - 1
        if page_num >= len(doc):
            new_index[page_str] = entries
            continue

        page       = doc[page_num]
        raw_images = page.get_images(full=True)
        new_entries: list[dict] = []

        # Match existing entries to images by filename index
        for entry in entries:
            # entry may be old-format str or new-format dict
            if isinstance(entry, str):
                path = entry
                caption = ""
            else:
                path    = entry["path"]
                caption = entry.get("caption", "")

            # Re-extract caption from the corresponding image on this page
            filename  = os.path.basename(path)
            # filename: page_N_img_I.ext → extract img_idx
            try:
                img_idx = int(filename.split("_img_")[1].split(".")[0])
                if img_idx < len(raw_images):
                    xref    = raw_images[img_idx][0]
                    caption = _extract_caption(page, xref)
            except (IndexError, ValueError):
                pass

            new_entries.append({"path": path, "caption": caption})
            updated += 1

        new_index[page_str] = new_entries

    doc.close()

    with open(index_path, "w") as f:
        json.dump(new_index, f, indent=2)

    print(f"  Updated captions for {updated} images → {index_path}")
    return new_index


def lookup_images(pages: list, collection: str, window: int = 1) -> list[dict]:
    """
    Given retrieved page numbers and a collection name, return image entries.
    Each entry: {"path": str, "caption": str}

    Checks ±window pages to catch figures on adjacent pages.
    Returns [] if no image index exists.
    """
    index_path = os.path.join(IMAGES_DIR, collection, "index.json")
    if not os.path.exists(index_path):
        return []

    with open(index_path) as f:
        index = json.load(f)

    seen:   set[str]   = set()
    images: list[dict] = []

    for page in pages:
        if not isinstance(page, int):
            log.warning("lookup_images: skipping non-integer page metadata",
                        extra={"event": "image_lookup_bad_page", "page": page, "collection": collection})
            continue
        for p in range(page - window, page + window + 1):
            for entry in index.get(str(p), []):
                # Handle both old-format (str) and new-format (dict)
                if isinstance(entry, str):
                    entry = {"path": entry, "caption": ""}
                path = entry["path"]
                if path not in seen:
                    seen.add(path)
                    images.append(entry)

    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image index tools")
    parser.add_argument("--reindex", metavar="COLLECTION", help="Rebuild captions for a collection")
    parser.add_argument("--pdf", metavar="PDF_PATH", help="Path to the PDF (required for --reindex)")
    args = parser.parse_args()

    if args.reindex:
        if not args.pdf:
            print("--pdf is required with --reindex")
            sys.exit(1)
        reindex_captions(args.pdf, args.reindex)
    else:
        parser.print_help()
