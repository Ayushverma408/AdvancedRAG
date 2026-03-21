# Advanced RAG - Medical Book Intelligence

## The Idea

My sister is a surgical resident doing MCh. She has notes from her AIIMS preparation for super-specialisation, the exam that comes after completing an MS Surgery PG degree. Those notes were going to get thrown away. I decided to back them up and turn them into embeddings so she (and anyone else) can quickly look up and revise anything.

For now I'm starting with a book she bought. Fischer's Mastery of Surgery, 8th Edition. All 6,000+ pages of it. Chunked, embedded, retrieval pipeline on top.

---

## Pipeline

Naive RAG doesn't work well on medical text. The vocabulary gap between how you ask a question and how a surgery textbook writes the answer is too wide. So the pipeline is:

```
HyDE -> Dense + Sparse (BM25) -> RRF -> Cross-Encoder Rerank -> GPT-4o
```

**HyDE** - instead of embedding your raw question, GPT-4o first writes a hypothetical passage the book would contain to answer it. That gets embedded and retrieved. Bridges the question-to-textbook vocabulary gap before retrieval even starts.

**Dense + Sparse in parallel** - semantic vectors (text-embedding-3-small) and BM25 keyword match run simultaneously. BM25 catches exact procedure names, eponyms, drug names that semantic search sometimes drifts past.

**RRF** - Reciprocal Rank Fusion merges three result lists (HyDE dense, original dense, BM25). Anything ranking highly across multiple lists floats up.

**Cross-Encoder Rerank** - ms-marco-MiniLM-L-6-v2 reads each (question, passage) pair jointly and rescores. Much more accurate than embedding similarity alone. Cuts to top 6 passages.

---

## RAGAS Evaluation

Six pipelines evaluated on a synthetic test set generated from the book. Metrics: **Faithfulness** (answer grounded in retrieved context), **Answer Relevancy** (actually answers the question), **Context Precision** (retrieved chunks were on-point), **Context Recall** (right chunks retrieved).

| Pipeline | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| **HyDE** | **0.931** | 0.826 | **0.773** | 0.867 |
| Rerank | 0.905 | **0.867** | 0.758 | 0.874 |
| Query Expansion | 0.856 | 0.805 | 0.756 | **0.917** |
| Hybrid | 0.892 | 0.793 | 0.669 | **0.917** |
| Naive | 0.883 | 0.811 | 0.613 | 0.734 |
| Decomposition | 0.832 | 0.840 | 0.730 | 0.828 |

HyDE is the default. Highest faithfulness (0.931) and context precision (0.773). For a surgical reference, those two matter the most. You don't want an answer that sounds right but isn't actually in the book.

A few things worth noting from the numbers:

- Naive gets surprisingly decent faithfulness (0.883) but falls apart on precision (0.613) and recall (0.734). It's retrieving chunks, just not the right ones consistently.
- Hybrid and Query Expansion both hit 0.917 recall (joint best) but precision is lower. Wide net, right answer somewhere in it, just with more noise around it.
- Rerank is close to HyDE on everything and has the best answer relevancy (0.867). If latency is a concern it's the better tradeoff.
- Decomposition has solid answer relevancy (0.840) but it's the slowest by far (~20s avg) and worst faithfulness (0.832). Breaking the question into sub-questions introduces drift in what gets retrieved.
- HyDE's precision gain (+0.015 over Rerank, +0.10 over naive) makes sense. When you generate the hypothetical doc first, retrieval is already operating in the book's vocabulary before any search happens.

**HyDE cost and latency on the test set:**
- avg latency: 7.98s, p95: 10.43s
- avg cost per query: $0.005, per 1k queries: $5.05

> ℹ️ Eval above was run on **Fischer only** (single-book, n=11). See multi-book results below.

---

## v2 — Multi-Book Pipeline (March 2026)

Expanded from 1 book to 4 surgical textbooks searched simultaneously:

| Collection | Book | Chunks |
|---|---|---|
| `fischer_surgery` | Fischer's Mastery of Surgery, 8th ed | ~36k |
| `sabiston_surgery` | Sabiston Textbook of Surgery, 22nd ed | ~25k |
| `shackelford_alimentary` | Shackelford's Surgery of the Alimentary Tract, 9th ed | ~19k |
| `blumgart_liver` | Blumgart's Surgery of the Liver, Biliary Tract and Pancreas | ~21k |

**Total: ~101k chunks, ~600MB vector store (ChromaDB), ~155-310MB BM25 in-memory index.**

### Multi-book retrieval architecture

```
Query
  │
  ├─ [HyDE] gpt-4o-mini → hypothetical passage   (~3-4s, optional)
  │
  ├─ Embed: query + hyp_doc concurrently          (2 API calls in parallel)
  │
  ├─ 4 books in parallel (ThreadPoolExecutor)
  │     └─ each book: dense(hyp_emb) + dense(query_emb) + BM25  → per-book RRF
  │
  ├─ Global RRF across all 4 books
  │
  └─ Cross-encoder rerank (ms-marco-MiniLM-L-6-v2) → top 6 chunks → GPT-4o
```

Key design decision: embeddings are pre-computed **before** threads start. Each thread does only local work (ChromaDB HNSW disk search + BM25 CPU) — zero OpenAI API calls inside threads. Retrieval time is `max(4 books)` not `sum(8 API calls)`. This dropped retrieval from ~90s to <1s for the search phase itself.

### Pipeline modes

Three modes exposed via the UI settings gear:

| Mode | What it does | Typical latency |
|---|---|---|
| 🏆 **HyDE** (default) | Generates hypothetical passage → 2 embeddings → full pipeline | ~8-12s |
| ⚡ **Fast** | Skips HyDE, embeds query only → full pipeline | ~4-6s |
| 🔓 **Free** | Bypasses RAG entirely, GPT-4o answers from its own knowledge | ~3-5s |

### Retrieval sub-timing breakdown

Every answer now shows a per-phase latency footer in the UI:

```
🟡 9.2s total  ·  📥 retrieval 4.1s (↳ HyDE 3.2s · embed 0.4s · search 0.3s · rerank 0.2s)  ·  🤖 LLM 5.1s  ·  🏆 HyDE
```

### Cross-encoder warmup

Cross-encoder model (ms-marco-MiniLM-L-6-v2) is loaded at API startup via `lifespan()` so the first user query pays zero cold-start cost (~4s saved on first hit).

### Multi-book RAGAS Evaluation (March 2026, quick mode — n=10, gpt-4o-mini scorer)

| Pipeline | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency | Cost/1k Q |
|---|---|---|---|---|---|---|
| **multi-book-hyde** | **0.9698** | 0.8855 | 0.9200 | **0.9667** | 7.2s | $6.28 |
| multi-book-fast | 0.9368 | 0.8768 | **0.9233** | **0.9667** | 7.4s | $6.38 |
| free (no RAG) | ⚠️ artifact | **0.8908** | ⚠️ artifact | ⚠️ artifact | **6.8s** | $6.26 |

> ⚠️ **Free mode retrieval metrics are not meaningful** — RAGAS computes context precision/recall/faithfulness against the retrieved contexts, and free mode returns no contexts. Those scores are RAGAS artefacts. Only answer relevancy is interpretable for free mode.

**vs. previous single-book HyDE** (Fischer only, n=11):

| | Single-book HyDE | Multi-book HyDE | Δ |
|---|---|---|---|
| Faithfulness | 0.931 | **0.9698** | +0.039 |
| Context Precision | 0.773 | **0.920** | **+0.147** |
| Context Recall | 0.867 | **0.967** | **+0.100** |
| Answer Relevancy | 0.826 | **0.886** | +0.060 |

**Analysis:**

The jump from single-book to multi-book is the headline result. Context precision went up **+0.147** — the biggest gain of any metric, and the one that matters most in production. When you search 4 books instead of 1, you're far more likely to retrieve the passage that actually answers the question rather than a loosely related one from the same chapter. Recall gained **+0.100** for the same reason: the right chunk now exists in the pool.

HyDE vs Fast is a tighter race than expected. Fast mode (no hypothetical doc generation) loses **0.033 on faithfulness** — that's the real cost of skipping HyDE. The LLM is slightly more likely to drift from the context when the retrieved passages are marginally less relevant. But it gains +0.003 on context precision, which is statistically noise at n=10. Latency difference is negligible (~0.2s) — almost certainly because OpenAI API variance dominates.

The latency gap between HyDE and Fast should be ~3-4s (cost of one gpt-4o-mini call for hypothetical doc). The fact that it shows as only 0.2s in this eval suggests the API was under light load and HyDE was fast. Under production load the gap will be more visible.

**Practical recommendation:** Use HyDE as default (0.9698 faithfulness — nearly hallucination-free). Offer Fast as explicit user choice when latency is critical. Free mode has no place in a clinical reference tool — answer relevancy looks good but there's no textbook grounding guarantee.

### Quick eval mode

```bash
python evaluation/evaluate.py --pipeline hyde --quick
# Runs 10 questions with gpt-4o-mini for RAGAS scoring — completes in ~2-3 min vs ~15 min full
```

### PDF page preview

- Each answer surfaces **📄 page preview buttons**, one per unique retrieved page
- Clicking any button opens a **gallery of all retrieved pages concurrently** (±1 pages for selected chunk, single page for the rest)
- Selected chunk text is **highlighted in yellow** directly on the PDF page (PyMuPDF `search_for` + `add_highlight_annot`; whitespace-normalised before matching)

### Extracted figure images

Figures extracted from PDFs at ingest time are **not shown by default**. A **🖼️ Show figures (N)** button appears when figures exist for the answer — click to load them on demand.

### Startup script

```bash
./medrag.sh        # kill existing, start API + Chainlit UI, open browser, start Cloudflare tunnel
./medrag.sh stop   # stop everything including tunnel
```

Cloudflare tunnel: `https://medrag.shuf.site` → `localhost:7860`
