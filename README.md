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
