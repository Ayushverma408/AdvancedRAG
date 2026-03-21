"""
RAGAS evaluation of a RAG pipeline.
Tracks quality metrics, latency, and estimated cost per query.

Usage:
    python evaluation/evaluate.py --pipeline naive
    python evaluation/evaluate.py --pipeline hybrid
    python evaluation/evaluate.py --pipeline rerank
    python evaluation/evaluate.py --compare

Quick mode (< 3 min, uses 10 questions + gpt-4o-mini for RAGAS scoring):
    python evaluation/evaluate.py --pipeline hyde --quick
"""

import sys
import json
import time
import argparse
import warnings
import numpy as np
import httpx
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Seconds to sleep between questions — prevents TPM rate limit when running
# multiple LLM calls per question (HyDE/decomposition generate 4-5 calls each)
INTER_QUESTION_SLEEP = 2

# Quick-mode defaults: smaller testset + cheaper/faster model for RAGAS scoring
QUICK_N_QUESTIONS  = 10
QUICK_RAGAS_MODEL  = "gpt-4o-mini"   # ~10x cheaper, ~5x faster than gpt-4o
FULL_RAGAS_MODEL   = "gpt-4o"

# Multi-book pipeline modes — these call the running API instead of build_chain()
MULTI_BOOK_MODES = {"multi-book-hyde", "multi-book-fast", "free"}
API_URL = "http://localhost:8000"

from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, "src")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chain import build_chain

# OpenAI pricing (update if rates change)
GPT4O_INPUT_PER_TOKEN  = 2.50 / 1_000_000   # $2.50 per 1M input tokens
GPT4O_OUTPUT_PER_TOKEN = 10.0 / 1_000_000   # $10.00 per 1M output tokens
EMBED_PER_TOKEN        = 0.02 / 1_000_000   # $0.02 per 1M tokens (text-embedding-3-small)
AVG_TOKENS_PER_CHAR    = 0.25               # rough approximation


def estimate_query_cost(question: str, context_chunks: list[str], answer: str) -> dict:
    """Estimate OpenAI API cost for a single query."""
    # Rough token counts
    system_prompt_tokens = 200
    question_tokens = len(question) * AVG_TOKENS_PER_CHAR
    context_tokens = sum(len(c) for c in context_chunks) * AVG_TOKENS_PER_CHAR
    answer_tokens = len(answer) * AVG_TOKENS_PER_CHAR
    embed_tokens = len(question) * AVG_TOKENS_PER_CHAR

    input_tokens = system_prompt_tokens + question_tokens + context_tokens
    output_tokens = answer_tokens

    cost = (
        input_tokens * GPT4O_INPUT_PER_TOKEN
        + output_tokens * GPT4O_OUTPUT_PER_TOKEN
        + embed_tokens * EMBED_PER_TOKEN
    )
    return {
        "input_tokens": round(input_tokens),
        "output_tokens": round(output_tokens),
        "cost_usd": round(cost, 6),
    }


def load_testset(path: str = "evaluation/testset.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _invoke_with_retry(chain, retriever, question: str, max_retries: int = 5):
    """Invoke chain+retriever with exponential backoff on rate limit errors."""
    delay = 5
    for attempt in range(max_retries):
        try:
            answer = chain.invoke(question)
            docs = retriever.invoke(question)
            ctx = [doc.page_content for doc in docs]
            return answer, ctx
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower():
                wait = delay * (2 ** attempt)
                print(f"    Rate limit hit — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    ERROR: {e}")
                return "", []
    print(f"    Failed after {max_retries} retries")
    return "", []


def run_pipeline_api(testset: list[dict], pipeline_name: str) -> dict:
    """Run each question through the live API — used for multi-book-hyde, multi-book-fast, free."""
    free_mode = (pipeline_name == "free")
    use_hyde  = (pipeline_name == "multi-book-hyde")

    questions, answers, contexts, ground_truths = [], [], [], []
    latencies, costs = [], []

    for i, item in enumerate(testset):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1}/{len(testset)}] {q[:75]}...")

        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                f"{API_URL}/query",
                json={"question": q, "free_mode": free_mode, "use_hyde": use_hyde},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]
            ctx    = [c["content"] for c in data.get("chunks", [])]
        except Exception as e:
            print(f"    API error: {e}")
            answer, ctx = "", []
        latency = time.perf_counter() - t0

        if i < len(testset) - 1:
            time.sleep(INTER_QUESTION_SLEEP)

        cost_info = estimate_query_cost(q, ctx, answer)
        latencies.append(round(latency, 3))
        costs.append(cost_info["cost_usd"])

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "latencies": latencies,
        "costs": costs,
    }


def run_pipeline(testset: list[dict], chain, retriever) -> dict:
    """Run each question through the RAG pipeline, tracking latency and cost."""
    questions, answers, contexts, ground_truths = [], [], [], []
    latencies, costs = [], []

    for i, item in enumerate(testset):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1}/{len(testset)}] {q[:75]}...")

        t0 = time.perf_counter()
        answer, ctx = _invoke_with_retry(chain, retriever, q)
        latency = time.perf_counter() - t0

        if i < len(testset):
            time.sleep(INTER_QUESTION_SLEEP)

        cost_info = estimate_query_cost(q, ctx, answer)
        latencies.append(round(latency, 3))
        costs.append(cost_info["cost_usd"])

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "latencies": latencies,
        "costs": costs,
    }


def safe_mean(val) -> float:
    if isinstance(val, (list, np.ndarray)):
        clean = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return round(float(np.mean(clean)) if clean else 0.0, 4)
    return round(float(val), 4)


def run_evaluation(
    pipeline_name: str,
    testset_path: str = "evaluation/testset.json",
    quick: bool = False,
    ragas_model: str | None = None,
):
    print(f"\n{'='*60}")
    print(f"Evaluating pipeline: {pipeline_name.upper()}")
    if quick:
        print(f"Mode: QUICK ({QUICK_N_QUESTIONS} questions, RAGAS={QUICK_RAGAS_MODEL})")
    print(f"{'='*60}")

    testset = load_testset(testset_path)
    if quick:
        # Use first N questions for a fast representative sample
        testset = testset[:QUICK_N_QUESTIONS]
    print(f"Loaded {len(testset)} test questions")

    print("\nRunning pipeline...")
    if pipeline_name in MULTI_BOOK_MODES:
        print(f"  → Using live API at {API_URL}")
        results = run_pipeline_api(testset, pipeline_name)
    else:
        chain, retriever = build_chain(mode=pipeline_name)
        results = run_pipeline(testset, chain, retriever)

    # Choose RAGAS scoring model: explicit arg > quick default > full default
    _ragas_model = ragas_model or (QUICK_RAGAS_MODEL if quick else FULL_RAGAS_MODEL)
    print(f"\nRunning RAGAS scoring (model={_ragas_model})...")
    dataset = Dataset.from_dict({
        "question": results["question"],
        "answer": results["answer"],
        "contexts": results["contexts"],
        "ground_truth": results["ground_truth"],
    })

    llm = ChatOpenAI(model=_ragas_model, temperature=0)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
    )

    latencies = results["latencies"]
    costs = results["costs"]

    score_dict = {
        "pipeline": pipeline_name,
        # Quality
        "faithfulness": safe_mean(scores["faithfulness"]),
        "answer_relevancy": safe_mean(scores["answer_relevancy"]),
        "context_precision": safe_mean(scores["context_precision"]),
        "context_recall": safe_mean(scores["context_recall"]),
        # Performance
        "avg_latency_s": round(float(np.mean(latencies)), 3),
        "p95_latency_s": round(float(np.percentile(latencies, 95)), 3),
        "avg_cost_usd": round(float(np.mean(costs)), 6),
        "cost_per_1k_queries_usd": round(float(np.mean(costs)) * 1000, 4),
    }

    output = {
        "scores": score_dict,
        "per_question": [
            {
                "question": results["question"][i],
                "answer": results["answer"][i],
                "ground_truth": results["ground_truth"][i],
                "latency_s": results["latencies"][i],
                "cost_usd": results["costs"][i],
                "contexts_retrieved": len(results["contexts"][i]),
            }
            for i in range(len(results["question"]))
        ]
    }

    suffix = "_quick" if quick else ""
    out_path = f"evaluation/results_{pipeline_name}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {pipeline_name.upper()}{' (QUICK)' if quick else ''}")
    print(f"{'='*60}")
    print(f"  Quality:")
    print(f"    Faithfulness      : {score_dict['faithfulness']:.4f}  (no hallucination)")
    print(f"    Answer Relevancy  : {score_dict['answer_relevancy']:.4f}  (answers the question)")
    print(f"    Context Precision : {score_dict['context_precision']:.4f}  (retrieved chunks are relevant)")
    print(f"    Context Recall    : {score_dict['context_recall']:.4f}  (right chunks retrieved)")
    print(f"  Performance:")
    print(f"    Avg latency       : {score_dict['avg_latency_s']:.3f}s")
    print(f"    P95 latency       : {score_dict['p95_latency_s']:.3f}s")
    print(f"    Cost per query    : ${score_dict['avg_cost_usd']:.6f}")
    print(f"    Cost per 1k Q     : ${score_dict['cost_per_1k_queries_usd']:.4f}")
    print(f"\nFull results saved to: {out_path}")

    return score_dict


def compare_results():
    import glob
    result_files = glob.glob("evaluation/results_*.json")
    if not result_files:
        print("No results found yet.")
        return

    all_scores = []
    for f in sorted(result_files):
        with open(f) as fh:
            data = json.load(fh)
            all_scores.append(data["scores"])

    # Quality table
    print(f"\n{'─'*80}")
    print("QUALITY METRICS")
    print(f"{'─'*80}")
    print(f"{'Pipeline':<12} {'Faithful':>10} {'Relevancy':>11} {'Precision':>11} {'Recall':>9}")
    print(f"{'─'*80}")
    for s in all_scores:
        print(f"{s['pipeline']:<12} {s['faithfulness']:>10.4f} {s['answer_relevancy']:>11.4f} {s['context_precision']:>11.4f} {s['context_recall']:>9.4f}")

    # Performance table (only if latency/cost data exists)
    if "avg_latency_s" in all_scores[0]:
        print(f"\n{'─'*80}")
        print("PERFORMANCE & COST (per query)")
        print(f"{'─'*80}")
        print(f"{'Pipeline':<12} {'Avg Lat':>10} {'P95 Lat':>10} {'Cost/Q':>12} {'Cost/1kQ':>12}")
        print(f"{'─'*80}")
        for s in all_scores:
            if "avg_latency_s" in s:
                print(f"{s['pipeline']:<12} {s['avg_latency_s']:>9.3f}s {s['p95_latency_s']:>9.3f}s"
                      f" ${s['avg_cost_usd']:>10.6f} ${s['cost_per_1k_queries_usd']:>10.4f}")
            else:
                print(f"{s['pipeline']:<12} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="naive",
                        help="Pipeline name (naive, hybrid, rerank, query_expansion, etc.)")
    parser.add_argument("--compare", action="store_true", help="Show comparison table")
    parser.add_argument("--testset", default="evaluation/testset.json")
    parser.add_argument(
        "--quick", action="store_true",
        help=f"Fast mode: only {QUICK_N_QUESTIONS} questions, RAGAS scored with {QUICK_RAGAS_MODEL}. "
             f"Completes in ~2-3 min vs ~15 min for full run.",
    )
    parser.add_argument(
        "--ragas-model", default=None,
        help=f"Override the model used for RAGAS scoring (default: {FULL_RAGAS_MODEL}, quick default: {QUICK_RAGAS_MODEL})",
    )
    args = parser.parse_args()

    if args.compare:
        compare_results()
    else:
        run_evaluation(args.pipeline, args.testset, quick=args.quick, ragas_model=args.ragas_model)
