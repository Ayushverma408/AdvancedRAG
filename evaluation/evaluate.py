"""
RAGAS evaluation of a RAG pipeline.
Usage:
    python evaluation/evaluate.py --pipeline naive
    python evaluation/evaluate.py --pipeline hybrid
    python evaluation/evaluate.py --compare
Saves results to evaluation/results_{pipeline}.json and prints a score table.
"""

import sys
import json
import argparse
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chain import build_chain


def load_testset(path: str = "evaluation/testset.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_pipeline(testset: list[dict], chain, retriever) -> dict:
    """Run each question through the RAG pipeline and collect inputs/outputs."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(testset):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i+1}/{len(testset)}] {q[:80]}...")
        try:
            answer = chain.invoke(q)
            docs = retriever.invoke(q)
            ctx = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"    ERROR: {e}")
            answer = ""
            ctx = []

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def safe_mean(val) -> float:
    """Handle both scalar and list score formats across RAGAS versions."""
    if isinstance(val, (list, np.ndarray)):
        clean = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return round(float(np.mean(clean)) if clean else 0.0, 4)
    return round(float(val), 4)


def run_evaluation(pipeline_name: str, testset_path: str = "evaluation/testset.json"):
    print(f"\n{'='*60}")
    print(f"Evaluating pipeline: {pipeline_name.upper()}")
    print(f"{'='*60}")

    testset = load_testset(testset_path)
    print(f"Loaded {len(testset)} test questions")

    print("\nRunning pipeline...")
    chain, retriever = build_chain(mode=pipeline_name)
    results = run_pipeline(testset, chain, retriever)

    print("\nRunning RAGAS scoring...")
    dataset = Dataset.from_dict(results)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
    )

    score_dict = {
        "pipeline": pipeline_name,
        "faithfulness": safe_mean(scores["faithfulness"]),
        "answer_relevancy": safe_mean(scores["answer_relevancy"]),
        "context_precision": safe_mean(scores["context_precision"]),
        "context_recall": safe_mean(scores["context_recall"]),
    }

    output = {
        "scores": score_dict,
        "per_question": [
            {
                "question": results["question"][i],
                "answer": results["answer"][i],
                "ground_truth": results["ground_truth"][i],
                "contexts_retrieved": len(results["contexts"][i]),
            }
            for i in range(len(results["question"]))
        ]
    }

    out_path = f"evaluation/results_{pipeline_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {pipeline_name.upper()}")
    print(f"{'='*60}")
    print(f"  Faithfulness      : {score_dict['faithfulness']:.4f}  (no hallucination)")
    print(f"  Answer Relevancy  : {score_dict['answer_relevancy']:.4f}  (answers the question)")
    print(f"  Context Precision : {score_dict['context_precision']:.4f}  (retrieved chunks are relevant)")
    print(f"  Context Recall    : {score_dict['context_recall']:.4f}  (right chunks retrieved)")
    print(f"\nFull results saved to: {out_path}")

    return score_dict


def compare_results():
    """Print a comparison table of all evaluated pipelines."""
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

    print(f"\n{'Pipeline':<20} {'Faithfulness':>14} {'Relevancy':>12} {'Precision':>12} {'Recall':>10}")
    print("-" * 72)
    for s in all_scores:
        print(f"{s['pipeline']:<20} {s['faithfulness']:>14.4f} {s['answer_relevancy']:>12.4f} {s['context_precision']:>12.4f} {s['context_recall']:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="naive", help="Pipeline name (naive, hybrid, rerank, etc.)")
    parser.add_argument("--compare", action="store_true", help="Show comparison table of all results")
    parser.add_argument("--testset", default="evaluation/testset.json")
    args = parser.parse_args()

    if args.compare:
        compare_results()
    else:
        run_evaluation(args.pipeline, args.testset)
