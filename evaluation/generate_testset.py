"""
Generate a test set of Q&A pairs from the book using GPT-4o.
Samples random pages, generates realistic surgical questions + ground truth answers.
Saves to evaluation/testset.json
"""

import os
import json
import random
import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PDF_PATH = "data/raw/splitted_E_Christopher.pdf"
OUTPUT_PATH = "evaluation/testset.json"
NUM_QUESTIONS = 50
SEED = 42

GENERATION_PROMPT = """You are a surgical examiner creating MCQ-style and short-answer questions for PG surgery residents.

Given the following passage from Fischer's Mastery of Surgery (8th Edition), generate {n} high-quality questions that:
1. Can be answered directly and specifically from the passage
2. Are clinically relevant (not trivial facts)
3. Cover different aspects: indications, technique, complications, management decisions

For each question, provide:
- question: the question text
- ground_truth: the complete correct answer, grounded in the passage
- relevant_page: the page number provided

Return as a JSON array only, no other text.

Passage (Page {page}):
{text}
"""


def sample_rich_pages(pdf_path: str, n: int, seed: int) -> list[dict]:
    """Sample pages with substantial clinical content."""
    random.seed(seed)
    doc = fitz.open(pdf_path)
    candidates = []

    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        # Filter: good length, not TOC/index/references, has clinical keywords
        if (len(text) > 800
                and not text.startswith("CHAPTER") and len(text) < 4000
                and any(kw in text.lower() for kw in [
                    "patient", "surgery", "surgical", "management",
                    "treatment", "complication", "technique", "incision",
                    "dissection", "anastomosis", "resection", "repair"
                ])):
            candidates.append({"page": i + 1, "text": text})

    doc.close()
    sampled = random.sample(candidates, min(n * 3, len(candidates)))  # oversample, then filter
    return sampled


def generate_questions(pages: list[dict], target: int) -> list[dict]:
    client = OpenAI()
    all_questions = []

    print(f"Generating questions from {len(pages)} candidate pages...")

    for i, page_data in enumerate(pages):
        if len(all_questions) >= target:
            break

        n_per_page = 2 if len(all_questions) < target - 10 else 1

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": GENERATION_PROMPT.format(
                        n=n_per_page,
                        page=page_data["page"],
                        text=page_data["text"][:2000]
                    )
                }]
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            questions = json.loads(raw)

            for q in questions:
                q["page"] = page_data["page"]
                all_questions.append(q)

            print(f"  [{i+1}/{len(pages)}] Page {page_data['page']}: +{len(questions)} questions (total: {len(all_questions)})")

        except Exception as e:
            print(f"  [{i+1}] Page {page_data['page']}: skipped ({e})")
            continue

    return all_questions[:target]


if __name__ == "__main__":
    pages = sample_rich_pages(PDF_PATH, NUM_QUESTIONS, SEED)
    questions = generate_questions(pages, NUM_QUESTIONS)

    os.makedirs("evaluation", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved {len(questions)} questions to {OUTPUT_PATH}")
    print("\nSample question:")
    print(json.dumps(questions[0], indent=2))
