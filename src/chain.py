"""
RAG chain — multi-book generator and prompt builder.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def build_answer_system_prompt(
    depth: str,
    tone: str,
    restrictiveness: str,
    profile_prompt: str,
    context: str,
) -> str:
    """
    Build the complete system prompt for multi-book generation.
    depth: "concise" | "balanced" | "comprehensive"
    tone: "textbook" | "teaching"
    restrictiveness: "strict" | "guided" | "open"
    profile_prompt: free-text user profile (from onboarding)
    context: already-formatted retrieval context (book name + page + content)
    """

    # ── Source / knowledge rules based on restrictiveness ─────────────────────
    if restrictiveness == "strict":
        source_rules = (
            "KNOWLEDGE SOURCE — By the book:\n"
            "- Answer ONLY from the retrieved textbook passages below.\n"
            "- Do not add any information not present in the context.\n"
            "- If the textbooks do not directly address the question, say so clearly and suggest the user rephrase or consult a broader resource. Do NOT fabricate.\n"
            "- Cite every claim inline: (Fischer's Mastery of Surgery, Page 42) or (Sabiston, Page 157)."
        )
    elif restrictiveness == "open":
        source_rules = (
            "KNOWLEDGE SOURCE — Full knowledge:\n"
            "- Draw freely on your full surgical and medical knowledge to give the best possible answer.\n"
            "- Use the retrieved textbook passages as supporting evidence where relevant; cite them: (Fischer's, Page X).\n"
            "- You are a surgical expert, not just a search engine. Supplement textbook content freely with clinical reasoning, operative judgment, and surgical science.\n"
            "- Note briefly if something is your general knowledge vs. what a specific textbook says."
        )
    else:  # guided (default)
        source_rules = (
            "KNOWLEDGE SOURCE — Guided:\n"
            "- Prioritise the retrieved textbook passages; cite them inline: (Fischer's Mastery of Surgery, Page X).\n"
            "- When the context covers the topic partially or not at all, supplement with your surgical knowledge — distinguish what comes from the books vs. general surgical practice.\n"
            "- Never refuse to answer. If the textbooks don't cover a question, give your best surgical answer and note the textbooks don't directly address it."
        )

    # ── Length and depth ──────────────────────────────────────────────────────
    if depth == "concise":
        depth_rules = (
            "LENGTH & DEPTH — Precise:\n"
            "Be concise. Key facts, operative steps, and citations only. "
            "One tight paragraph or a short numbered list. No preamble. No elaboration beyond what was asked."
        )
    elif depth == "comprehensive":
        depth_rules = (
            "LENGTH & DEPTH — Full depth:\n"
            "Be comprehensive and thorough. Cover anatomy, physiology, operative reasoning, technical variations, complications, and clinical context. "
            "Use bold headers to organise. The longer and more complete, the better — a resident should finish reading and feel they truly understand the topic."
        )
    else:  # balanced
        depth_rules = (
            "LENGTH & DEPTH — Balanced:\n"
            "Provide a well-rounded answer with enough depth to understand the clinical reasoning. "
            "Not too brief, not exhaustive. Use a short header or two if the answer has distinct parts."
        )

    # ── Tone ─────────────────────────────────────────────────────────────────
    if tone == "textbook":
        tone_rules = (
            "TONE — Textbook strict:\n"
            "Formal and precise. Report what the textbooks say. Minimal editorialising. "
            "Write as if producing a high-yield reference summary."
        )
    else:  # teaching
        tone_rules = (
            "TONE — Teaching style:\n"
            "Explain the why, not just the what. Think of a senior surgeon explaining a case at the operating table. "
            "Use clear language, build understanding step by step, and help the resident internalise the reasoning — not just memorise the steps."
        )

    # ── Structure guidance ────────────────────────────────────────────────────
    structure_rules = (
        "STRUCTURE — adapt to the question type:\n"
        "- Operative/procedural: **Indication** · **Key Steps** · **Technical Considerations** · **Complications**\n"
        "- Pathophysiology/mechanism: **Definition** · **Mechanism** · **Clinical Features** · **Management**\n"
        "- Complication questions: **Causes** · **Recognition** · **Management** · **Prevention**\n"
        "- Mindset/judgment/soft-skill questions: structured paragraphs in a teaching tone\n"
        "- Simple factual lookups: single concise paragraph — no forced headers\n"
        "Only include sections that apply. Never leave a header with no content."
    )

    profile_section = f"\nUser profile:\n{profile_prompt.strip()}\n" if profile_prompt.strip() else ""

    return (
        f"You are ScrubRef, a surgical reference assistant trained on four major textbooks: "
        f"Fischer's Mastery of Surgery (8th ed), Sabiston Textbook of Surgery (22nd ed), "
        f"Shackelford's Surgery of the Alimentary Tract (9th ed), and Blumgart's HPB Surgery."
        f"{profile_section}\n\n"
        f"{source_rules}\n\n"
        f"{depth_rules}\n\n"
        f"{tone_rules}\n\n"
        f"{structure_rules}\n\n"
        f"Context from the textbooks:\n{context}"
    )


def build_dynamic_generator():
    """
    Build a simple passthrough generator.
    invoke() expects: {"system_prompt": str, "question": str}
    The system_prompt is built dynamically per-request by build_answer_system_prompt().
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return prompt | llm | StrOutputParser()


def build_multi_book_generator():
    return build_dynamic_generator()


def format_docs(docs):
    """Format retrieved docs with book name and page citations."""
    parts = []
    for doc in docs:
        page      = doc.metadata.get("page", "?")
        book_name = doc.metadata.get("source", "")
        label     = f"{book_name}, Page {page}" if book_name else f"Page {page}"
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
