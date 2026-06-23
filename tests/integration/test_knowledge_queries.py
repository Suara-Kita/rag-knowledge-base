"""
RAG query-quality tests against the ingested Johor/Malaysia economics document.

Two suites:
  - IN-SCOPE  : questions the knowledge base should answer with substance and citations.
  - OUT-OF-SCOPE: questions whose answers are NOT in the document — the LLM must
                  admit it rather than hallucinate.

Requires:
  - Running Neo4j with the Johor economics document already ingested.
  - OPENAI_API_KEY set (skipped otherwise).

Run with:
  pytest tests/integration/test_knowledge_queries.py -v -m integration
"""

import os
from pathlib import Path

import pytest

from src.config import settings
from src.db.neo4j import get_driver
from src.knowledge.retriever import build_filtered_rag, search
from src.knowledge.works_cited import extract_works_cited

pytestmark = pytest.mark.integration

_DOC_FILTER = "Perbezaan Malaysia dan Johor"

# Phrases that indicate the LLM is correctly admitting a lack of context.
# The model may respond in Malay or English, so we cover both.
_NO_CONTEXT_SIGNALS = [
    "tidak ada maklumat",
    "tiada maklumat",
    "tidak dapat",
    "tidak terdapat",
    "tidak dinyatakan",
    "saya tidak mempunyai",
    "saya tidak dapat",
    "tiada",
    "maaf",
    "i don't have",
    "i do not have",
    "no information",
    "not mentioned",
    "not covered",
    "context does not",
    "context provided does not",
    "unable to",
    "cannot answer",
    "not available",
    "not found",
    "outside the scope",
    "please visit",        # LLM redirects to an external source
    "sila lawati",
    "hubungi",
]


# ── shared fixture ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def johor_rag():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    processed_dir = Path(settings.processed_dir).resolve()
    matches = sorted(
        (md for md in processed_dir.glob("*.md") if _DOC_FILTER in md.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        pytest.skip(
            f"No ingested document matching '{_DOC_FILTER}' found in {processed_dir}. "
            "Run the watcher to ingest the document first."
        )

    works_cited = extract_works_cited(matches[0])
    return build_filtered_rag(get_driver(), _DOC_FILTER, works_cited)


# ── in-scope questions ────────────────────────────────────────────────────────
# Each entry: (question, keywords_at_least_one_must_appear, reason)

_IN_SCOPE = [
    (
        "Berapakah kadar pertumbuhan KDNK Johor pada tahun 2023?",
        ["4.1", "4.0", "johor", "kdnk", "pertumbuhan", "%"],
        "Johor 2023 GDP growth rate is explicitly stated in the document",
    ),
    (
        "Bagaimana pertumbuhan KDNK Johor berbanding Malaysia pada 2023?",
        ["johor", "malaysia", "pertumbuhan", "4.1", "3.5", "3.6"],
        "Document directly compares Johor vs national GDP growth",
    ),
    (
        "Apakah sektor ekonomi utama yang menyumbang kepada KDNK Johor?",
        ["perkhidmatan", "pembuatan", "pembinaan", "sektor"],
        "Sectoral breakdown is a core section of the document",
    ),
    (
        "Apakah itu JS-SEZ dan bagaimana ia memberi kesan kepada ekonomi Johor?",
        ["js-sez", "johor-singapore", "zon", "ekonomi", "singapore"],
        "JS-SEZ is discussed in detail in the policy section",
    ),
    (
        "Berapa nilai pelaburan pusat data yang diumumkan di Johor?",
        ["data", "pelaburan", "bilion", "pusat"],
        "Data centre investment figures are cited in the document",
    ),
    (
        "Apakah unjuran KDNK nominal Johor untuk tahun 2025 dan 2026?",
        ["2025", "2026", "bilion", "unjuran"],
        "GDP projections for 2025-2026 appear in the document",
    ),
    (
        "Berapa sumbangan Johor kepada KDNK nasional Malaysia?",
        ["9", "%", "sumbangan", "nasional", "johor"],
        "Johor's share of national GDP is stated in the document",
    ),
    (
        "Bagaimana FDI Johor berbanding negeri-negeri lain di Malaysia?",
        ["fdi", "pelaburan", "johor", "negeri"],
        "FDI comparison between states is covered in the document",
    ),
    (
        "Apakah impak NIMP terhadap sektor pembuatan Malaysia?",
        ["nimp", "pembuatan", "industri", "malaysia"],
        "NIMP (National Industrial Master Plan) is referenced in the document",
    ),
    (
        "Apakah kadar inflasi di Johor berbanding purata nasional?",
        ["inflasi", "johor", "malaysia", "%"],
        "Inflation data for Johor vs national is in the document",
    ),
]


@pytest.mark.parametrize(
    "question,keywords,reason",
    _IN_SCOPE,
    ids=[q[0][:50] for q in _IN_SCOPE],
)
def test_in_scope_returns_substantive_answer(johor_rag, question, keywords, reason):
    """In-scope questions must return an answer with substance and at least one expected keyword."""
    answer = search(johor_rag, question, top_k=8)

    assert isinstance(answer, str), "Answer must be a string"
    assert len(answer) > 80, (
        f"Answer too short for in-scope question.\nReason: {reason}\nGot: {answer!r}"
    )

    answer_lower = answer.lower()
    matched = [kw for kw in keywords if kw.lower() in answer_lower]
    assert matched, (
        f"{reason}\n"
        f"Expected at least one of {keywords} in the answer.\n"
        f"Answer (first 400 chars): {answer[:400]}"
    )


def test_in_scope_answer_includes_references_section(johor_rag):
    """Answers to in-scope questions should include citation evidence.

    Accepts either a '## References' section (multi-fact answer) or at least one
    inline citation marker [n] / 【n】 (single-fact answer where the LLM omits
    the section heading but still cites inline).
    """
    import re

    question = "Berapakah KDNK nominal Johor pada 2024?"
    answer = search(johor_rag, question, top_k=8)

    has_references_heading = "## References" in answer or "references" in answer.lower()
    has_inline_citation = bool(re.search(r"[\[【]\d+[\]】]", answer))

    assert has_references_heading or has_inline_citation, (
        "Expected a '## References' section or at least one inline citation [n] / 【n】.\n"
        f"Got: {answer[:400]}"
    )


def test_in_scope_answer_contains_inline_citation(johor_rag):
    """Inline citation markers like [9] or [19] should appear in the answer body."""
    question = "Berapa jumlah KDNK Johor pada 2024?"
    answer = search(johor_rag, question, top_k=8)

    import re
    # LLMs may use standard [9] or Unicode【9】citation brackets
    has_citation = bool(re.search(r"[\[【]\d+[\]】]", answer))
    assert has_citation, (
        "Expected at least one inline citation like [9] in the answer.\n"
        f"Got: {answer[:400]}"
    )


# ── out-of-scope questions ────────────────────────────────────────────────────
# Each entry: (question, reason_why_not_in_document)

_OUT_OF_SCOPE = [
    (
        "Apakah resipi nasi lemak yang paling sedap di Malaysia?",
        "Food recipe — not related to economics",
    ),
    (
        "Siapakah pemenang Piala Dunia FIFA 2022?",
        "Sports result — not in any ingested document",
    ),
    (
        "Apakah nombor telefon pejabat Jabatan Perangkaan Malaysia di Johor Bahru?",
        "Specific office contact — operational data not in the document",
    ),
    (
        "Apakah KDNK negara Iceland pada tahun 2023?",
        "Iceland economy — not covered, document is Malaysia/Johor only",
    ),
    (
        "Berapakah harga semasa Bitcoin dalam USD?",
        "Cryptocurrency — not mentioned in the document",
    ),
    (
        "Siapakah Perdana Menteri Malaysia yang pertama?",
        "Historical politics — not in the economics document",
    ),
    (
        "Apakah ubat yang digunakan untuk merawat diabetes?",
        "Medical question — completely unrelated topic",
    ),
    (
        "Berikan saya jadual kereta api KTM dari KL ke Johor Bahru.",
        "Train schedule — operational data not in the document",
    ),
]


@pytest.mark.parametrize(
    "question,reason",
    _OUT_OF_SCOPE,
    ids=[q[0][:50] for q in _OUT_OF_SCOPE],
)
def test_out_of_scope_does_not_hallucinate(johor_rag, question, reason):
    """Out-of-scope questions must not produce a confident hallucinated answer.

    The LLM should either:
    - Explicitly admit it lacks context (detected via _NO_CONTEXT_SIGNALS), or
    - Return a very short response indicating no useful retrieval occurred.
    """
    answer = search(johor_rag, question, top_k=5)

    assert isinstance(answer, str), "Answer must be a string"

    answer_lower = answer.lower()
    has_no_context_signal = any(signal in answer_lower for signal in _NO_CONTEXT_SIGNALS)
    is_very_short = len(answer.strip()) < 150

    assert has_no_context_signal or is_very_short, (
        f"Out-of-scope question got a confident answer — possible hallucination.\n"
        f"Reason why it's out-of-scope: {reason}\n"
        f"Answer: {answer[:500]}"
    )
