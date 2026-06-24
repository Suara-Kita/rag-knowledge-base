"""
RAG query-quality tests against the 9 Johor policy/economy research documents
ingested via the watch-johor-research/ source (see WATCH_DIR in .env).

Unlike test_knowledge_queries.py (which filters to a single document), this
suite queries the unfiltered knowledge base — answers may draw on any
ingested document, mirroring how the MCP query_knowledge tool is actually used.

Requires:
  - Running Neo4j with the 9 Johor research documents already ingested.
  - OPENAI_API_KEY set (skipped otherwise).

Run with:
  pytest tests/integration/test_johor_research_queries.py -v -m integration
"""

import os
from pathlib import Path

import pytest

from src.config import settings
from src.db.neo4j import get_driver
from src.knowledge.retriever import build_rag, search

pytestmark = pytest.mark.integration

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
    "please visit",
    "sila lawati",
    "hubungi",
]


@pytest.fixture(scope="module")
def johor_research_rag():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    processed_dir = Path(settings.processed_dir).resolve()
    expected_docs = [
        "Analisis Makroekonomi Perbandingan Malaysia dan Johor.md",
        "Laporan Impak JS-SEZ Kepada Johor.md",
    ]
    missing = [d for d in expected_docs if not (processed_dir / d).exists()]
    if missing:
        pytest.skip(
            f"Johor research documents not yet ingested (missing {missing} in {processed_dir}). "
            "Run the watcher against watch-johor-research/ first."
        )

    return build_rag(get_driver())


# ── in-scope questions ────────────────────────────────────────────────────────
# Each entry: (question, keywords_at_least_one_must_appear, reason)

_IN_SCOPE = [
    (
        "Berapa peratus pertumbuhan KDNK Malaysia pada tahun 2022?",
        ["8.7", "%"],
        "Malaysia 2022 GDP growth of 8.7% is stated in the macroeconomic comparison report",
    ),
    (
        "Berapakah jumlah pelaburan diluluskan di Johor bagi sembilan bulan pertama 2025?",
        ["91.1", "bilion", "johor"],
        "Johor's 9M2025 approved investment of RM91.1 billion is stated in the investment performance report",
    ),
    (
        "Apakah sasaran pekerjaan baharu JETP menjelang tahun 2030?",
        ["200,000", "200000", "jetp", "pekerjaan"],
        "JETP's target of 200,000 new jobs by 2030 is stated in the human capital report",
    ),
    (
        "Berapakah kadar pengangguran di Johor pada tahun 2024?",
        ["2.3", "pengangguran"],
        "Johor's 2024 unemployment rate of 2.3% is stated in the social safety net report",
    ),
    (
        "Berapakah jumlah pelaburan terkumpul di Johor sehingga akhir 2025?",
        ["218", "bilion"],
        "Cumulative approved investment of RM218 billion is stated in the infrastructure report",
    ),
    (
        "Bilakah Dewan Negeri Johor dibubarkan dan bilakah PRN Johor dijangka diadakan?",
        ["1 jun", "julai 2026", "prn"],
        "DNJ dissolution date and PRN Johor ke-16 timeline are stated in the Bantuan Kasih Johor 2.0 report",
    ),
    (
        "Bilakah perjanjian JS-SEZ ditandatangani dan oleh siapa?",
        ["7 januari 2025", "anwar", "lawrence wong"],
        "JS-SEZ signing date and signatories are stated in the JS-SEZ impact report",
    ),
    (
        "Berapa peratus sumbangan JS-SEZ kepada jumlah pelaburan diluluskan Johor pada 2025?",
        ["74.6", "68", "bilion"],
        "JS-SEZ's 74.6% / RM68 billion contribution share is stated in the JS-SEZ impact report",
    ),
    (
        "Berapakah hasil negeri Johor yang dikutip pada tahun 2025?",
        ["2.67", "2.676", "bilion", "hasil"],
        "Johor's record 2025 state revenue is stated in the fiscal performance report",
    ),
]


@pytest.mark.parametrize(
    "question,keywords,reason",
    _IN_SCOPE,
    ids=[q[0][:50] for q in _IN_SCOPE],
)
def test_in_scope_returns_substantive_answer(johor_research_rag, question, keywords, reason):
    """In-scope questions must return an answer with substance and at least one expected keyword."""
    answer = search(johor_research_rag, question, top_k=8)

    assert isinstance(answer, str), "Answer must be a string"
    # 40 chars is enough to filter out empty/near-empty responses without
    # penalizing a correct, terse, single-fact answer (e.g. "X ialah 2.3%").
    assert len(answer) > 40, (
        f"Answer too short for in-scope question.\nReason: {reason}\nGot: {answer!r}"
    )

    answer_lower = answer.lower()
    matched = [kw for kw in keywords if kw.lower() in answer_lower]
    assert matched, (
        f"{reason}\n"
        f"Expected at least one of {keywords} in the answer.\n"
        f"Answer (first 400 chars): {answer[:400]}"
    )


# ── out-of-scope questions ────────────────────────────────────────────────────
# Topics not covered by any of the 9 Johor research documents.

_OUT_OF_SCOPE = [
    (
        "Apakah resipi rendang daging yang paling sedap?",
        "Food recipe — not related to any ingested document",
    ),
    (
        "Berapakah harga emas dunia hari ini?",
        "Commodity price — not in any ingested document",
    ),
    (
        "Apakah keputusan perlawanan bola sepak Liga Super Malaysia minggu lepas?",
        "Sports result — not in any ingested document",
    ),
    (
        "Apakah KDNK negara Vietnam pada tahun 2024?",
        "Vietnam economy — not covered, documents are Johor/Malaysia-specific",
    ),
    (
        "Bagaimana cara menempah pasport antarabangsa di Malaysia?",
        "Passport application process — operational/unrelated to economic or welfare policy",
    ),
]


@pytest.mark.parametrize(
    "question,reason",
    _OUT_OF_SCOPE,
    ids=[q[0][:50] for q in _OUT_OF_SCOPE],
)
def test_out_of_scope_does_not_hallucinate(johor_research_rag, question, reason):
    """Out-of-scope questions must not produce a confident hallucinated answer."""
    answer = search(johor_research_rag, question, top_k=5)

    assert isinstance(answer, str), "Answer must be a string"

    answer_lower = answer.lower()
    has_no_context_signal = any(signal in answer_lower for signal in _NO_CONTEXT_SIGNALS)
    is_very_short = len(answer.strip()) < 150

    assert has_no_context_signal or is_very_short, (
        f"Out-of-scope question got a confident answer — possible hallucination.\n"
        f"Reason why it's out-of-scope: {reason}\n"
        f"Answer: {answer[:500]}"
    )
