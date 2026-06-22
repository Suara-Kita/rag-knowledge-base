import asyncio
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_full_flow_pdf_to_query(
    neo4j_driver,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Drop a PDF in watch/ → poller picks it → query finds it."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    watch = tmp_path / "watch"
    processed = tmp_path / "processed"
    watch.mkdir()
    processed.mkdir()

    monkeypatch.setattr("src.config.settings.watch_dir", str(watch))
    monkeypatch.setattr("src.config.settings.processed_dir", str(processed))
    monkeypatch.setattr("src.config.settings.watch_interval_ms", 30000)

    pdf_content = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Johor needs early warning systems for floods.) Tj ET\n"
        b"endstream\nendobj\nxref\n0 6\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n0000000079 00000 n \n"
        b"0000000131 00000 n \n0000000220 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n269\n%%EOF\n"
    )
    pdf_path = watch / "research.pdf"
    pdf_path.write_bytes(pdf_content)

    from src.watcher.poller import _ingest_pdfs

    await _ingest_pdfs()

    assert not pdf_path.exists()
    assert (processed / "research.pdf").exists()

    with neo4j_driver.session() as session:
        docs = session.run("MATCH (d:Document) RETURN d.path AS path").data()
        paths = [d["path"] for d in docs]
        assert "research.pdf" in paths

    from src.db.neo4j import get_driver
    from src.knowledge.retriever import build_rag, search

    rag = build_rag(get_driver())
    answer = search(rag, "What does Johor need for floods?", top_k=5)
    assert isinstance(answer, str)
    assert len(answer) > 0
