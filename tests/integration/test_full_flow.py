import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_full_flow_md_to_query(
    neo4j_driver,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Drop a Markdown file in watch/ → poller picks it → query finds it."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    watch = tmp_path / "watch"
    processed = tmp_path / "processed"
    watch.mkdir()
    processed.mkdir()

    monkeypatch.setattr("src.config.settings.watch_dir", str(watch))
    monkeypatch.setattr("src.config.settings.processed_dir", str(processed))
    monkeypatch.setattr("src.config.settings.watch_interval_ms", 30000)

    md_path = watch / "research.md"
    md_path.write_text(
        "# Johor Flood Research\n\nJohor needs early warning systems for floods.",
        encoding="utf-8",
    )

    from src.watcher.poller import _ingest_files

    await _ingest_files()

    assert not md_path.exists()
    assert (processed / "research.md").exists()

    with neo4j_driver.session() as session:
        docs = session.run("MATCH (d:Document) RETURN d.path AS path").data()
        paths = [d["path"] for d in docs]
        assert "research.md" in paths

    from src.db.neo4j import get_driver
    from src.knowledge.retriever import build_rag, search

    rag = build_rag(get_driver())
    answer = search(rag, "What does Johor need for floods?", top_k=5)
    assert isinstance(answer, str)
    assert len(answer) > 0
