import os

import pytest

from src.knowledge.retriever import build_retriever, build_rag, search

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_retriever_builds(neo4j_driver) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    retriever = build_retriever(neo4j_driver)
    assert retriever is not None


@pytest.mark.asyncio
async def test_search_without_data(neo4j_driver) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    rag = build_rag(neo4j_driver)
    answer = search(rag, "test query", top_k=3)
    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_search_after_ingestion(neo4j_driver) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from src.knowledge.pipeline import build_pipeline

    pipeline = build_pipeline(neo4j_driver)

    text = "# Flood Mitigation\n\nFlood mitigation in Johor requires building retention ponds and early warning systems."
    await pipeline.run_async(text=text, file_path="flood-mitigation.md")

    rag = build_rag(neo4j_driver)
    answer = search(rag, "What flood mitigation strategies are mentioned?", top_k=5)

    assert isinstance(answer, str)
    assert len(answer) > 20
