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

    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

    llm = OpenAILLM(model_name="openai/gpt-oss-120b", model_params={"temperature": 0})
    embedder = OpenAIEmbeddings(model="openai/gpt-oss-120b")

    database = os.getenv("NEO4J_DATABASE", "neo4j")
    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        from_file=False,
        on_error="IGNORE",
        neo4j_database=database,
    )

    text = "Flood mitigation in Johor requires building retention ponds and early warning systems."
    await pipeline.run_async(text=text, file_path="flood-mitigation.pdf")

    rag = build_rag(neo4j_driver)
    answer = search(rag, "What flood mitigation strategies are mentioned?", top_k=5)

    assert isinstance(answer, str)
    assert len(answer) > 20
