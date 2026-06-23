import os

import pytest

pytestmark = pytest.mark.integration


def test_driver_connectivity(neo4j_driver) -> None:
    neo4j_driver.verify_connectivity()


def test_vector_index_creation(neo4j_driver, vector_index_name) -> None:
    unique_label = f"TestChunk_{vector_index_name.replace('-', '_')}"
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with neo4j_driver.session(database=database) as session:
        session.run(
            f"CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS "
            f"FOR (n:{unique_label}) ON (n.embedding) "
            "OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } }"
        ).data()
        result = session.run(
            "SHOW INDEXES WHERE name = $name",
            name=vector_index_name,
        ).data()
        assert len(result) == 1
        assert result[0]["type"] == "VECTOR"


@pytest.mark.asyncio
async def test_pipeline_creates_document_and_chunks(neo4j_driver) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from src.knowledge.pipeline import build_pipeline

    pipeline = build_pipeline(neo4j_driver)

    text = "# Johor Economy\n\nJohor state government plans to build flood barriers along Sungai Segamat."
    await pipeline.run_async(text=text, file_path="test-paper.md")

    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with neo4j_driver.session(database=database) as session:
        docs = session.run("MATCH (d:Document) RETURN d").data()
        assert len(docs) >= 1
        paths = [d["d"]["path"] for d in docs]
        assert "test-paper.md" in paths

        chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()
        assert chunks["cnt"] > 0
