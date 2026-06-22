import os

import pytest

from src.db.neo4j import close_driver, get_driver

pytestmark = pytest.mark.integration


def test_driver_connectivity(neo4j_driver) -> None:
    neo4j_driver.verify_connectivity()


def test_vector_index_creation(neo4j_driver, vector_index_name) -> None:
    # Use a unique label to avoid "one vector index per label" limit
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
    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

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

    text = "Johor state government plans to build flood barriers along Sungai Segamat."
    await pipeline.run_async(text=text, file_path="test-paper.pdf")

    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with neo4j_driver.session(database=database) as session:
        docs = session.run("MATCH (d:Document) RETURN d").data()
        assert len(docs) >= 1
        assert docs[0]["d"]["path"] == "test-paper.pdf"

        chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()
        assert chunks["cnt"] > 0
