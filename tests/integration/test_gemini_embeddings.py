import os

import pytest

pytestmark = pytest.mark.integration

JOHOR_TEXT = (
    "Johor mencatatkan pertumbuhan KDNK benar sebanyak 6.4% pada tahun 2024, "
    "menjadikannya negeri dengan pertumbuhan terpantas di Malaysia. "
    "Ini mengatasi purata nasional sebanyak 5.1%, dengan jurang positif 1.3 mata peratusan. "
    "Pelaburan dalam pusat data berskala hiper dan Zon Ekonomi Khas Johor-Singapura (JS-SEZ) "
    "menjadi pemacu utama pertumbuhan ini."
)


@pytest.fixture(scope="module")
def gemini_embedder():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    from src.embeddings.gemini import GeminiEmbedder
    return GeminiEmbedder()


@pytest.fixture(scope="module")
def gemini_index(neo4j_driver):
    from src.embeddings.gemini import GEMINI_EMBEDDING_DIMS, GEMINI_INDEX_NAME
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with neo4j_driver.session(database=database) as s:
        s.run(
            f"CREATE VECTOR INDEX {GEMINI_INDEX_NAME} IF NOT EXISTS "
            "FOR (n:GeminiChunk) ON (n.embedding) "
            f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {GEMINI_EMBEDDING_DIMS}, "
            "`vector.similarity_function`: 'cosine' } }"
        ).consume()
    return GEMINI_INDEX_NAME


def test_gemini_embedder_returns_correct_dimensions(gemini_embedder):
    from src.embeddings.gemini import GEMINI_EMBEDDING_DIMS
    vec = gemini_embedder.embed_query("Johor GDP growth 2024")
    assert isinstance(vec, list)
    assert len(vec) == GEMINI_EMBEDDING_DIMS


@pytest.mark.asyncio
async def test_gemini_ingest_and_retrieve(neo4j_driver, gemini_embedder, gemini_index):
    from neo4j_graphrag.retrievers import VectorCypherRetriever
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.llm import OpenAILLM
    from src.knowledge.pipeline import build_pipeline
    from src.config import settings

    database = os.getenv("NEO4J_DATABASE", "neo4j")

    # Ingest using the standard pipeline (writes Chunk nodes)
    pipeline = build_pipeline(neo4j_driver)
    await pipeline.run_async(text=JOHOR_TEXT, file_path="johor-gemini-test.md")

    # Re-embed chunks from this document using Gemini and write GeminiChunk nodes
    with neo4j_driver.session(database=database) as s:
        chunks = s.run(
            'MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d:Document {path: "johor-gemini-test.md"}) '
            "RETURN elementId(c) AS eid, c.text AS text"
        ).data()

    assert len(chunks) > 0, "No chunks found after ingestion"

    with neo4j_driver.session(database=database) as s:
        for chunk in chunks:
            vec = gemini_embedder.embed_query(chunk["text"])
            s.run(
                "MERGE (g:GeminiChunk {source_eid: $eid}) "
                "SET g.text = $text, g.embedding = $vec",
                eid=chunk["eid"], text=chunk["text"], vec=vec,
            ).consume()

    # Query via Gemini vector index
    retriever = VectorCypherRetriever(
        driver=neo4j_driver,
        index_name=gemini_index,
        embedder=gemini_embedder,
        retrieval_query="RETURN node.text AS text, score",
        neo4j_database=database,
    )
    llm = OpenAILLM(model_name=settings.llm_model, model_params={"temperature": 0})
    rag = GraphRAG(llm=llm, retriever=retriever)

    result = rag.search(
        query_text="Apakah pertumbuhan KDNK Johor pada 2024 dan apa pemacu utamanya?",
        retriever_config={"top_k": 3},
    )

    assert isinstance(result.answer, str)
    assert len(result.answer) > 20
    # Answer should reference Johor or GDP or the key facts
    answer_lower = result.answer.lower()
    assert any(kw in answer_lower for kw in ["johor", "6.4", "gdp", "kdnk", "js-sez", "5.1"])
