import os

import pytest
from neo4j import GraphDatabase

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def neo4j_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7688")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "changeme")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password), database=database)
    driver.verify_connectivity()
    yield driver
    driver.close()


@pytest.fixture(autouse=True)
def clean_graph(neo4j_driver):
    """Remove all test data between tests but keep indexes."""
    with neo4j_driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield


def _ensure_vector_index(driver, name: str) -> None:
    """Create vector index only if it doesn't exist."""
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=database) as session:
        exists = session.run(
            "SHOW INDEXES WHERE name = $name",
            name=name,
        ).single()
        if not exists:
            session.run(
                f"CREATE VECTOR INDEX {name} "
                "FOR (n:Chunk) ON (n.embedding) "
                "OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } }"
            ).data()


@pytest.fixture(scope="session")
def vector_index_name(neo4j_driver):
    name = "chunk_embeddings"
    _ensure_vector_index(neo4j_driver, name)
    return name
