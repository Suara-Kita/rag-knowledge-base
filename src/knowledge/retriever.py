from neo4j import Driver
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from src.config import settings


_retrieval_query = """
OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(d:Document)
OPTIONAL MATCH (e:Entity)-[:FROM_CHUNK]->(c)
RETURN c.text AS text,
       d.path AS document,
       collect(DISTINCT e.name) AS entities,
       score
"""


def build_retriever(driver: Driver) -> VectorCypherRetriever:
    embedder = OpenAIEmbeddings(model=settings.embedding_model)
    return VectorCypherRetriever(
        driver=driver,
        index_name="chunk_embeddings",
        embedder=embedder,
        retrieval_query=_retrieval_query,
        neo4j_database=settings.neo4j_database,
    )


def build_rag(driver: Driver) -> GraphRAG:
    llm = OpenAILLM(
        model_name=settings.llm_model,
        model_params={"temperature": 0},
    )
    retriever = build_retriever(driver)
    return GraphRAG(llm=llm, retriever=retriever)


def search(rag: GraphRAG, question: str, top_k: int = 5) -> str:
    result = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
    )
    return result.answer
