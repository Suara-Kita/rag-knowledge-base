from neo4j import Driver
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from src.config import settings


def build_pipeline(driver: Driver) -> SimpleKGPipeline:
    llm = OpenAILLM(
        model_name=settings.llm_model,
        model_params={"temperature": 0},
    )
    embedder = OpenAIEmbeddings(model=settings.embedding_model)

    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_file=False,
        on_error="IGNORE",
        neo4j_database=settings.neo4j_database,
    )
    return pipeline
