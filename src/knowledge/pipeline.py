from neo4j import Driver
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from src.config import settings
from src.embeddings.factory import get_embedder
from src.knowledge.md_splitter import MarkdownSectionSplitter

# Explicit schema avoids LLM schema-inference call (which can fail on large docs).
_ENTITIES = [
    {"label": "Region", "description": "A geographic region, state, or country e.g. Johor, Malaysia, Selangor"},
    {"label": "EconomicIndicator", "description": "A macroeconomic metric e.g. GDP growth, median salary, FDI"},
    {"label": "Sector", "description": "An economic sector e.g. manufacturing, digital economy, services"},
    {"label": "Policy", "description": "A government policy, initiative, or economic zone e.g. JS-SEZ, NIMP"},
    {"label": "Organization", "description": "A government body, agency, or institution"},
    {"label": "Year", "description": "A calendar year or time period"},
]

_RELATIONS = [
    {"label": "HAS_INDICATOR", "description": "Region has an economic indicator value"},
    {"label": "OUTPERFORMS", "description": "One region outperforms another on a metric"},
    {"label": "IMPLEMENTS", "description": "An organization or region implements a policy"},
    {"label": "LOCATED_IN", "description": "A region is located within another region"},
    {"label": "IMPACTS", "description": "A sector or policy impacts a region or indicator"},
    {"label": "RECORDED_IN", "description": "An indicator value recorded in a year"},
]


def build_pipeline(driver: Driver) -> SimpleKGPipeline:
    llm = OpenAILLM(
        model_name=settings.llm_model,
        model_params={"temperature": 0},
    )
    embedder = get_embedder()

    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        entities=_ENTITIES,
        relations=_RELATIONS,
        text_splitter=MarkdownSectionSplitter(min_chars=500, max_chars=4000, overlap=200),
        from_file=False,
        on_error="IGNORE",
        neo4j_database=settings.neo4j_database,
    )
    return pipeline
