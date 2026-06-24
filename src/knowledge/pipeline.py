from neo4j import Driver
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import AnthropicLLM

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
    # Used for entity/relationship extraction only; query-time answer
    # generation still uses settings.llm_model (gpt-oss) via retriever.py.
    llm = AnthropicLLM(
        model_name=settings.entity_llm_model,
        # api_key explicit (falls back to the SDK's own ANTHROPIC_API_KEY env
        # lookup when unset) so the key actually goes through Settings/.env
        # instead of silently depending on load_dotenv() populating the same
        # env var name by coincidence.
        api_key=settings.anthropic_api_key or None,
        # 8192 leaves headroom for dense chunks (many entities/relations per
        # chunk) without truncating mid-JSON — 4096 was cutting it close for
        # max_chars=4000 chunks with heavy entity density.
        model_params={"temperature": 0, "max_tokens": 8192},
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
