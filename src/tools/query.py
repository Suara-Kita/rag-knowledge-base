import logging
from pathlib import Path

from mcp.types import Tool, TextContent

from src.config import settings
from src.db.neo4j import get_driver
from src.knowledge.retriever import build_rag, build_filtered_rag, search
from src.knowledge.works_cited import extract_works_cited

logger = logging.getLogger(__name__)

_JOHOR_DOC_FILTER = "Perbezaan Malaysia dan Johor"

_rag_instance = None
_johor_rag_instance = None


def _load_works_cited(doc_filter: str) -> str:
    """Find the best-matching processed markdown and extract its Works Cited section.

    Resolves processed_dir at call time (not import time) to handle processes
    launched from non-project-root directories. Picks the most recently modified
    matching file to avoid non-determinism when multiple files share the filter string.
    """
    processed_dir = Path(settings.processed_dir).resolve()
    matches = sorted(
        (md for md in processed_dir.glob("*.md") if doc_filter in md.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for md in matches:
        try:
            return extract_works_cited(md)
        except Exception:
            logger.warning("Could not parse Works Cited from %s", md.name)
    return ""


def _get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = build_rag(get_driver())
    return _rag_instance


def _get_johor_rag():
    global _johor_rag_instance
    if _johor_rag_instance is None:
        works_cited = _load_works_cited(_JOHOR_DOC_FILTER)
        _johor_rag_instance = build_filtered_rag(get_driver(), _JOHOR_DOC_FILTER, works_cited)
    return _johor_rag_instance


def invalidate_rag_cache() -> None:
    """Drop cached RAG instances so the next call rebuilds with fresh works_cited.

    Call this after ingesting a new document to ensure updated citations appear.
    """
    global _rag_instance, _johor_rag_instance
    _rag_instance = None
    _johor_rag_instance = None


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="query_knowledge",
            description="Search the knowledge base for relevant information from ingested research documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to search the knowledge base for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="query_johor_economy",
            description=(
                "Query the Johor/Malaysia macroeconomic analysis document. "
                "Use this for questions about GDP growth, sectoral breakdown, FDI, wages, "
                "JS-SEZ, data centre investment, subnational economic comparison, "
                "or Johor's economic trajectory (2022–2026). "
                "Results are filtered to that document only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Economic question about Johor or Malaysia",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="list_documents",
            description="List all ingested documents in the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_knowledge":
        question = arguments["question"]
        top_k = arguments.get("top_k", 5)
        try:
            answer = search(_get_rag(), question, top_k)
            return [TextContent(type="text", text=answer)]
        except Exception as e:
            logger.exception("query_knowledge failed")
            return [TextContent(type="text", text=f"Error: {e}")]

    if name == "query_johor_economy":
        question = arguments["question"]
        try:
            answer = search(_get_johor_rag(), question, top_k=8)
            return [TextContent(type="text", text=answer)]
        except Exception as e:
            logger.exception("query_johor_economy failed")
            return [TextContent(type="text", text=f"Error: {e}")]

    if name == "list_documents":
        driver = get_driver()
        query = """
            MATCH (d:Document)
            RETURN d.path AS title, d.path AS source, d.createdAt AS ingested
            ORDER BY d.createdAt DESC
        """
        with driver.session(database=settings.neo4j_database) as session:
            results = session.run(query).data()
        if not results:
            return [TextContent(type="text", text="No documents ingested yet.")]
        lines = ["## Ingested Documents\n"]
        for r in results:
            title = r.get("title") or r.get("source", "unknown")
            ingested = r.get("ingested", "unknown")
            lines.append(f"- **{title}** (ingested: {ingested})")
        return [TextContent(type="text", text="\n".join(lines))]

    raise ValueError(f"Unknown tool: {name}")
