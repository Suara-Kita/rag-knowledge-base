import logging

from mcp.types import Tool, TextContent

from src.config import settings
from src.db.neo4j import get_driver
from src.knowledge.retriever import build_rag, search

logger = logging.getLogger(__name__)


_rag_instance = None


def _get_rag():
    global _rag_instance
    if _rag_instance is None:
        driver = get_driver()
        _rag_instance = build_rag(driver)
    return _rag_instance


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
            rag = _get_rag()
            answer = search(rag, question, top_k)
            return [TextContent(type="text", text=answer)]
        except Exception as e:
            logger.exception("Query failed")
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
