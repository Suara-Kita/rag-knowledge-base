import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from src.config import settings  # noqa: E402

os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("knowledge-base")

from mcp.server import Server  # noqa: E402
from mcp.server.sse import SseServerTransport  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, Response  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402

from src.db.neo4j import get_driver, close_driver  # noqa: E402
from src.tools.query import get_tools, handle_call_tool  # noqa: E402
from src.watcher.poller import start_poller  # noqa: E402

mcp_server = Server("suara-kita-knowledge")
sse = SseServerTransport("/messages/")


@mcp_server.list_tools()
async def list_tools():
    return get_tools()


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    return await handle_call_tool(name, arguments)


async def handle_sse(request: Request) -> Response:
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
    return Response()


async def handle_query(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    raw_top_k = body.get("top_k")
    top_k = int(raw_top_k) if isinstance(raw_top_k, (int, float)) and raw_top_k > 0 else 5
    results = await handle_call_tool("query_knowledge", {"question": question, "top_k": top_k})
    return JSONResponse({"answer": results[0].text if results else ""})


def _ensure_database(driver) -> None:
    if settings.neo4j_database == "neo4j":
        return
    try:
        with driver.session(database="system") as session:
            session.run("CREATE DATABASE $name IF NOT EXISTS", name=settings.neo4j_database).consume()
        logger.info("Database '%s' ready", settings.neo4j_database)
    except Exception as e:
        logger.warning("Database '%s' not available (Neo4j Community?): %s", settings.neo4j_database, e)


def _create_indexes(driver) -> None:
    query = f"""
        CREATE VECTOR INDEX {settings.vector_index_name} IF NOT EXISTS
        FOR (n:Chunk) ON (n.embedding)
        OPTIONS {{ indexConfig: {{ `vector.dimensions`: toInteger($dims), `vector.similarity_function`: "cosine" }} }}
    """
    try:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(query, dims=settings.embedding_dims).consume()
        logger.info("Vector index '%s' ready (%d dims)", settings.vector_index_name, settings.embedding_dims)
    except Exception as e:
        logger.warning("Could not create vector index: %s", e)


@asynccontextmanager
async def lifespan(app: Starlette):
    driver = get_driver()
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
    except Exception as e:
        logger.error("Neo4j connection failed: %s — exiting", e)
        sys.exit(1)

    _ensure_database(driver)
    _create_indexes(driver)

    stop_event = asyncio.Event()
    watcher_task = asyncio.create_task(start_poller(stop_event))
    logger.info("MCP SSE server ready — listening on /sse")

    yield

    stop_event.set()
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass
    close_driver()
    logger.info("Shutdown complete")


app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Route("/query", endpoint=handle_query, methods=["POST"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MCP_PORT", "8002"))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, log_level="info")
