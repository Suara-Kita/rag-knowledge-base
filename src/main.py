import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure OpenAI-compatible env vars are set for neo4j-graphrag
from src.config import settings
os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("knowledge-base")

from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.db.neo4j import get_driver, close_driver
from src.tools.query import get_tools, handle_call_tool
from src.watcher.poller import start_poller

server = Server("suara-kita-knowledge")


@server.list_tools()
async def list_tools():
    return get_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    return await handle_call_tool(name, arguments)


def _ensure_database(driver) -> None:
    """Create the target database if running Neo4j Enterprise.

    Community Edition silently ignores this — the default 'neo4j' database is always available.
    """
    if settings.neo4j_database == "neo4j":
        return
    try:
        with driver.session(database="system") as session:
            session.run("CREATE DATABASE $name IF NOT EXISTS", name=settings.neo4j_database).consume()
        logger.info("Database '%s' ready", settings.neo4j_database)
    except Exception as e:
        logger.warning("Database '%s' not available (Neo4j Community?): %s", settings.neo4j_database, e)


def _create_indexes(driver) -> None:
    # Index name cannot be parameterized in Cypher schema commands — settings value is safe (config-only).
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


async def main():
    logger.info("Starting Suara Kita Knowledge Base MCP server")

    driver = get_driver()
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
    except Exception as e:
        logger.error("Neo4j connection failed: %s", e)
        sys.exit(1)

    _ensure_database(driver)
    _create_indexes(driver)

    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("Shutting down...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    watcher_task = asyncio.create_task(start_poller(stop_event))

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        stop_event.set()
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
        close_driver()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
