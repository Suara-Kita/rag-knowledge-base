import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

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

import redis.asyncio as aioredis  # noqa: E402

from src.db.neo4j import get_driver, close_driver  # noqa: E402
from src.tools.query import get_tools, handle_call_tool  # noqa: E402
from src.watcher.poller import start_poller  # noqa: E402

_VOTER_INPUT_QUEUE = "queue:voter_inputs"
_NO_MATCH_BM = "Maaf, maklumat yang diperlukan tidak terdapat dalam konteks yang diberikan"
_NO_MATCH_EN = "the required information is not available in the provided context"
_redis_client: aioredis.Redis | None = None


def _is_unmatched(answer: str) -> bool:
    return _NO_MATCH_BM in answer or _NO_MATCH_EN in answer


async def _publish_unmatched(question: str) -> None:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
            await _redis_client.ping()
            logger.info("Redis reconnected — unmatched queries will be queued")
        except Exception as e:
            logger.warning("Redis unavailable, skipping unmatched publish: %s", e)
            _redis_client = None
            return
    try:
        payload = json.dumps({
            "pipeline_metadata": {
                "ingestion_id": str(uuid.uuid4()),
                "source_channel": "web_portal",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "trace_url": None,
            },
            "source_profile": {
                "client_identifier": "onn_ai",
                "display_name": "Onn AI",
                "contact_info": None,
                "inferred_constituency": None,
            },
            "content_payload": {
                "raw_text": question,
                "content_type": "text_only",
                "media_attachments": [],
            },
            "context_anchor": None,
        })
        await _redis_client.lpush(_VOTER_INPUT_QUEUE, payload)
        logger.info("Unmatched onn-ai query pushed to %s: %.80s", _VOTER_INPUT_QUEUE, question)
    except Exception:
        logger.warning("Failed to publish unmatched query to Redis", exc_info=True)
        _redis_client = None

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


async def _track_stats(client: aioredis.Redis, unmatched: bool) -> None:
    try:
        pipe = client.pipeline()
        pipe.incr("stats:onn-ai:questions_asked")
        if unmatched:
            pipe.incr("stats:onn-ai:maklum_balas")
        await pipe.execute()
    except Exception:
        logger.warning("Failed to track onn-ai stats", exc_info=True)


async def handle_stats(request: Request) -> JSONResponse:
    if _redis_client is None:
        return JSONResponse({"questions": None, "maklumBalas": None})
    try:
        questions, maklum_balas = await asyncio.wait_for(
            asyncio.gather(
                _redis_client.get("stats:onn-ai:questions_asked"),
                _redis_client.get("stats:onn-ai:maklum_balas"),
            ),
            timeout=2.0,
        )
        return JSONResponse({
            "questions": int(questions or 0),
            "maklumBalas": int(maklum_balas or 0),
        })
    except Exception:
        return JSONResponse({"questions": None, "maklumBalas": None})


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
    answer = results[0].text if results else ""
    unmatched = _is_unmatched(answer)
    if unmatched:
        await _publish_unmatched(question)
    if _redis_client is not None:
        asyncio.create_task(_track_stats(_redis_client, unmatched))
    return JSONResponse({"answer": answer})


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
    global _redis_client
    driver = get_driver()
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
    except Exception as e:
        logger.error("Neo4j connection failed: %s — exiting", e)
        sys.exit(1)

    _ensure_database(driver)
    _create_indexes(driver)

    try:
        _redis_client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
        await _redis_client.ping()
        logger.info("Redis connected — unmatched queries will be queued")
    except Exception as e:
        logger.warning("Redis unavailable (%s) — unmatched queries will not be published", e)
        _redis_client = None

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
    if _redis_client:
        await _redis_client.aclose()
    close_driver()
    logger.info("Shutdown complete")


app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Route("/query", endpoint=handle_query, methods=["POST"]),
        Route("/stats", endpoint=handle_stats, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MCP_PORT", "8002"))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, log_level="info")
