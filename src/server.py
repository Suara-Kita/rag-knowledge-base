"""HTTP server wrapping the RAG knowledge base for the Onn AI chatbot.

Run from the rag-knowledge-base directory:
    python -m src.server

Exposes:
    GET  /health  — liveness check
    POST /query   — { "question": str } → { "reply": str, "sources": [] }
"""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from src.config import settings  # noqa: E402

# Propagate OpenRouter credentials to env so neo4j-graphrag's OpenAI client picks them up
os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("kb-server")

import redis.asyncio as aioredis  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from src.db.neo4j import get_driver, close_driver  # noqa: E402
from src.tools.query import handle_call_tool  # noqa: E402

_UNAVAILABLE = "Maaf, sistem tidak tersedia buat masa ini. Sila cuba lagi."
_VOTER_INPUT_QUEUE = "queue:voter_inputs"

# Phrases the LLM emits when the KB has no relevant context
_NO_MATCH_BM = "Maaf, maklumat yang diperlukan tidak terdapat dalam konteks yang diberikan"
_NO_MATCH_EN = "the required information is not available in the provided context"

_redis_client: aioredis.Redis | None = None


def _is_unmatched(reply: str) -> bool:
    return _NO_MATCH_BM in reply or _NO_MATCH_EN in reply


async def _publish_unmatched(question: str) -> None:
    global _redis_client
    if _redis_client is None:
        return
    try:
        import uuid as _uuid
        payload = json.dumps({
            "pipeline_metadata": {
                "ingestion_id": str(_uuid.uuid4()),
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


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "model": settings.llm_model})


async def query(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    try:
        results = await handle_call_tool("query_knowledge", {"question": question})
        reply = results[0].text if results else _UNAVAILABLE
        if _is_unmatched(reply):
            await _publish_unmatched(question)
        return JSONResponse({"reply": reply, "sources": []})
    except Exception:
        logger.exception("query failed for: %s", question)
        return JSONResponse({"reply": _UNAVAILABLE}, status_code=200)


@asynccontextmanager
async def lifespan(app: Starlette):
    global _redis_client
    driver = get_driver()
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s (model: %s)", settings.neo4j_uri, settings.llm_model)
    except Exception as e:
        logger.error("Neo4j connection failed: %s — exiting", e)
        sys.exit(1)

    try:
        _redis_client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
        await _redis_client.ping()
        logger.info("Redis connected at %s", settings.redis_url)
    except Exception as e:
        logger.warning("Redis unavailable (%s) — unmatched queries will not be published", e)
        _redis_client = None

    yield

    if _redis_client:
        await _redis_client.aclose()
    close_driver()
    logger.info("Shutdown complete")


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/query", query, methods=["POST"]),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, log_level="info")
