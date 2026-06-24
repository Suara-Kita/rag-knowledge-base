import asyncio
import logging
from pathlib import Path

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.config import settings
from src.db.neo4j import get_driver
from src.markdown.processor import process_markdown

logger = logging.getLogger(__name__)


def _already_ingested(source_path: str) -> bool:
    driver = get_driver()
    query = """
        MATCH (d:Document {path: $source_path})
        RETURN count(d) > 0 AS exists
    """
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(query, source_path=source_path).single()
        exists = result["exists"] if result else False
    return exists


def _delete_partial_document(source_path: str) -> None:
    """Remove any Document/Chunk/Entity nodes left by a failed ingestion.

    SimpleKGPipeline may commit the Document node — and extract entities
    linked to chunks via FROM_CHUNK — before it finishes writing every chunk,
    so a crash mid-pipeline leaves orphans that (a) cause _already_ingested
    to return True on the next cycle, permanently skipping the file, and
    (b) pollute future entity-collection queries with disconnected Entity
    nodes if left behind.
    """
    driver = get_driver()
    query = """
        MATCH (d:Document {path: $source_path})
        OPTIONAL MATCH (d)<-[:FROM_DOCUMENT]-(c:Chunk)
        OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(e)
        DETACH DELETE d, c, e
    """
    with driver.session(database=settings.neo4j_database) as session:
        session.run(query, source_path=source_path)


async def _ingest_files() -> None:
    processed = Path(settings.processed_dir).resolve()
    processed.mkdir(parents=True, exist_ok=True)

    # Document identity (and the processed/ destination) is keyed on bare
    # filename only, so two watch_dirs can't both contribute a file with the
    # same name — silently treating the second as a duplicate would either
    # falsely skip genuinely different content or clobber the first file's
    # processed/ copy on rename. Surface the conflict loudly instead.
    claimed_by: dict[str, str] = {}

    for watch_dir in settings.watch_dirs:
        watch = Path(watch_dir).resolve()
        watch.mkdir(parents=True, exist_ok=True)

        for file_path in sorted(watch.glob("*.md")):
            name = file_path.name

            if name in claimed_by and claimed_by[name] != watch_dir:
                logger.error(
                    "Filename collision: %r exists in both %s and %s. "
                    "Skipping the copy in %s to avoid mis-deduplicating or "
                    "overwriting the other one in processed/ — rename one of them.",
                    name, claimed_by[name], watch_dir, watch_dir,
                )
                continue
            claimed_by[name] = watch_dir

            if _already_ingested(name):
                logger.info("Skipping already-ingested %s", name)
                file_path.rename(processed / name)
                continue

            logger.info("Ingesting %s...", name)
            try:
                await process_markdown(file_path)
                file_path.rename(processed / name)
                logger.info("Ingested %s", name)
            except (ServiceUnavailable, Neo4jError):
                # Infra-level failure, not a bad document — don't delete
                # anything (nothing is necessarily partial) and abort the
                # whole cycle instead of repeating this for every remaining
                # file across every watch_dir.
                logger.exception(
                    "Neo4j unavailable while ingesting %s — aborting this poll cycle", name
                )
                raise
            except Exception:
                logger.exception("Failed to ingest %s", name)
                _delete_partial_document(name)


async def start_poller(stop_event: asyncio.Event) -> None:
    interval = settings.watch_interval_ms / 1000
    logger.info(
        "Watcher polling %s every %.1fs",
        settings.watch_dirs,
        interval,
    )
    while not stop_event.is_set():
        try:
            await _ingest_files()
        except Exception:
            logger.exception("Watcher cycle failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
