import asyncio
import logging
from pathlib import Path

from src.config import settings
from src.db.neo4j import get_driver
from src.pdf.processor import process_pdf

logger = logging.getLogger(__name__)


def _already_ingested(source_path: str) -> bool:
    driver = get_driver()
    query = """
        MATCH (d:Document {source_path: $source_path})
        RETURN count(d) > 0 AS exists
    """
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(query, source_path=source_path).single()
        exists = result["exists"] if result else False
    return exists


async def _ingest_pdfs() -> None:
    watch = Path(settings.watch_dir).resolve()
    processed = Path(settings.processed_dir).resolve()
    watch.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(watch.glob("*.pdf")):
        name = pdf_path.name
        if _already_ingested(name):
            logger.info("Skipping already-ingested %s", name)
            pdf_path.rename(processed / name)
            continue

        logger.info("Ingesting %s...", name)
        try:
            await process_pdf(pdf_path)
            pdf_path.rename(processed / name)
            logger.info("Ingested %s", name)
        except Exception:
            logger.exception("Failed to ingest %s", name)


async def start_poller(stop_event: asyncio.Event) -> None:
    interval = settings.watch_interval_ms / 1000
    logger.info(
        "Watcher polling %s every %.1fs",
        settings.watch_dir,
        interval,
    )
    while not stop_event.is_set():
        try:
            await _ingest_pdfs()
        except Exception:
            logger.exception("Watcher cycle failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
