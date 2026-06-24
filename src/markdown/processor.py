import re
from pathlib import Path

from src.db.neo4j import get_driver
from src.knowledge.pipeline import build_pipeline


def extract_text(md_path: Path) -> str:
    return md_path.read_text(encoding="utf-8")


def strip_works_cited(text: str) -> str:
    """Remove the Works Cited / References section from the end of the document.

    The section is parsed separately by works_cited.py and injected into the
    RAG prompt at query time — keeping it out of the chunk index avoids filling
    the vector store with raw URL strings that add no semantic value.
    """
    # Match any heading level containing 'works cited' or 'references' (case-insensitive)
    pattern = re.compile(r"^#{1,6}\s.*?(works\s+cited|references)\b.*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


async def process_markdown(md_path: Path) -> None:
    raw = extract_text(md_path)
    text = strip_works_cited(raw)
    driver = get_driver()
    pipeline = build_pipeline(driver)
    await pipeline.run_async(text=text, file_path=md_path.name)
