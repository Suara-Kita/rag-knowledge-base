from pathlib import Path

from pypdf import PdfReader

from src.db.neo4j import get_driver
from src.knowledge.pipeline import build_pipeline


async def process_pdf(pdf_path: Path) -> None:
    text = extract_text(pdf_path)
    driver = get_driver()
    pipeline = build_pipeline(driver)
    await pipeline.run_async(text=text, file_path=pdf_path.name)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)
