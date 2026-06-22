from pathlib import Path

import pytest

from src.pdf.processor import extract_text


def test_extract_text_returns_content(test_pdf: Path) -> None:
    text = extract_text(test_pdf)
    assert "Flood mitigation strategies in Johor" in text
    assert "multi-agency coordination" in text


def test_extract_text_nonexistent_file() -> None:
    with pytest.raises(Exception):
        extract_text(Path("/nonexistent/file.pdf"))
