from pathlib import Path

import pytest

from src.markdown.processor import extract_text


def test_extract_text_returns_content(test_md: Path) -> None:
    text = extract_text(test_md)
    assert "Flood mitigation strategies in Johor" in text
    assert "multi-agency coordination" in text


def test_extract_text_nonexistent_file() -> None:
    with pytest.raises(Exception):
        extract_text(Path("/nonexistent/file.md"))
