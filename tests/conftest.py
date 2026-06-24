from pathlib import Path

import pytest


@pytest.fixture
def test_md(tmp_path: Path) -> Path:
    """Create a simple Markdown file with known text content."""
    md_path = tmp_path / "test-document.md"
    md_path.write_text(
        "# Flood Mitigation\n\nFlood mitigation strategies in Johor require multi-agency coordination.",
        encoding="utf-8",
    )
    return md_path


@pytest.fixture
def watch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "watch"
    d.mkdir()
    return d


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    return d
