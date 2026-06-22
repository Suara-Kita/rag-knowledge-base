from pathlib import Path

import pytest
from fpdf import FPDF


@pytest.fixture
def test_pdf(tmp_path: Path) -> Path:
    """Create a valid PDF with known text content."""
    pdf_path = tmp_path / "test-document.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Flood mitigation strategies in Johor require multi-agency coordination.")
    pdf.output(str(pdf_path))
    return pdf_path


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
