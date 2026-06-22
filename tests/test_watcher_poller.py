from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.watcher.poller import _already_ingested


@patch("src.watcher.poller.get_driver")
def test_already_ingested_true(mock_get_driver) -> None:
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"exists": True}
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_get_driver.return_value = mock_driver

    assert _already_ingested("paper.pdf") is True
    mock_session.run.assert_called_once()


@patch("src.watcher.poller.get_driver")
def test_already_ingested_false(mock_get_driver) -> None:
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"exists": False}
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_get_driver.return_value = mock_driver

    assert _already_ingested("new.pdf") is False


@patch("src.watcher.poller._already_ingested", return_value=False)
@patch("src.watcher.poller.process_pdf")
@patch("src.watcher.poller.settings")
async def test_ingest_pdfs_processes_new_file(
    mock_settings,
    mock_process_pdf,
    mock_already_ingested,
    watch_dir: Path,
    processed_dir: Path,
    test_pdf: Path,
) -> None:
    mock_settings.watch_dir = str(watch_dir)
    mock_settings.processed_dir = str(processed_dir)

    dest = watch_dir / test_pdf.name
    dest.write_bytes(test_pdf.read_bytes())

    from src.watcher.poller import _ingest_pdfs

    await _ingest_pdfs()

    assert not (watch_dir / test_pdf.name).exists()
    assert (processed_dir / test_pdf.name).exists()
    mock_process_pdf.assert_awaited_once()


@patch("src.watcher.poller._already_ingested", return_value=True)
@patch("src.watcher.poller.process_pdf")
@patch("src.watcher.poller.settings")
async def test_ingest_pdfs_skips_ingested(
    mock_settings,
    mock_process_pdf,
    mock_already_ingested,
    watch_dir: Path,
    processed_dir: Path,
    test_pdf: Path,
) -> None:
    mock_settings.watch_dir = str(watch_dir)
    mock_settings.processed_dir = str(processed_dir)

    dest = watch_dir / test_pdf.name
    dest.write_bytes(test_pdf.read_bytes())

    from src.watcher.poller import _ingest_pdfs

    await _ingest_pdfs()

    assert not (watch_dir / test_pdf.name).exists()
    assert (processed_dir / test_pdf.name).exists()
    mock_process_pdf.assert_not_awaited()
