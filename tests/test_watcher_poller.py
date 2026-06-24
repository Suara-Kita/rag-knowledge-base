from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.watcher.poller import _already_ingested, _delete_partial_document


@patch("src.watcher.poller.get_driver")
def test_already_ingested_true(mock_get_driver) -> None:
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"exists": True}
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_get_driver.return_value = mock_driver

    assert _already_ingested("paper.md") is True
    mock_session.run.assert_called_once()


@patch("src.watcher.poller.get_driver")
def test_already_ingested_false(mock_get_driver) -> None:
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"exists": False}
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_get_driver.return_value = mock_driver

    assert _already_ingested("new.md") is False


@patch("src.watcher.poller.get_driver")
def test_delete_partial_document_also_removes_entities(mock_get_driver) -> None:
    """Cleanup must reach Entity nodes (linked via FROM_CHUNK to a deleted Chunk), not just Chunk/Document."""
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_get_driver.return_value = mock_driver

    _delete_partial_document("paper.md")

    query = mock_session.run.call_args.args[0]
    assert "FROM_CHUNK" in query
    assert "DETACH DELETE d, c, e" in query


@patch("src.watcher.poller._already_ingested", return_value=False)
@patch("src.watcher.poller.process_markdown")
@patch("src.watcher.poller.settings")
async def test_ingest_files_processes_new_file(
    mock_settings,
    mock_process_markdown,
    mock_already_ingested,
    watch_dir: Path,
    processed_dir: Path,
    test_md: Path,
) -> None:
    mock_settings.watch_dirs = [str(watch_dir)]
    mock_settings.processed_dir = str(processed_dir)

    dest = watch_dir / test_md.name
    dest.write_text(test_md.read_text())

    from src.watcher.poller import _ingest_files

    await _ingest_files()

    assert not (watch_dir / test_md.name).exists()
    assert (processed_dir / test_md.name).exists()
    mock_process_markdown.assert_awaited_once()


@patch("src.watcher.poller._already_ingested", return_value=False)
@patch("src.watcher.poller.process_markdown")
@patch("src.watcher.poller.settings")
async def test_ingest_files_processes_multiple_watch_dirs(
    mock_settings,
    mock_process_markdown,
    mock_already_ingested,
    tmp_path: Path,
    processed_dir: Path,
    test_md: Path,
) -> None:
    watch_a = tmp_path / "watch-a"
    watch_b = tmp_path / "watch-b"
    watch_a.mkdir()
    watch_b.mkdir()

    mock_settings.watch_dirs = [str(watch_a), str(watch_b)]
    mock_settings.processed_dir = str(processed_dir)

    (watch_a / "doc-a.md").write_text(test_md.read_text())
    (watch_b / "doc-b.md").write_text(test_md.read_text())

    from src.watcher.poller import _ingest_files

    await _ingest_files()

    assert not (watch_a / "doc-a.md").exists()
    assert not (watch_b / "doc-b.md").exists()
    assert (processed_dir / "doc-a.md").exists()
    assert (processed_dir / "doc-b.md").exists()
    assert mock_process_markdown.await_count == 2


@patch("src.watcher.poller._already_ingested", return_value=False)
@patch("src.watcher.poller.process_markdown")
@patch("src.watcher.poller.settings")
async def test_ingest_files_skips_duplicate_filename_across_watch_dirs(
    mock_settings,
    mock_process_markdown,
    mock_already_ingested,
    tmp_path: Path,
    processed_dir: Path,
    test_md: Path,
) -> None:
    """Same filename in two watch_dirs must not silently overwrite/mis-skip the second copy."""
    watch_a = tmp_path / "watch-a"
    watch_b = tmp_path / "watch-b"
    watch_a.mkdir()
    watch_b.mkdir()

    mock_settings.watch_dirs = [str(watch_a), str(watch_b)]
    mock_settings.processed_dir = str(processed_dir)

    (watch_a / "report.md").write_text(test_md.read_text())
    (watch_b / "report.md").write_text("# Different content\n\nThis is not the same document.")

    from src.watcher.poller import _ingest_files

    await _ingest_files()

    # First copy ingested and moved normally.
    assert not (watch_a / "report.md").exists()
    assert (processed_dir / "report.md").exists()
    # Second copy is left untouched in its own watch dir rather than being
    # silently skipped or clobbering the first copy's processed/ destination.
    assert (watch_b / "report.md").exists()
    mock_process_markdown.assert_awaited_once()


@patch("src.watcher.poller._already_ingested", return_value=True)
@patch("src.watcher.poller.process_markdown")
@patch("src.watcher.poller.settings")
async def test_ingest_files_skips_ingested(
    mock_settings,
    mock_process_markdown,
    mock_already_ingested,
    watch_dir: Path,
    processed_dir: Path,
    test_md: Path,
) -> None:
    mock_settings.watch_dirs = [str(watch_dir)]
    mock_settings.processed_dir = str(processed_dir)

    dest = watch_dir / test_md.name
    dest.write_text(test_md.read_text())

    from src.watcher.poller import _ingest_files

    await _ingest_files()

    assert not (watch_dir / test_md.name).exists()
    assert (processed_dir / test_md.name).exists()
    mock_process_markdown.assert_not_awaited()
