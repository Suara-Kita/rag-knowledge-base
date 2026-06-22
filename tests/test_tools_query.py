import pytest

from src.tools.query import get_tools


def test_get_tools_returns_correct_tools() -> None:
    tools = get_tools()
    names = {t.name for t in tools}
    assert "query_knowledge" in names
    assert "list_documents" in names


def test_query_knowledge_schema() -> None:
    tools = get_tools()
    qt = next(t for t in tools if t.name == "query_knowledge")

    assert qt.description
    props = qt.inputSchema.get("properties", {})
    assert "question" in props
    assert props["question"]["type"] == "string"
    assert "question" in qt.inputSchema.get("required", [])

    assert "top_k" in props
    assert props["top_k"]["type"] == "integer" or props["top_k"].get("default") == 5


@pytest.mark.asyncio
async def test_list_documents_empty() -> None:
    from unittest.mock import MagicMock, patch

    mock_session = MagicMock()
    mock_session.run.return_value.data.return_value = []
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("src.tools.query.get_driver", return_value=mock_driver):
        from src.tools.query import handle_call_tool

        result = await handle_call_tool("list_documents", {})
        assert len(result) == 1
        assert "No documents ingested yet" in result[0].text


@pytest.mark.asyncio
async def test_handle_unknown_tool() -> None:
    from src.tools.query import handle_call_tool

    with pytest.raises(ValueError, match="foo"):
        await handle_call_tool("foo", {})
