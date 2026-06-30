"""Unit tests for _track_stats and handle_stats added to src/main.py."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _track_stats
# ---------------------------------------------------------------------------

class TestTrackStats:
    @pytest.fixture
    def redis_client(self):
        client = MagicMock()
        pipe = AsyncMock()
        pipe.incr = MagicMock()
        pipe.execute = AsyncMock(return_value=[1, 1])
        client.pipeline.return_value = pipe
        return client, pipe

    async def test_increments_questions_on_matched(self, redis_client):
        from src.main import _track_stats
        client, pipe = redis_client
        await _track_stats(client, unmatched=False)
        pipe.incr.assert_called_once_with("stats:onn-ai:questions_asked")
        pipe.execute.assert_awaited_once()

    async def test_increments_both_on_unmatched(self, redis_client):
        from src.main import _track_stats
        client, pipe = redis_client
        await _track_stats(client, unmatched=True)
        calls = [c.args[0] for c in pipe.incr.call_args_list]
        assert "stats:onn-ai:questions_asked" in calls
        assert "stats:onn-ai:maklum_balas" in calls

    async def test_logs_warning_on_redis_failure(self, redis_client):
        from src.main import _track_stats
        client, pipe = redis_client
        pipe.execute.side_effect = Exception("Redis connection lost")
        with patch("src.main.logger") as mock_log:
            await _track_stats(client, unmatched=False)
            mock_log.warning.assert_called_once()
            assert "stats" in mock_log.warning.call_args.args[0].lower()


# ---------------------------------------------------------------------------
# handle_stats
# ---------------------------------------------------------------------------

class TestHandleStats:
    async def test_returns_nulls_when_redis_none(self):
        import src.main as main_mod
        original = main_mod._redis_client
        main_mod._redis_client = None
        try:
            from starlette.requests import Request
            scope = {"type": "http", "method": "GET", "path": "/stats",
                     "query_string": b"", "headers": []}
            req = Request(scope)
            res = await main_mod.handle_stats(req)
            assert res.status_code == 200
            import json
            body = json.loads(res.body)
            assert body == {"questions": None, "maklumBalas": None}
        finally:
            main_mod._redis_client = original

    async def test_returns_counts_from_redis(self):
        import src.main as main_mod
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["17", "3"])
        original = main_mod._redis_client
        main_mod._redis_client = mock_redis
        try:
            from starlette.requests import Request
            scope = {"type": "http", "method": "GET", "path": "/stats",
                     "query_string": b"", "headers": []}
            req = Request(scope)
            with patch("asyncio.gather", new=AsyncMock(return_value=("17", "3"))):
                res = await main_mod.handle_stats(req)
            import json
            body = json.loads(res.body)
            assert body["questions"] == 17
            assert body["maklumBalas"] == 3
        finally:
            main_mod._redis_client = original

    async def test_returns_zeros_for_unset_keys(self):
        import src.main as main_mod
        original = main_mod._redis_client
        main_mod._redis_client = AsyncMock()
        try:
            from starlette.requests import Request
            scope = {"type": "http", "method": "GET", "path": "/stats",
                     "query_string": b"", "headers": []}
            req = Request(scope)
            with patch("asyncio.gather", new=AsyncMock(return_value=(None, None))):
                res = await main_mod.handle_stats(req)
            import json
            body = json.loads(res.body)
            assert body["questions"] == 0
            assert body["maklumBalas"] == 0
        finally:
            main_mod._redis_client = original

    async def test_returns_nulls_on_redis_error(self):
        import src.main as main_mod
        original = main_mod._redis_client
        main_mod._redis_client = AsyncMock()
        try:
            from starlette.requests import Request
            scope = {"type": "http", "method": "GET", "path": "/stats",
                     "query_string": b"", "headers": []}
            req = Request(scope)
            with patch("asyncio.gather", new=AsyncMock(side_effect=Exception("timeout"))):
                res = await main_mod.handle_stats(req)
            import json
            body = json.loads(res.body)
            assert body == {"questions": None, "maklumBalas": None}
        finally:
            main_mod._redis_client = original
