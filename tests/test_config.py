from src.config import Settings


def test_defaults(monkeypatch) -> None:
    for var in ("NEO4J_URI", "WATCH_INTERVAL_MS", "LLM_MODEL", "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.neo4j_uri == "bolt://localhost:7688"
    assert s.watch_interval_ms == 30000
    assert s.llm_model == "openai/gpt-oss-120b"


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://custom:9999")
    monkeypatch.setenv("WATCH_INTERVAL_MS", "5000")

    s = Settings()
    assert s.neo4j_uri == "bolt://custom:9999"
    assert s.watch_interval_ms == 5000


def test_watch_dirs_single_path() -> None:
    s = Settings(_env_file=None, watch_dir="./watch")
    assert s.watch_dirs == ["./watch"]


def test_watch_dirs_multiple_paths() -> None:
    s = Settings(_env_file=None, watch_dir="./watch, ./watch-johor-economy ,./watch-flood")
    assert s.watch_dirs == ["./watch", "./watch-johor-economy", "./watch-flood"]
