from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    neo4j_uri: str = Field(default="bolt://localhost:7688")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="changeme")
    neo4j_database: str = Field(default="neo4j")

    openai_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openai_api_key: str = Field(default="")
    llm_model: str = Field(default="openai/gpt-oss-120b")
    entity_llm_model: str = Field(default="claude-haiku-4-5-20251001")
    anthropic_api_key: str = Field(default="")
    embedding_model: str = Field(default="google/gemini-embedding-2")
    embedding_dims: int = Field(default=3072)
    vector_index_name: str = Field(default="chunk_embeddings")

    watch_dir: str = Field(default="./watch")
    processed_dir: str = Field(default="./processed")
    watch_interval_ms: int = Field(default=30000)

    log_level: str = Field(default="INFO")
    redis_url: str = Field(default="redis://default:redis@localhost:6380")

    @property
    def watch_dirs(self) -> list[str]:
        """WATCH_DIR may hold one path or a comma-separated list of paths,
        letting multiple markdown sources be polled in the same cycle."""
        return [d.strip() for d in self.watch_dir.split(",") if d.strip()]


settings = Settings()
