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
    embedding_model: str = Field(default="google/gemini-embedding-2")
    embedding_dims: int = Field(default=3072)

    watch_dir: str = Field(default="./watch")
    processed_dir: str = Field(default="./processed")
    watch_interval_ms: int = Field(default=30000)

    log_level: str = Field(default="INFO")


settings = Settings()
