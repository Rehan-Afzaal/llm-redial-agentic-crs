"""Application configuration using pydantic-settings."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "movies"

    # Data
    data_dir: str = "./data/movie"

    # Server
    log_level: str = "INFO"

    # RAG
    rag_top_k: int = 10
    rag_few_shot_k: int = 3

    # Streaming
    stream_delay: float = 0.0  # delay between SSE chunks (seconds)

    @property
    def data_path(self) -> Path:
        """Resolve absolute data directory path."""
        return Path(self.data_dir).resolve()

    @property
    def chroma_path(self) -> Path:
        """Resolve absolute ChromaDB persistence path."""
        return Path(self.chroma_persist_dir).resolve()


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
