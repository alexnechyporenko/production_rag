"""Centralised, environment-driven configuration for Production RAG.

All configuration values come from environment variables (or a local .env file).
The single `Settings` object is exposed through `get_settings()` and is cached
for the lifetime of the process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application configuration.

    Every field maps to an environment variable of the same name (case-insensitive).
    A local `.env` file is automatically loaded when present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database -----------------------------------------------------------
    database_url: str = Field(
        default="sqlite+aiosqlite:///:memory:",
        description="SQLAlchemy async URL. Use postgresql+asyncpg:// for pgvector.",
    )

    # --- LLM ----------------------------------------------------------------
    llm_provider: Literal["openai", "mock"] = Field(default="mock")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_api_key: str = Field(default="")
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: Literal["openai", "mock"] = Field(default="mock")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=64, ge=1, le=4096)
    embedding_base_url: str = Field(default="https://api.openai.com/v1")
    embedding_api_key: str = Field(default="")
    embedding_batch_size: int = Field(default=64, ge=1)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Retrieval ----------------------------------------------------------
    top_k: int = Field(default=10, ge=1)
    rerank_top_k: int = Field(default=5, ge=1)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    min_similarity_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    retrieval_candidate_limit: int = Field(default=100, ge=1, le=10_000)
    reranker_provider: Literal["mock", "none"] = Field(default="mock")
    reranker_model: str = Field(default="")

    # --- Context ------------------------------------------------------------
    max_context_tokens: int = Field(default=3500, ge=128)
    max_context_chunks: int = Field(default=12, ge=1)
    max_chunk_chars: int = Field(default=1200, ge=64)
    chunk_overlap_chars: int = Field(default=200, ge=0)

    # --- Caching ------------------------------------------------------------
    cache_enabled: bool = Field(default=True)
    cache_max_entries: int = Field(default=2048, ge=1)

    # --- Resource Limits ---------------------------------------------------
    max_document_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_query_chars: int = Field(default=2048, ge=16)

    # --- Observability ------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)
    redact_logs: bool = Field(default=True)

    # --- API ----------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = Field(default="*")

    # --- Pricing (USD per 1K tokens) ---------------------------------------
    # These are deliberately flat keys so we can extend without code changes.
    price_llm_input_gpt_4o_mini: float = Field(default=0.000150, ge=0)
    price_llm_output_gpt_4o_mini: float = Field(default=0.000600, ge=0)
    price_llm_input_gpt_4o: float = Field(default=0.002500, ge=0)
    price_llm_output_gpt_4o: float = Field(default=0.010000, ge=0)
    price_embedding_text_embedding_3_small: float = Field(default=0.00000002, ge=0)

    # --- Internal ----------------------------------------------------------
    # If true, the in-memory vector store is used regardless of database_url.
    # Useful for tests and for running without external infrastructure.
    force_in_memory_store: bool = Field(default=False)

    @field_validator("database_url")
    @classmethod
    def _validate_db_url(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("DATABASE_URL must not be empty")
        return v.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def llm_input_price_per_1k(self, model: str | None = None) -> float | None:
        """Return the configured input price for a model, or None if unknown.

        Returning None signals "unknown pricing" — callers MUST NOT fabricate a cost.
        """
        m = (model or self.llm_model).lower().replace("-", "_")
        return {
            "gpt_4o_mini": self.price_llm_input_gpt_4o_mini,
            "gpt_4o": self.price_llm_input_gpt_4o,
        }.get(m)

    def llm_output_price_per_1k(self, model: str | None = None) -> float | None:
        m = (model or self.llm_model).lower().replace("-", "_")
        return {
            "gpt_4o_mini": self.price_llm_output_gpt_4o_mini,
            "gpt_4o": self.price_llm_output_gpt_4o,
        }.get(m)

    def embedding_price_per_1k(self, model: str | None = None) -> float | None:
        m = (model or self.embedding_model).lower().replace("-", "_")
        return {
            "text_embedding_3_small": self.price_embedding_text_embedding_3_small,
        }.get(m)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. Intended for tests only."""
    get_settings.cache_clear()
