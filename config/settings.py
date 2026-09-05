"""
Application Configuration Settings

Loads environment variables and provides typed configuration
for the RAG Documentation Assistant.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "RAG Documentation Assistant"
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Google AI (Gemini)
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")

    # Langfuse (Observability)
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    # Redis (Caching)
    upstash_redis_url: Optional[str] = Field(default=None, alias="UPSTASH_REDIS_URL")
    upstash_redis_token: Optional[str] = Field(default=None, alias="UPSTASH_REDIS_TOKEN")

    # RAG Settings
    chunk_size: int = Field(default=1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=8, alias="TOP_K_RESULTS")

    # Ingestion limits (guards against a single request doing unbounded work)
    max_upload_files: int = Field(default=20, alias="MAX_UPLOAD_FILES")
    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB")
    max_total_upload_mb: int = Field(default=50, alias="MAX_TOTAL_UPLOAD_MB")

    # Model Settings
    default_model: str = Field(default="gemini-2.5-flash", alias="DEFAULT_MODEL")
    fallback_model: str = Field(
        default="groq/llama-3.3-70b-versatile", alias="FALLBACK_MODEL"
    )
    temperature: float = Field(default=0.3, alias="TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="MAX_TOKENS")

    # Per-provider chat model IDs. Vendors deprecate/rename models regularly, so
    # these are overridable via env rather than hard-coded in the pipeline.
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    anthropic_model: str = Field(
        default="claude-3-5-haiku-latest", alias="ANTHROPIC_MODEL"
    )
    # Groq retired the Llama 3.x chat models; openai/gpt-oss-20b is a current,
    # fast, rate-limit-friendly default. Override with GROQ_MODEL if your key
    # exposes a different set (see https://console.groq.com/docs/models).
    groq_model: str = Field(
        default="openai/gpt-oss-20b", alias="GROQ_MODEL"
    )

    # API Settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: list = Field(
        default=["*"],  # Allow all origins (HF Spaces uses dynamic subdomains)
        alias="CORS_ORIGINS"
    )

    @property
    def environment(self) -> str:
        """Get the environment name."""
        return self.app_env

    # Paths
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data directory for documents."""
        return self.project_root / "data"

    @property
    def chroma_dir(self) -> Path:
        """Get the ChromaDB persistence directory."""
        return self.project_root / "chroma_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings instance."""
    return settings
