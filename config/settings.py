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
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")

    # Model Settings
    default_model: str = Field(default="gemini-2.5-flash", alias="DEFAULT_MODEL")
    fallback_model: str = Field(
        default="groq/llama-3.3-70b-versatile", alias="FALLBACK_MODEL"
    )
    temperature: float = Field(default=0.3, alias="TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="MAX_TOKENS")

    # API Settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: list = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8501",
            "http://localhost:7860",
            "https://aienthussp-documind-ai.hf.space",
            "https://huggingface.co",
        ],
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
