from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://commonplace:commonplace@localhost:5433/commonplace"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    inngest_dev: bool = True
    inngest_base_url: str = "http://localhost:8288"
    inngest_signing_key: str = ""
    app_port: int = 8000

    user_agent: str = "commonplace/0.1"
    request_timeout_s: float = 20.0
    max_article_bytes: int = 5_000_000

    # Characters per passage, and characters repeated between neighbours.
    chunk_size: int = 1000
    chunk_overlap: int = 150
    publications_file: str = "publications.yaml"
    # How many of a feed's newest entries add-publication ingests.
    discovery_limit: int = 5

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 96
    # text-embedding-3-small list price; embeddings have no output tokens.
    embedding_price_per_million_tokens: float = 0.02


settings = Settings()
