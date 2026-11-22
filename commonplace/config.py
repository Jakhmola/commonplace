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
    max_page_bytes: int = 5_000_000

    # Characters per passage, and characters repeated between neighbours.
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # How many of a feed's newest entries add-publication ingests.
    discovery_limit: int = 5

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_usd_per_mtok: float = 0.02


settings = Settings()
