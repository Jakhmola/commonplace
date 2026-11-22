from typing import Literal

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
    # backfill-publication sends cp/article.discovered in batches of this size.
    backfill_batch_size: int = 25

    # --- Flow control on ingest-article ---
    # Articles of the same publication in flight at once.
    concurrency_per_publication: int = 3
    # Runs that may start per minute across all publications. One run makes about
    # one embedding request, so this is roughly "embedding requests per minute".
    ingest_runs_per_minute: int = 50
    # Inngest priority is a time shift in seconds (-600..600), not a rank. A manual
    # run is queued as if it had arrived 600 s earlier, so it overtakes whatever was
    # queued in the last ten minutes. Everything else stays at 0: a negative value
    # holds the run for that long even when the queue is empty.
    priority_manual_s: int = 600
    priority_backfill_s: int = 0

    # --- Retrieval and answers ---
    rrf_k: int = 60
    retrieval_limit: int = 5
    synthesis_model: str = "claude-opus-5-5"
    # Thinking tokens count against this too, so it is a ceiling, not a target length.
    synthesis_max_tokens: int = 8000
    # --- Claim extraction and the graph ---
    extraction_model: str = "claude-opus-5-5"
    extraction_max_tokens: int = 16000
    extraction_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Extraction runs in flight at once, across all articles.
    concurrency_extract_claims: int = 2
    memgraph_uri: str = "bolt://localhost:7687"
    memgraph_user: str = ""
    memgraph_password: str = ""
    graph_retrieval_limit: int = 15

    synthesis_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 96
    # text-embedding-3-small list price; embeddings have no output tokens.
    embedding_price_per_million_tokens: float = 0.02


settings = Settings()
