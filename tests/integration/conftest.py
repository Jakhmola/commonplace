"""Fixtures for tests that need the real Postgres from `make up`.

The database is never mocked. Each test runs inside one outer transaction
that is always rolled back: every `get_session()` in the code under test
joins that connection through a savepoint, so an inner `commit()` only
releases the savepoint. Tests leave nothing behind and can run against the
dev database as often as you like.

The whole directory is skipped when Postgres is not reachable.
"""

import socket
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import commonplace.db.session as db_session
from commonplace.config import settings
from commonplace.db.queries import upsert_article, upsert_publication
from commonplace.graph import client as graph_client


def _reachable(url: str) -> bool:
    parts = urlsplit(url.replace("postgresql+asyncpg", "postgresql"))
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), 1):
            return True
    except OSError:
        return False


POSTGRES_UP = _reachable(settings.database_url)
INNGEST_UP = _reachable(settings.inngest_base_url)
MEMGRAPH_UP = _reachable(settings.memgraph_uri)


@pytest.fixture(autouse=True)
def _needs_postgres() -> None:
    if not POSTGRES_UP:
        pytest.skip("Postgres is not running (make up)")


@pytest.fixture(autouse=True)
async def _fresh_graph_driver() -> AsyncIterator[None]:
    """The graph driver belongs to the event loop that created it, and each test
    has its own loop: close it after every test that may have used it.
    """
    yield
    await graph_client.close_driver()


@pytest.fixture
async def db(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        outer = await connection.begin()
        factory = async_sessionmaker(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        monkeypatch.setattr(db_session, "async_session_factory", factory)
        async with factory() as session:
            try:
                yield session
            finally:
                await outer.rollback()
    await engine.dispose()


ARTICLE_URL = "https://integration-test.example/p/a-seeded-post"


@pytest.fixture
async def article_id(db: AsyncSession) -> uuid.UUID:
    """A publication and one article with no passages, gone again after the test."""
    publication_id = await upsert_publication(
        db,
        slug="integration-test",
        name="Integration Test",
        feed_url="https://integration-test.example/feed",
        homepage="https://integration-test.example",
    )
    return await upsert_article(
        db,
        canonical_id="pub:integration-test:a-seeded-post",
        publication_id=publication_id,
        url=ARTICLE_URL,
        title="A seeded post",
        author=None,
        published_at=datetime(2026, 11, 17, tzinfo=UTC),
        content_hash="v1",
        text="first version",
    )


def passages(*texts: str) -> list[dict[str, Any]]:
    return [{"chunk_index": i, "text": t, "embedding": [0.1] * 1536} for i, t in enumerate(texts)]
