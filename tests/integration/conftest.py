"""Fixtures for tests that need the real Postgres from `make up`.

The database is never mocked. Each test runs inside one outer transaction
that is always rolled back: every `get_session()` in the code under test
joins that connection through a savepoint, so an inner `commit()` only
releases the savepoint. Tests leave nothing behind and can run against the
dev database as often as you like.

The whole directory is skipped when Postgres is not reachable.
"""

import socket
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import commonplace.db.session as db_session
from commonplace.config import settings


def _reachable(url: str) -> bool:
    parts = urlsplit(url.replace("postgresql+asyncpg", "postgresql"))
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), 1):
            return True
    except OSError:
        return False


POSTGRES_UP = _reachable(settings.database_url)
INNGEST_UP = _reachable(settings.inngest_base_url)


@pytest.fixture(autouse=True)
def _needs_postgres() -> None:
    if not POSTGRES_UP:
        pytest.skip("Postgres is not running (make up)")


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
