"""Runs migrations over the app's async engine, against the ORM metadata."""

import asyncio
from logging.config import fileConfig
from typing import Any

from sqlalchemy.engine import Connection

from alembic import context
from commonplace.db.models import Base
from commonplace.db.session import engine

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

# Created with raw SQL in the migrations, so not part of the ORM metadata.
# Without this, autogenerate and `alembic check` would want to drop them.
_RAW_SQL_INDEXES = {"ix_passages_embedding_hnsw"}


def _include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    return not (type_ == "index" and name in _RAW_SQL_INDEXES)


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=Base.metadata, include_object=_include_object
    )
    with context.begin_transaction():
        context.run_migrations()


async def _main() -> None:
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


asyncio.run(_main())
