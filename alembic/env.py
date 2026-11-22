"""Migrations are hand-written SQL (no ORM models), run over the app's async engine."""

import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context
from commonplace.db import engine

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)


def _run(conn: Connection) -> None:
    context.configure(connection=conn, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


async def _main() -> None:
    async with engine.connect() as conn:
        await conn.run_sync(_run)
    await engine.dispose()


asyncio.run(_main())
