"""Keyword retrieval: Postgres full-text rank."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.queries import search_passages_bm25


async def search(session: AsyncSession, query: str, limit: int) -> list[dict[str, Any]]:
    return await search_passages_bm25(session, query, limit)
