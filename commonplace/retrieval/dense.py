"""Semantic retrieval: embed the query, then nearest passages by cosine similarity."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.queries import search_passages_dense
from commonplace.sources.embedder import embed_texts


async def search(session: AsyncSession, query: str, limit: int) -> list[dict[str, Any]]:
    (query_embedding,) = (await embed_texts([query]))["embeddings"]
    return await search_passages_dense(session, query_embedding, limit)
