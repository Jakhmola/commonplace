"""Memgraph over Bolt. Thin: a driver, one way to run a statement, and the few
graph operations the rest of the project needs.

The graph is derived data. Postgres holds the truth (the `claims` table); the
graph is rebuilt from it by merging, so losing it costs a re-sync, not data.
"""

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from commonplace.config import settings
from commonplace.graph import cypher
from commonplace.graph.ontology import AuthorRef

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None
_schema_ready = False


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        auth = (settings.memgraph_user, settings.memgraph_password)
        _driver = AsyncGraphDatabase.driver(
            settings.memgraph_uri,
            auth=auth if settings.memgraph_user else None,
            connection_timeout=3.0,
        )
    return _driver


async def close_driver() -> None:
    global _driver, _schema_ready
    if _driver is not None:
        await _driver.close()
    _driver, _schema_ready = None, False


async def execute_cypher(
    statement: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    async with get_driver().session() as session:
        result = await session.run(statement, params or {})
        return await result.data()


async def ensure_graph_schema() -> None:
    """Create the indexes and uniqueness constraints once per process."""
    global _schema_ready
    if _schema_ready:
        return
    for statement in cypher.SCHEMA_STATEMENTS:
        try:
            await execute_cypher(statement)
        except Exception as exc:  # noqa: BLE001
            # Memgraph reports an existing index or constraint as an error.
            logger.debug("Schema statement skipped (%s): %s", statement, exc)
    _schema_ready = True


async def check_memgraph_health() -> str:
    try:
        await execute_cypher("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc.__class__.__name__}"
    return "ok"


async def count_graph_entities() -> tuple[int, int]:
    """(current claims, topics those claims are about)."""
    (row,) = await execute_cypher(cypher.COUNT_ACTIVE_CLAIMS_AND_TOPICS)
    return int(row["claims"]), int(row["topics"])


async def sync_article_claims_to_graph(
    *,
    publication: dict[str, str],
    author: AuthorRef,
    article: dict[str, str],
    claims: list[dict[str, Any]],
    supersedes_pairs: list[dict[str, str]],
    now_iso: str,
) -> dict[str, int]:
    """Bring one article's part of the graph in line with its current claims.

    Everything is merged on keys, so running this again with the same input
    changes nothing. Claims of the article that are not in `claims` are closed.
    """
    await ensure_graph_schema()
    await execute_cypher(
        cypher.MERGE_ARTICLE_SKELETON,
        {
            "pub_slug": publication["slug"],
            "pub_name": publication["name"],
            "pub_homepage": publication["homepage"],
            "author_slug": author.slug,
            "author_name": author.name,
            **article,
        },
    )
    closed = await execute_cypher(
        cypher.SUPERSEDE_ARTICLE_CLAIMS,
        {
            "canonical_id": article["canonical_id"],
            "active_claim_ids": [c["claim_id"] for c in claims],
            "now": now_iso,
        },
    )
    if claims:
        await execute_cypher(
            cypher.MERGE_CLAIMS_BATCH,
            {"canonical_id": article["canonical_id"], "claims": claims},
        )
    if supersedes_pairs:
        await execute_cypher(cypher.MERGE_SUPERSEDES_EDGES, {"pairs": supersedes_pairs})
    return {
        "synced_claims": len(claims),
        "superseded_claims": int(closed[0]["superseded"]) if closed else 0,
        "supersedes_edges": len(supersedes_pairs),
    }


async def search_graph_by_topics(
    *, topic_slugs: list[str], search_terms: list[str], limit: int
) -> list[dict[str, Any]]:
    """Current claims about the given topics, best match first."""
    return await execute_cypher(
        cypher.SEARCH_CLAIMS_BY_TOPICS,
        {"topic_slugs": topic_slugs, "search_terms": search_terms, "limit": limit},
    )


async def topics_shared_across_publications(limit: int = 20) -> list[dict[str, Any]]:
    return await execute_cypher(cypher.TOPICS_SHARED_ACROSS_PUBLICATIONS, {"limit": limit})
