"""The claim graph in a real Memgraph: syncing converges, and old claims stop being found.

Skipped when Memgraph is not running. Every node these tests create hangs off
one throwaway article and is deleted afterwards.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from commonplace.graph import client
from commonplace.graph.ontology import AuthorRef
from tests.integration.conftest import MEMGRAPH_UP

pytestmark = pytest.mark.skipif(not MEMGRAPH_UP, reason="Memgraph is not running (make up)")

CANONICAL_ID = "pub:graphtest:a-seeded-post"
PUBLICATION = {"slug": "graphtest", "name": "Graph Test", "homepage": "https://graphtest.example"}
AUTHOR = AuthorRef("graphtest-author", "Graph Test Author")
ARTICLE = {
    "canonical_id": CANONICAL_ID,
    "article_id": "6f1c0a3e-9d5b-4c55-8f0e-2b7a4d1c9e10",
    "url": "https://graphtest.example/p/a-seeded-post",
    "title": "A seeded post",
    "published_at": "2026-11-17T00:00:00+00:00",
    "content_hash": "v1",
}
CLEANUP = """
MATCH (n)
WHERE (n:Article AND n.canonical_id = $canonical_id)
   OR (n:Claim AND n.claim_id STARTS WITH 'graphtest-')
   OR (n:Topic AND n.slug STARTS WITH 'graphtest-')
   OR (n:Publication AND n.slug = 'graphtest')
   OR (n:Author AND n.slug = 'graphtest-author')
DETACH DELETE n
"""


def _claim(n: int, topic: str) -> dict[str, Any]:
    return {
        "claim_id": f"graphtest-{n}",
        "article_id": ARTICLE["article_id"],
        "text": f"Claim number {n}",
        "quote_span": f"the exact words of claim number {n}",
        "claim_type": "opinion",
        "confidence": 0.9,
        "valid_from": "2026-11-23T12:00:00+00:00",
        "topics": [{"slug": f"graphtest-{topic}", "name": f"Graphtest {topic.title()}"}],
    }


@pytest.fixture
async def graph() -> AsyncIterator[None]:
    await client.execute_cypher(CLEANUP, {"canonical_id": CANONICAL_ID})
    yield
    await client.execute_cypher(CLEANUP, {"canonical_id": CANONICAL_ID})
    await client.close_driver()  # the driver belongs to this test's event loop


async def _sync(claims: list[dict[str, Any]], pairs: list[dict[str, str]] | None = None) -> None:
    await client.sync_article_claims_to_graph(
        publication=PUBLICATION,
        author=AUTHOR,
        article=ARTICLE,
        claims=claims,
        supersedes_pairs=pairs or [],
        now_iso="2026-11-24T00:00:00+00:00",
    )


async def _counts() -> dict[str, int]:
    rows = await client.execute_cypher(
        """
        MATCH (art:Article {canonical_id: $canonical_id})
        OPTIONAL MATCH (art)-[:ASSERTS]->(c:Claim)
        OPTIONAL MATCH (c)-[:ABOUT]->(t:Topic)
        OPTIONAL MATCH (art)-[:PUBLISHED_IN]->(p:Publication)
        OPTIONAL MATCH (art)-[:WRITTEN_BY]->(au:Author)
        RETURN count(DISTINCT art) AS articles, count(DISTINCT c) AS claims,
               count(DISTINCT t) AS topics, count(DISTINCT p) AS publications,
               count(DISTINCT au) AS authors
        """,
        {"canonical_id": CANONICAL_ID},
    )
    return rows[0]


async def _search(topic: str) -> list[str]:
    rows = await client.search_graph_by_topics(
        topic_slugs=[f"graphtest-{topic}"], search_terms=[], limit=10
    )
    return [row["claim_id"] for row in rows]


async def test_syncing_twice_creates_each_node_once(graph: None) -> None:
    claims = [_claim(1, "alpha"), _claim(2, "alpha"), _claim(3, "beta")]
    await _sync(claims)
    first = await _counts()
    assert first == {"articles": 1, "claims": 3, "topics": 2, "publications": 1, "authors": 1}

    await _sync(claims)  # a replay of the same extraction
    assert await _counts() == first
    assert sorted(await _search("alpha")) == ["graphtest-1", "graphtest-2"]


async def test_superseded_claims_leave_the_search_results(graph: None) -> None:
    await _sync([_claim(1, "alpha")])
    assert await _search("alpha") == ["graphtest-1"]

    # the article was edited: a new claim on the same topic replaces the old one
    pair = {
        "new_claim_id": "graphtest-2",
        "old_claim_id": "graphtest-1",
        "at": "2026-11-24T00:00:00+00:00",
    }
    await _sync([_claim(2, "alpha")], [pair])

    assert await _search("alpha") == ["graphtest-2"]  # only the current claim is found
    history = await client.execute_cypher(
        "MATCH (new:Claim {claim_id: 'graphtest-2'})-[:SUPERSEDES]->(old:Claim) "
        "RETURN old.claim_id AS old, old.valid_to AS closed"
    )
    assert history == [{"old": "graphtest-1", "closed": "2026-11-24T00:00:00+00:00"}]


async def test_health_and_counts_report_the_graph(graph: None) -> None:
    assert await client.check_memgraph_health() == "ok"
    before_claims, before_topics = await client.count_graph_entities()
    await _sync([_claim(1, "alpha"), _claim(2, "beta")])
    assert await client.count_graph_entities() == (before_claims + 2, before_topics + 2)
