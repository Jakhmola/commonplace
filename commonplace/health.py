"""What `GET /health` reports: is each dependency reachable, and how big is the corpus."""

from typing import Any

import httpx
from sqlalchemy import text

from commonplace.config import settings
from commonplace.db.queries import count_active_claims, count_articles, count_passages
from commonplace.db.session import get_session
from commonplace.graph.client import check_memgraph_health, count_graph_entities


async def check_health() -> dict[str, Any]:
    """Never raises: a health check has to say which dependency is down."""
    checks: dict[str, str] = {}
    articles = passages = claims = 0

    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            articles = await count_articles(session)
            passages = await count_passages(session)
            claims = await count_active_claims(session)
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc.__class__.__name__}"

    try:
        async with httpx.AsyncClient(timeout=3.0) as http:
            (await http.get(settings.inngest_base_url)).raise_for_status()
        checks["inngest"] = "ok"
    except httpx.HTTPError as exc:
        checks["inngest"] = f"error: {exc.__class__.__name__}"

    # The graph is optional for reads, but a missing one is still worth reporting.
    checks["memgraph"] = await check_memgraph_health()
    topics = 0
    if checks["memgraph"] == "ok":
        _, topics = await count_graph_entities()

    return {
        "status": "ok" if all(v == "ok" for v in checks.values()) else "degraded",
        **checks,
        "articles": articles,
        "passages": passages,
        "claims": claims,
        "topics": topics,
    }
