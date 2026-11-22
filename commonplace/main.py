"""HTTP API. Slow work is an event (returns a handle at once); reads hit Postgres directly."""

from datetime import UTC, datetime
from typing import Any, Literal

import httpx
import inngest.fast_api
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from commonplace.config import settings
from commonplace.db.queries import get_job_status, upsert_publication
from commonplace.db.session import get_session
from commonplace.functions import FUNCTIONS
from commonplace.health import check_health
from commonplace.inngest_client import client
from commonplace.retrieval import dense, hybrid, sparse
from commonplace.schemas.events import (
    ARTICLE_DISCOVERED,
    PUBLICATION_ADDED,
    ArticleDiscovered,
    PublicationAdded,
    event,
)
from commonplace.schemas.ids import canonical_id, clean_url
from commonplace.sources.allowlist import PublicationEntry, assert_allowed, load_publications

app = FastAPI(title="commonplace")
inngest.fast_api.serve(app, client, FUNCTIONS)


def _trace_url(event_id: str) -> str:
    return f"{settings.inngest_base_url}/event/{event_id}"


async def _inngest_runs(event_id: str) -> list[dict[str, Any]]:
    """The runs an event started, as the Inngest server reports them."""
    headers = {}
    if settings.inngest_signing_key:
        headers["Authorization"] = f"Bearer {settings.inngest_signing_key}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            response = await http.get(
                f"{settings.inngest_base_url}/v1/events/{event_id}/runs", headers=headers
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, f"Could not reach Inngest at {settings.inngest_base_url}: {exc}"
        ) from exc
    return list(response.json().get("data") or []) if response.status_code == 200 else []


def run_summary(runs: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    """(run status, failure messages) for the first run an event started.

    Two things the raw API gets wrong for a caller: a run that is waiting to
    retry is labelled "Failed" although it has not ended, and a failure's
    output carries a stack trace with local paths. Only the message is returned.
    """
    if not runs:
        return None, []
    run = runs[0]
    if run["status"] != "Failed":
        return run["status"], []
    if run.get("ended_at") is None:
        return "Running", []
    output = run.get("output")
    message = output.get("message") if isinstance(output, dict) else output
    return "Failed", [str(message or "unknown error")]


@app.get("/health")
async def health() -> dict[str, Any]:
    return await check_health()


@app.post("/publications")
async def add_publication(payload: PublicationAdded) -> dict[str, str]:
    """Queue ingestion of an allowlisted feed's newest articles. Poll GET /jobs/{event_id}."""
    (event_id,) = await client.send(event(PUBLICATION_ADDED, payload))
    return {"event_id": event_id, "trace_url": _trace_url(event_id), "status": "queued"}


class ArticleManual(BaseModel):
    url: str
    title: str | None = None


def resolve_manual_article(
    url: str, publications: list[PublicationEntry]
) -> tuple[PublicationEntry, str]:
    """(publication, canonical id) for a hand-submitted URL, or ValueError if it is unusable."""
    pub = assert_allowed(url, publications)
    return pub, canonical_id(pub.slug, url)


@app.post("/articles")
async def add_article(payload: ArticleManual) -> dict[str, Any]:
    """Ingest one article now, ahead of any queued backfill.

    The event carries `via="manual"`, which the priority expression on
    `ingest-article` turns into a head start: the run takes the next free slot
    for its publication. The URL's host must be on the allowlist.
    """
    try:
        pub, cid = resolve_manual_article(payload.url, load_publications())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    async with get_session() as session:
        publication_id = await upsert_publication(
            session, slug=pub.slug, name=pub.name, feed_url=pub.feed_url, homepage=pub.homepage
        )
        await session.commit()

    discovered = ArticleDiscovered(
        canonical_id=cid,
        publication_id=publication_id,
        url=clean_url(payload.url),
        # Empty unless the caller names it: ingest then falls back to the page's own title.
        title=payload.title or "",
        # A fallback only: the parse step prefers the date on the page.
        published_at=datetime.now(UTC),
        via="manual",
    )
    (event_id,) = await client.send(event(ARTICLE_DISCOVERED, discovered))
    return {
        "event_id": event_id,
        "trace_url": _trace_url(event_id),
        "status": "queued",
        "priority_offset_s": settings.priority_manual_s,
    }


@app.get("/jobs/{event_id}")
async def job_status(event_id: str) -> dict[str, Any]:
    """Our own job row (article counts) combined with Inngest's status for the run."""
    run_status, failures = run_summary(await _inngest_runs(event_id))

    async with get_session() as session:
        job = await get_job_status(session, event_id)
        await session.commit()  # keeps the recounted done/status

    if job is None and run_status is None:
        raise HTTPException(404, f"No job or Inngest run found for event_id={event_id!r}.")

    return {
        "status": "failed" if failures else (job["status"] if job else run_status),
        "articles_done": job["done"] if job else None,
        "articles_total": job["total"] if job else None,
        "run_status": run_status,
        "failures": failures,
        "trace_url": _trace_url(event_id),
    }


@app.get("/search")
async def search(
    q: str = Query(min_length=1),
    mode: Literal["hybrid", "sparse", "dense"] = "hybrid",
    limit: int = Query(default=5, ge=1, le=25),
) -> dict[str, Any]:
    """Passages for a query, no LLM. `hybrid` fuses the two engines by rank;
    `sparse` and `dense` ask one engine each, to compare all three on one query.
    """
    retriever = {"hybrid": hybrid, "sparse": sparse, "dense": dense}[mode]
    async with get_session() as session:
        results = await retriever.search(session, q, limit)
    return {"mode": mode, "results": results}
