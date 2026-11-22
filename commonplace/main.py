"""HTTP API. Slow work is an event (returns a handle at once); reads hit Postgres directly."""

from typing import Any, Literal

import httpx
import inngest
import inngest.fast_api
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text

from commonplace import db
from commonplace.config import settings
from commonplace.events import PUBLICATION_ADDED, PublicationAdded, event
from commonplace.functions import FUNCTIONS
from commonplace.inngest_client import client
from commonplace.sources.embed import embed

app = FastAPI(title="commonplace")
inngest.fast_api.serve(app, client, FUNCTIONS)


def _trace_url(event_id: str) -> str:
    return f"{settings.inngest_base_url}/event/{event_id}"


async def _inngest_runs(event_id: str) -> list[dict[str, Any]]:
    headers = {}
    if settings.inngest_signing_key:
        headers["Authorization"] = f"Bearer {settings.inngest_signing_key}"
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(f"{settings.inngest_base_url}/v1/events/{event_id}/runs", headers=headers)
    return list(r.json().get("data") or []) if r.status_code == 200 else []


@app.get("/health")
async def health() -> dict[str, Any]:
    out: dict[str, Any] = {"database": "ok", "inngest": "ok"}
    try:
        async with db.tx() as conn:
            await conn.execute(text("SELECT 1"))
            out |= await db.counts(conn)
    except Exception as e:  # report, don't crash the health check
        out["database"] = f"error: {e.__class__.__name__}"
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            (await c.get(settings.inngest_base_url)).raise_for_status()
    except httpx.HTTPError as e:
        out["inngest"] = f"error: {e.__class__.__name__}"
    out["status"] = "ok" if out["database"] == out["inngest"] == "ok" else "degraded"
    return out


@app.post("/publications", status_code=202)
async def add_publication(body: PublicationAdded) -> dict[str, str]:
    """Queue ingestion of a feed's newest articles. Poll GET /jobs/{event_id}."""
    (event_id,) = await client.send(event(PUBLICATION_ADDED, body))
    return {"event_id": event_id, "trace_url": _trace_url(event_id), "status": "queued"}


@app.get("/jobs/{event_id}")
async def job(event_id: str) -> dict[str, Any]:
    runs = await _inngest_runs(event_id)
    run_status = runs[0]["status"] if runs else None
    async with db.tx() as conn:
        progress = await db.job_progress(conn, event_id)
    if progress is None and run_status is None:
        raise HTTPException(404, f"no job for event {event_id!r}")

    if run_status in ("Failed", "Cancelled"):
        status = "failed"
    elif progress is None:
        status = "queued" if run_status is None else "running"
    else:
        # ponytail: an article whose ingest failed for good keeps this at "running";
        # the failures table (phase 3) is what lets us report it as failed.
        status = "completed" if progress[0] == progress[1] else "running"
    return {
        "status": status,
        "articles_done": progress[0] if progress else None,
        "articles_total": progress[1] if progress else None,
        "run_status": run_status,
        "error": (runs[0].get("output") or {}).get("message") if run_status == "Failed" else None,
        "trace_url": _trace_url(event_id),
    }


@app.get("/search")
async def search(
    q: str = Query(min_length=1),
    mode: Literal["sparse", "dense"] = "sparse",
    k: int = Query(default=5, ge=1, le=50),
) -> dict[str, Any]:
    """sparse = Postgres full-text rank, dense = embedding cosine similarity."""
    if mode == "dense":
        (vector,), _ = await embed([q])
    async with db.tx() as conn:
        hits = (
            await db.search_dense(conn, vector, k)
            if mode == "dense"
            else await db.search_sparse(conn, q, k)
        )
    return {"mode": mode, "query": q, "results": hits}
