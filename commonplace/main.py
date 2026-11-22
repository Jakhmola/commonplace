"""HTTP API. Slow work is an event (returns a handle at once); reads hit Postgres directly."""

from typing import Any, Literal

import httpx
import inngest.fast_api
from fastapi import FastAPI, HTTPException, Query

from commonplace.config import settings
from commonplace.db.queries import get_job_status
from commonplace.db.session import get_session
from commonplace.functions import FUNCTIONS
from commonplace.health import check_health
from commonplace.inngest_client import client
from commonplace.retrieval import dense, sparse
from commonplace.schemas.events import PUBLICATION_ADDED, PublicationAdded, event

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
    mode: Literal["sparse", "dense"] = "sparse",
    limit: int = Query(default=5, ge=1, le=25),
) -> dict[str, Any]:
    """Two retrievers, queried on their own: keyword rank or embedding similarity."""
    retriever = sparse if mode == "sparse" else dense
    async with get_session() as session:
        results = await retriever.search(session, q, limit)
    return {"mode": mode, "results": results}
