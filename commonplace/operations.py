"""What the knowledge base can be asked to do, independent of who is asking.

The HTTP API (`main.py`) and the MCP tools (`mcp/tools.py`) both call these
functions and only translate `OperationError` into their own kind of error.
Slow work is sent to Inngest as an event and returns a handle at once; reads
go straight to Postgres.
"""

import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
import inngest
import openai

from commonplace.config import settings
from commonplace.db.queries import (
    get_job_status,
    get_source_provenance,
    list_publications_with_stats,
    upsert_publication,
)
from commonplace.db.session import get_session
from commonplace.functions.ask import ask_question
from commonplace.inngest_client import client
from commonplace.retrieval import dense, hybrid, sparse
from commonplace.schemas.events import (
    ARTICLE_DISCOVERED,
    FEED_POLL,
    PUBLICATION_ADDED,
    ArticleDiscovered,
    FeedPoll,
    PublicationAdded,
    event,
)
from commonplace.schemas.ids import canonical_id, clean_url
from commonplace.sources.allowlist import PublicationEntry, assert_allowed, load_publications

_RETRIEVERS = {"hybrid": hybrid, "sparse": sparse, "dense": dense}


class OperationError(Exception):
    """A request that cannot be carried out. `status` follows HTTP: 400, 404 or 502."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _handle(event_id: str, **extra: Any) -> dict[str, Any]:
    trace_url = f"{settings.inngest_base_url}/event/{event_id}"
    return {"event_id": event_id, "trace_url": trace_url, **{"status": "queued", **extra}}


# --- writes: send an event, return a handle ---


async def _send(events: inngest.Event | list[inngest.Event]) -> list[str]:
    try:
        return await client.send(events)
    except Exception as exc:
        raise OperationError(
            502, f"Could not reach Inngest at {settings.inngest_base_url}: {exc}"
        ) from exc


async def queue_publication(feed_url: str) -> dict[str, Any]:
    """Queue a backfill of one feed. An unlisted feed is rejected by the function itself."""
    (event_id,) = await _send(event(PUBLICATION_ADDED, PublicationAdded(feed_url=feed_url)))
    return _handle(event_id)


def _publications(named: str | None) -> list[PublicationEntry]:
    """The allowlisted publications a caller means: one by slug or feed URL, or all of them."""
    publications = load_publications()
    if named in (None, "all"):
        return publications
    chosen = [p for p in publications if named in (p.slug, p.feed_url)]
    if not chosen:
        raise OperationError(404, f"Publication {named!r} is not in publications.yaml")
    return chosen


async def queue_backfill(publication: str = "all") -> dict[str, Any]:
    """Queue a backfill for one publication (slug or feed URL) or for all of them."""
    chosen = _publications(publication)
    event_ids = await _send(
        [event(PUBLICATION_ADDED, PublicationAdded(feed_url=p.feed_url)) for p in chosen]
    )
    return {
        "status": "queued",
        "jobs": [
            {"publication": p.slug, **_handle(event_id)}
            for p, event_id in zip(chosen, event_ids, strict=True)
        ],
    }


async def trigger_poll(publication: str | None = None) -> dict[str, Any]:
    """Check feeds for new articles now instead of waiting for the schedule."""
    slug = None if publication in (None, "all") else _publications(publication)[0].slug
    (event_id,) = await _send(event(FEED_POLL, FeedPoll(publication=slug)))
    return _handle(event_id, status="poll_triggered")


def resolve_manual_article(
    url: str, publications: list[PublicationEntry]
) -> tuple[PublicationEntry, str]:
    """(publication, canonical id) for a hand-submitted URL, or ValueError if it is unusable."""
    if urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"Only http(s) URLs can be ingested: {url!r}")
    pub = assert_allowed(url, publications)
    return pub, canonical_id(pub.slug, url)


async def queue_article(url: str, title: str | None = None) -> dict[str, Any]:
    """Queue one article with `via="manual"`, which lets it overtake a queued backfill."""
    try:
        pub, cid = resolve_manual_article(url, load_publications())
    except ValueError as exc:
        raise OperationError(400, str(exc)) from exc

    async with get_session() as session:
        publication_id = await upsert_publication(
            session, slug=pub.slug, name=pub.name, feed_url=pub.feed_url, homepage=pub.homepage
        )
        await session.commit()

    discovered = ArticleDiscovered(
        canonical_id=cid,
        publication_id=publication_id,
        url=clean_url(url),
        # Empty unless the caller names it: ingest then falls back to the page's own title.
        title=title or "",
        # A fallback only: the parse step prefers the date on the page.
        published_at=datetime.now(UTC),
        via="manual",
    )
    (event_id,) = await _send(event(ARTICLE_DISCOVERED, discovered))
    return _handle(event_id, priority_offset_s=settings.priority_manual_s)


# --- job status ---


async def _inngest_get(path: str) -> httpx.Response:
    headers = {}
    if settings.inngest_signing_key:
        headers["Authorization"] = f"Bearer {settings.inngest_signing_key}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            return await http.get(f"{settings.inngest_base_url}{path}", headers=headers)
    except httpx.HTTPError as exc:
        raise OperationError(
            502, f"Could not reach Inngest at {settings.inngest_base_url}: {exc}"
        ) from exc


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


def job_state(job: dict[str, Any] | None, run_status: str | None, failures: list[str]) -> str:
    """One lowercase word for a job: queued, running, completed or failed."""
    if failures or run_status == "Cancelled":
        return "failed"
    if job is not None:
        return str(job["status"])
    # No job row yet: the function has not reached the step that writes it.
    return {"Completed": "completed", None: "queued"}.get(run_status, "running")


async def job_status(event_id: str) -> dict[str, Any]:
    """Our own job row (article counts) combined with Inngest's status for the run."""
    response = await _inngest_get(f"/v1/events/{event_id}/runs")
    runs = list(response.json().get("data") or []) if response.status_code == 200 else []
    run_status, failures = run_summary(runs)

    async with get_session() as session:
        job = await get_job_status(session, event_id)
        await session.commit()  # keeps the recounted done/status

    # Right after a request the event exists but its run may not have started yet.
    unknown = job is None and run_status is None
    if unknown and (await _inngest_get(f"/v1/events/{event_id}")).status_code != 200:
        raise OperationError(404, f"No job or Inngest run found for event_id={event_id!r}.")

    return {
        "status": job_state(job, run_status, failures),
        "articles_done": job["done"] if job else None,
        "articles_total": job["total"] if job else None,
        "articles_failed": job["failed"] if job else None,
        "run_status": run_status,
        # Why the triggering run failed, then why any of its articles did.
        "failures": failures + (job["errors"] if job else []),
        "trace_url": f"{settings.inngest_base_url}/event/{event_id}",
    }


# --- reads ---


async def search(query: str, mode: str = "hybrid", limit: int = 5) -> list[dict[str, Any]]:
    """Passages for a query, no LLM. `hybrid` fuses the two engines by rank;
    `sparse` and `dense` ask one engine each.
    """
    try:
        async with get_session() as session:
            return await _RETRIEVERS[mode].search(session, query, limit)
    except openai.OpenAIError as exc:
        # Only `dense` gets here: hybrid falls back to keywords on its own.
        raise OperationError(502, f"The embedding provider failed: {exc}") from exc


async def ask(question: str, limit: int | None = None) -> dict[str, Any]:
    """An answer with inline citations, from retrieved passages only."""
    async with get_session() as session:
        return await ask_question(session, question, limit)


async def list_publications() -> list[dict[str, Any]]:
    async with get_session() as session:
        return await list_publications_with_stats(session)


async def get_source(claim_or_passage_id: str) -> dict[str, Any]:
    """Trace a cited claim or passage to its article, exact text and validity."""
    try:
        target = uuid.UUID(claim_or_passage_id.strip())
    except ValueError as exc:
        raise OperationError(400, f"Not a valid id: {claim_or_passage_id!r}") from exc
    async with get_session() as session:
        source = await get_source_provenance(session, target)
    if source is None:
        raise OperationError(404, f"No claim or passage has the id {claim_or_passage_id!r}")
    return source
