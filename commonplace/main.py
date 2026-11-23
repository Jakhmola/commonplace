"""HTTP API: a thin layer over `operations.py`, plus the Inngest function endpoint."""

import secrets
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import inngest.fast_api
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from commonplace import operations
from commonplace.config import settings
from commonplace.functions import FUNCTIONS
from commonplace.health import check_health
from commonplace.inngest_client import client
from commonplace.schemas.events import PublicationAdded

app = FastAPI(title="commonplace")
inngest.fast_api.serve(app, client, FUNCTIONS)


@app.middleware("http")
async def require_token(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """With API_TOKEN set, every route needs it, except the health probe and the
    Inngest endpoint (whose requests Inngest signs and the SDK verifies).
    """
    path = request.url.path
    if settings.api_token and path != "/health" and not path.startswith("/api/inngest"):
        supplied = request.headers.get("authorization", "").encode()
        if not secrets.compare_digest(supplied, f"Bearer {settings.api_token}".encode()):
            return JSONResponse(status_code=401, content={"detail": "A bearer token is required"})
    return await call_next(request)


@app.exception_handler(operations.OperationError)
async def operation_error(_: Request, exc: operations.OperationError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, Any]:
    return await check_health()


@app.post("/publications")
async def add_publication(payload: PublicationAdded) -> dict[str, Any]:
    """Queue ingestion of every post in an allowlisted feed. Poll GET /jobs/{event_id}."""
    return await operations.queue_publication(payload.feed_url)


class ArticleManual(BaseModel):
    url: str
    title: str | None = None


@app.post("/articles")
async def add_article(payload: ArticleManual) -> dict[str, Any]:
    """Ingest one article now, ahead of any queued backfill. Its host must be allowlisted."""
    return await operations.queue_article(payload.url, payload.title)


@app.get("/jobs/{event_id}")
async def job_status(event_id: str) -> dict[str, Any]:
    return await operations.job_status(event_id)


@app.get("/sources/{claim_or_passage_id}")
async def source(claim_or_passage_id: str) -> dict[str, Any]:
    """Where a cited claim or passage comes from: article, exact text, and whether it is current."""
    return await operations.get_source(claim_or_passage_id)


@app.get("/search")
async def search(
    q: str = Query(min_length=1, max_length=2000),
    mode: Literal["hybrid", "sparse", "dense"] = "hybrid",
    limit: int = Query(default=5, ge=1, le=25),
) -> dict[str, Any]:
    """Passages for a query, no LLM. Compare `hybrid`, `sparse` and `dense` on one query."""
    return {"mode": mode, "results": await operations.search(q, mode, limit)}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int | None = Field(default=None, ge=1, le=25)


@app.post("/ask")
async def ask(payload: AskRequest) -> dict[str, Any]:
    """Answer a question with inline citations. A read: it never creates an Inngest run."""
    return await operations.ask(payload.question, payload.limit)


@app.post("/evals/run")
async def run_evals() -> dict[str, Any]:
    """Queue the retrieval benchmark. Poll GET /jobs/{event_id}, then read GET /evals/latest."""
    return await operations.queue_eval_suite()


@app.get("/evals/latest")
async def latest_evals(limit: int = Query(default=10, ge=1, le=50)) -> dict[str, Any]:
    """Pass rates, failure modes, recent failures with reasons, and the last variant comparison."""
    return await operations.eval_report(limit)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # At most this many rounds of searching, and this many dollars for planning and
    # reflection. The report and its scoring are added on top (about $0.15).
    max_iterations: int | None = Field(default=None, ge=1, le=5)
    max_usd: float | None = Field(default=None, gt=0, le=5)


@app.post("/research")
async def start_research(payload: ResearchRequest) -> dict[str, Any]:
    """Investigate a question over several searches. Returns at once; read the
    report with GET /research/{research_id}.
    """
    return await operations.queue_research(
        payload.question, payload.max_iterations, payload.max_usd
    )


@app.get("/research/{research_id}")
async def research_result(research_id: str) -> dict[str, Any]:
    """Status of a research run, and once it has finished the report, its
    citations, each round of searching, why it stopped, its cost and its scores.
    """
    return await operations.get_research(research_id)


class DigestRequest(BaseModel):
    # How many complete days before today the digest covers (default 7).
    lookback_days: int | None = Field(default=None, ge=1, le=90)


@app.post("/digests/run")
async def run_digest(payload: DigestRequest | None = None) -> dict[str, Any]:
    """Write the digest now instead of waiting for Monday 08:00 UTC. Poll
    GET /jobs/{event_id}, then read GET /digests/latest.
    """
    return await operations.queue_digest(payload.lookback_days if payload else None)


@app.get("/digests/latest")
async def latest_digest() -> dict[str, Any]:
    """The most recent digest: a cited briefing of what the window's articles claim."""
    return await operations.latest_digest()
