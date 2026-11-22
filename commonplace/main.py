"""HTTP API: a thin layer over `operations.py`, plus the Inngest function endpoint."""

from typing import Any, Literal

import inngest.fast_api
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from commonplace import operations
from commonplace.db.session import get_session
from commonplace.functions import FUNCTIONS
from commonplace.functions.ask import ask_question
from commonplace.health import check_health
from commonplace.inngest_client import client
from commonplace.schemas.events import PublicationAdded

app = FastAPI(title="commonplace")
inngest.fast_api.serve(app, client, FUNCTIONS)


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


@app.get("/search")
async def search(
    q: str = Query(min_length=1),
    mode: Literal["hybrid", "sparse", "dense"] = "hybrid",
    limit: int = Query(default=5, ge=1, le=25),
) -> dict[str, Any]:
    """Passages for a query, no LLM. Compare `hybrid`, `sparse` and `dense` on one query."""
    return {"mode": mode, "results": await operations.search(q, mode, limit)}


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=25)


@app.post("/ask")
async def ask(payload: AskRequest) -> dict[str, Any]:
    """Answer a question with inline citations. A read: it never creates an Inngest run."""
    async with get_session() as session:
        return await ask_question(session, payload.question, payload.limit)
