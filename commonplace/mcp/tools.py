"""The MCP tools. Each is a thin call into `operations.py`.

Reads (`cp_health`, `cp_search`, `cp_ask`, `cp_sources`, `cp_eval_report`,
`cp_research_result`, `cp_weekly_digest`, `cp_job_status`, `cp_list_publications`)
answer directly from Postgres and the graph. Tools that start work (`cp_research`,
`cp_add_publication`, `cp_backfill`, `cp_sync`) send an event and return a
handle at once: an agent should never sit waiting
on a long tool call, so it gets something to poll instead.

The docstrings are the tool descriptions an agent reads to decide what to call.
"""

import functools
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from commonplace import operations
from commonplace.health import check_health

type Tool = Callable[..., Awaitable[dict[str, Any]]]


def _tool(fn: Tool) -> Tool:
    """Report a request that cannot be carried out as a tool error, not a traceback."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return await fn(*args, **kwargs)
        except operations.OperationError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


@_tool
async def cp_health() -> dict[str, Any]:
    """Check that Postgres, Inngest and Memgraph are reachable, and how big the corpus is.

    Returns per-dependency status (`ok` or `error: ...`) with article, passage,
    claim and topic counts. Call this first when another tool fails unexpectedly: it says which
    dependency is down.
    """
    return await check_health()


@_tool
async def cp_search(
    query: Annotated[str, Field(description="Free-text query, e.g. 'evaluating coding agents'")],
    limit: Annotated[int, Field(ge=1, le=25, description="How many passages to return")] = 5,
) -> dict[str, Any]:
    """Search the ingested newsletters and return the best matching passages.

    Hybrid search: keyword, semantic and claim-graph results fused by rank. Each
    result has the passage text, a short `snippet`, the article's title, author,
    URL and date, which engines found it, and the claim that led to it if the
    graph did. Returns an empty list when nothing matches.
    Use this to find sources; use cp_ask for a written answer.
    """
    results = await operations.search(query, "hybrid", limit)
    return {"results": [{**r, "snippet": r["text"][:300]} for r in results]}


@_tool
async def cp_ask(
    question: Annotated[str, Field(description="A question to answer from the knowledge base")],
) -> dict[str, Any]:
    """Answer a question from the ingested newsletters, with inline citations.

    The answer uses only retrieved passages and cites them as [1], [2]. Returns
    the answer, the cited sources with URLs, `as_of` (the date of the newest
    source cited) and a `snapshot_id` identifying the stored record of this answer.
    """
    return await operations.ask(question)


@_tool
async def cp_sources(
    claim_or_passage_id: Annotated[
        str,
        Field(
            description="A claim_id or passage_id from a citation returned by cp_ask or cp_search"
        ),
    ],
) -> dict[str, Any]:
    """Trace a cited claim or passage back to its source and check it is still current.

    Use cp_search to find passages for a query. Use this when you already hold
    an id from a citation and want to verify it: it returns the exact quoted
    text, the article, author and publication, for a claim its type (opinion,
    measurement or prediction) and topics, and its validity (`is_active`, and
    `valid_to` if the article has since been edited).
    """
    return await operations.get_source(claim_or_passage_id)


@_tool
async def cp_eval_report(
    limit: Annotated[int, Field(ge=1, le=50, description="How many recent failures to list")] = 10,
) -> dict[str, Any]:
    """Report how trustworthy the knowledge base's answers currently are.

    Returns pass rates of the automatic checks and judges over evaluated
    answers, a count per failure mode (FM-1 to FM-5), the most recent failing
    answers with the judge's reason, and the latest comparison of retrieval
    variants (recall@k and MRR for keyword, vector, 2-way and 3-way search).
    Use it before relying on cp_ask for something that matters.
    """
    return await operations.eval_report(limit)


@_tool
async def cp_research(
    question: Annotated[
        str,
        Field(description="A question that needs several searches, e.g. a comparison of authors"),
    ],
    max_iterations: Annotated[
        int | None, Field(ge=1, le=5, description="Rounds of searching at most (default 3)")
    ] = None,
) -> dict[str, Any]:
    """Start a research run: several planned searches, then a cited report.

    Use this instead of cp_ask when one search would not be enough: comparing
    what several authors say, following a topic across publications, or a
    question with distinct parts. It takes one to three minutes and costs more
    than cp_ask, and is bounded by a cap on search rounds and on dollars.
    Returns at once with a `research_id`; it does not wait for the report.
    Read the result with cp_research_result(research_id).
    """
    return await operations.queue_research(question, max_iterations)


@_tool
async def cp_research_result(
    research_id: Annotated[str, Field(description="The research_id returned by cp_research")],
) -> dict[str, Any]:
    """Read a research run started with cp_research.

    `status` is queued or running until it finishes; call again after a short
    wait. When it is completed (or budget_exceeded, which still has a report)
    the result has the `report` with inline citations, the cited sources, the
    searches of each round, `stop_reason` (why it stopped searching), the cost
    in dollars and a `confidence` from the automatic checks and judges.
    """
    return await operations.get_research(research_id)


@_tool
async def cp_weekly_digest() -> dict[str, Any]:
    """Read the latest weekly digest of the knowledge base.

    A short briefing of what the articles published in the last complete week
    claimed: what was measured, what authors argued, topics that several
    publications wrote about, and predictions. Every statement cites a claim by
    number; `citations` lists those claims with author, article and URL, and a
    `claim_id` that cp_sources can trace to the exact quote. `stats` has article
    and claim counts and whether the citations checked out. A new digest is
    written every Monday at 08:00 UTC.
    """
    return await operations.latest_digest()


@_tool
async def cp_list_publications() -> dict[str, Any]:
    """List the publications in the knowledge base with article counts and last poll time."""
    return {"publications": await operations.list_publications()}


@_tool
async def cp_job_status(
    event_id: Annotated[
        str, Field(description="The event_id returned by cp_add_publication or cp_backfill")
    ],
) -> dict[str, Any]:
    """Report progress of an ingestion job: status, articles done, failed and total."""
    return await operations.job_status(event_id)


@_tool
async def cp_add_publication(
    feed_url: Annotated[str, Field(description="The publication's RSS feed URL")],
) -> dict[str, Any]:
    """Start ingesting every article in a publication's feed.

    The feed must be listed in the project's publications.yaml allowlist; an
    unlisted feed is rejected. Returns at once with `event_id` and `trace_url`;
    it does not wait for ingestion. Poll cp_job_status(event_id) for progress.
    """
    return await operations.queue_publication(feed_url)


@_tool
async def cp_backfill(
    publication: Annotated[
        str, Field(description="A publication slug, its feed URL, or 'all'")
    ] = "all",
) -> dict[str, Any]:
    """Re-read one publication's feed, or all of them, and ingest whatever is missing.

    Articles already stored are skipped, so this is safe to repeat. Returns at
    once with one job per publication; poll cp_job_status with a job's event_id.
    """
    return await operations.queue_backfill(publication)


@_tool
async def cp_sync(
    publication: Annotated[
        str | None, Field(description="A publication slug, or omit to check every feed")
    ] = None,
) -> dict[str, Any]:
    """Check feeds for new articles right now instead of waiting for the 5-minute schedule.

    Unchanged feeds cost one small conditional request. Returns at once; new
    articles appear in cp_search once their ingestion finishes.
    """
    return await operations.trigger_poll(publication)
