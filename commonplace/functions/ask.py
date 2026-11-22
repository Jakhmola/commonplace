"""Answer a question from the knowledge base, with inline citations.

This is a read, so it never becomes an Inngest run. A workflow engine is for
expensive work that must not be lost; here someone is waiting, and a retry in
three minutes is no use to them. `ask_question` queries Postgres and Claude
directly, stores a snapshot of everything it did, and only then announces the
answer with a `cp/answer.produced` event that the caller does not wait for.
"""

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.config import settings
from commonplace.db.queries import save_snapshot
from commonplace.inngest_client import client
from commonplace.retrieval.hybrid import hybrid_search
from commonplace.schemas.events import ANSWER_PRODUCED, AnswerProduced, event
from commonplace.schemas.ids import snapshot_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You answer questions using passages retrieved from a personal library of AI engineering \
newsletters.

- Use only what the numbered passages say. Do not add facts from memory.
- Cite every claim inline with the number of the passage it comes from, like [1] or [2][4].
- If the passages do not cover the question, or only partly, say so plainly instead of guessing.
- Be direct and brief: a few sentences, or a short list when several things are asked for."""

NO_PASSAGES = "No relevant passages were found in the knowledge base for this question."

_llm: anthropic.AsyncAnthropic | None = None
# asyncio keeps only weak references to tasks, so a fire-and-forget task that
# nothing holds can be garbage collected before it finishes.
_background_tasks: set[asyncio.Task[None]] = set()


def get_llm() -> anthropic.AsyncAnthropic | None:
    """The Anthropic client, or None when no key is configured."""
    global _llm
    if not settings.anthropic_api_key:
        return None
    if _llm is None:
        _llm = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _llm


def format_context(passages: list[dict[str, Any]]) -> str:
    """Passages as numbered blocks; the numbers are what the answer cites."""
    return "\n\n".join(
        f"[{n}] (Title: {p['title']} | Author: {p.get('author') or 'Unknown'} | URL: {p['url']})\n"
        f"{p['text']}"
        for n, p in enumerate(passages, start=1)
    )


def _citation(n: int, passage: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_idx": n,
        "passage_id": passage["passage_id"],
        "title": passage["title"],
        "author": passage.get("author"),
        "url": passage["url"],
    }


def cited(answer_text: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Citations for the passages whose number actually appears in the answer."""
    return [_citation(n, p) for n, p in enumerate(passages, start=1) if f"[{n}]" in answer_text]


async def synthesize(
    llm: anthropic.AsyncAnthropic, question: str, passages: list[dict[str, Any]]
) -> str:
    """One Claude call. A provider failure becomes the answer text, never an HTTP error:
    the passages were still retrieved and the snapshot is still worth keeping.
    """
    prompt = (
        f"Passages:\n\n{format_context(passages)}\n\n"
        f"Question: {question}\n\n"
        "Answer the question from the passages, citing them inline."
    )
    try:
        response = await llm.beta.messages.create(
            model=settings.synthesis_model,
            max_tokens=settings.synthesis_max_tokens,
            # Someone is waiting on this answer; grounded Q&A needs little deliberation.
            output_config={"effort": settings.synthesis_effort},
            # If the model declines, the API re-runs the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        logger.exception("LLM synthesis failed")
        return f"LLM synthesis failed ({type(exc).__name__}): {exc}"

    if response.stop_reason == "refusal":
        return "The model declined to answer this question."
    return "\n\n".join(block.text for block in response.content if block.type == "text")


def _newest(passages: list[dict[str, Any]]) -> datetime | None:
    dates = []
    for p in passages:
        with contextlib.suppress(KeyError, TypeError, ValueError):
            dates.append(datetime.fromisoformat(p["published_at"]))
    return max(dates, default=None)


async def _announce(sid: str, retriever_variant: str) -> None:
    try:
        payload = AnswerProduced(snapshot_id=sid, retriever_variant=retriever_variant)
        await client.send(event(ANSWER_PRODUCED, payload))
    except Exception:
        # Telemetry must not fail the request, and must not fail silently either.
        # The snapshot is already committed, so nothing is lost.
        logger.exception("Could not send %s for %s", ANSWER_PRODUCED, sid)


async def ask_question(
    session: AsyncSession, question: str, limit: int | None = None
) -> dict[str, Any]:
    started = time.perf_counter()
    retriever_variant = f"hybrid_rrf_k{settings.rrf_k}"

    # The top passages go into the prompt; every candidate goes into the snapshot.
    top, candidates = await hybrid_search(
        session, question, limit=limit or settings.retrieval_limit
    )

    llm = get_llm()
    if not top:
        answer, citations = NO_PASSAGES, []
    elif llm is None:
        answer = (
            f"Retrieved {len(top)} relevant passages with hybrid search. "
            "Set ANTHROPIC_API_KEY to get an answer with inline citations."
        )
        citations = [_citation(n, p) for n, p in enumerate(top, start=1)]
    else:
        answer = await synthesize(llm, question, top)
        citations = cited(answer, top)

    # How fresh the answer is: the newest source it cites, else the newest retrieved.
    cited_ids = {c["passage_id"] for c in citations}
    as_of = (
        _newest([p for p in top if p["passage_id"] in cited_ids])
        or _newest(top)
        or datetime.now(UTC)
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    # The snapshot is committed before anything is sent anywhere.
    sid = snapshot_id()
    await save_snapshot(
        session,
        snapshot_id=sid,
        question=question,
        answer_text=answer,
        citations=citations,
        retrieved=candidates,
        retriever_variant=retriever_variant,
        as_of=as_of,
        latency_ms=latency_ms,
    )
    await session.commit()

    task = asyncio.create_task(_announce(sid, retriever_variant))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "answer": answer,
        "citations": citations,
        "as_of": as_of.isoformat(),
        "snapshot_id": sid,
        "latency_ms": latency_ms,
    }
