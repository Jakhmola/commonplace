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
import re
import time
from datetime import UTC, datetime
from typing import Any

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.agents.prompts import ASK_SYSTEM_PROMPT
from commonplace.agents.providers import call_llm, get_llm
from commonplace.config import settings
from commonplace.db.queries import insert_cost, save_snapshot
from commonplace.inngest_client import client
from commonplace.retrieval.hybrid import hybrid_search
from commonplace.schemas.events import ANSWER_PRODUCED, AnswerProduced, event
from commonplace.schemas.ids import snapshot_id

logger = logging.getLogger(__name__)

NO_PASSAGES = "No relevant passages were found in the knowledge base for this question."

# asyncio keeps only weak references to tasks, so a fire-and-forget task that
# nothing holds can be garbage collected before it finishes.
_background_tasks: set[asyncio.Task[None]] = set()


def format_context(passages: list[dict[str, Any]]) -> str:
    """Passages as numbered blocks; the numbers are what the answer cites.

    A passage the claim graph led to also shows that claim and what kind of
    statement it is, so the model can tell a measurement from an opinion.
    """
    blocks = []
    for n, p in enumerate(passages, start=1):
        author = p.get("author") or "Unknown"
        lines = [f"[{n}] (Title: {p['title']} | Author: {author} | URL: {p['url']})"]
        if p.get("claim_text"):
            lines.append(f"Attributed claim ({p.get('claim_type') or 'claim'}): {p['claim_text']}")
        lines.append(p["text"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _citation(n: int, passage: dict[str, Any]) -> dict[str, Any]:
    citation = {
        "citation_idx": n,
        "passage_id": passage["passage_id"],
        "title": passage["title"],
        "author": passage.get("author"),
        "url": passage["url"],
    }
    if passage.get("claim_id"):
        citation |= {"claim_id": passage["claim_id"], "claim_type": passage.get("claim_type")}
    return citation


_BRACKETS = re.compile(r"\[\s*(\d+(?:\s*[,\-\u2013]\s*\d+)*)\s*\]")


def _cited_numbers(answer_text: str) -> set[int]:
    """Passage numbers an answer refers to: [1], [1][2], [1, 2] and [1-3] all count."""
    numbers: set[int] = set()
    for group in _BRACKETS.findall(answer_text):
        for part in group.split(","):
            first, _, last = part.replace("\u2013", "-").partition("-")
            low, high = int(first), int(last or first)
            if high - low < 50:  # a range, not two unrelated numbers far apart
                numbers.update(range(low, high + 1))
    return numbers


def cited(answer_text: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Citations for the passages the answer actually refers to."""
    numbers = _cited_numbers(answer_text)
    return [_citation(n, p) for n, p in enumerate(passages, start=1) if n in numbers]


async def synthesize(
    llm: anthropic.AsyncAnthropic, question: str, passages: list[dict[str, Any]]
) -> tuple[str, dict[str, Any] | None]:
    """(answer text, usage of the call or None if nothing was billed).

    A failure becomes the answer text, never an HTTP error: the passages were
    still retrieved and the snapshot is still worth keeping.
    """
    prompt = (
        f"Passages:\n\n{format_context(passages)}\n\n"
        f"Question: {question}\n\n"
        "Answer the question from the passages, citing them inline."
    )
    try:
        result = await call_llm(
            system_prompt=ASK_SYSTEM_PROMPT,
            user_input=prompt,
            model=settings.synthesis_model,
            max_tokens=settings.synthesis_max_tokens,
            # Someone is waiting on this answer; grounded Q&A needs little deliberation.
            effort=settings.synthesis_effort,
            llm=llm,
        )
    except Exception as exc:
        # Not only API errors: the SDK also raises plain exceptions before sending.
        logger.exception("LLM synthesis failed")
        return f"LLM synthesis failed ({type(exc).__name__}): {exc}", None

    if result["stop_reason"] == "refusal":
        return "The model declined to answer this question.", result["usage"]
    if not result["output"]:
        # e.g. max_tokens spent before any answer was written
        answer = f"The model returned no answer (stop reason: {result['stop_reason']})."
        return answer, result["usage"]
    return result["output"], result["usage"]


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
    retriever_variant = f"hybrid_3way_rrf_k{settings.rrf_k}"

    # The top passages go into the prompt; every candidate goes into the snapshot.
    top, candidates = await hybrid_search(
        session, question, limit=limit or settings.retrieval_limit
    )
    await session.commit()  # hand the connection back while the model writes its answer

    llm = get_llm()
    usage: dict[str, Any] | None = None
    if not top:
        answer, citations = NO_PASSAGES, []
    elif llm is None:
        answer = (
            f"Retrieved {len(top)} relevant passages with hybrid search. "
            "Set ANTHROPIC_API_KEY to get an answer with inline citations."
        )
        citations = [_citation(n, p) for n, p in enumerate(top, start=1)]
    else:
        answer, usage = await synthesize(llm, question, top)
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
    if usage is not None:
        # Not inside an Inngest step, so the cost middleware never sees this call.
        await insert_cost(
            session,
            model=usage["model"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            est_usd=usage["est_usd"],
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
