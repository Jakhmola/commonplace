"""Research runs and digests in Postgres."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import Digest, ResearchRun
from commonplace.db.queries import (
    fail_research_run,
    finish_research_run,
    get_active_claims_for_articles,
    get_articles_in_window,
    get_latest_digest,
    get_research_run,
    persist_article_claims,
    start_research_run,
    upsert_digest,
)
from commonplace.schemas.ids import claim_id

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)


async def test_a_research_run_is_started_once_and_finished(db: AsyncSession) -> None:
    await start_research_run(db, research_id="res_itest_1", event_id="evt-1", question="First?")
    # the step ran again after a retry: the row is not reset
    await start_research_run(db, research_id="res_itest_1", event_id="evt-1", question="Other?")
    running = await get_research_run(db, "res_itest_1")
    assert running is not None
    assert (running["status"], running["question"], running["report"]) == (
        "running",
        "First?",
        None,
    )
    assert running["completed_at"] is None

    await finish_research_run(
        db,
        research_id="res_itest_1",
        status="completed",
        stop_reason="converged",
        plan={"subqueries": ["a"]},
        hops=[{"iteration": 1, "added": 4}],
        report_text="A report [1].",
        citations=[{"citation_idx": 1, "passage_id": "p1"}],
        retrieved=[{"passage_id": "p1", "in_prompt": True}],
        eval_summary={"deterministic_passed": True},
        confidence=1.0,
        iterations_used=1,
        total_cost_usd=0.2134,
        latency_ms=4200,
    )
    done = await get_research_run(db, "res_itest_1")
    assert done is not None
    assert (done["status"], done["stop_reason"], done["report"]) == (
        "completed",
        "converged",
        "A report [1].",
    )
    assert (done["confidence"], done["total_cost_usd"], done["iterations_used"]) == (1.0, 0.2134, 1)
    assert done["completed_at"] is not None
    assert "retrieved" not in done  # the pool is kept for audits, not sent to every reader

    count = (
        select(func.count())
        .select_from(ResearchRun)
        .where(ResearchRun.research_id == "res_itest_1")
    )
    assert (await db.execute(count)).scalar_one() == 1
    assert await get_research_run(db, "res_does_not_exist") is None


async def test_a_queued_run_becomes_running_and_a_finished_one_stays_finished(
    db: AsyncSession,
) -> None:
    async def status() -> str:
        run = await get_research_run(db, "res_itest_2")
        assert run is not None
        return str(run["status"])

    start = {"research_id": "res_itest_2", "event_id": "evt-2", "question": "Q?"}
    await start_research_run(db, **start, status="queued")  # the API, right after sending
    assert await status() == "queued"
    await start_research_run(db, **start, status="running")  # the function's first step
    assert await status() == "running"
    await start_research_run(db, **start, status="queued")  # the API lost the race: no step back
    assert await status() == "running"
    await finish_research_run(db, research_id="res_itest_2", status="completed")
    await start_research_run(db, **start, status="running")  # a replayed first step
    assert await status() == "completed"


async def test_a_failure_does_not_overwrite_a_run_that_finished(db: AsyncSession) -> None:
    start = {"research_id": "res_itest_3", "event_id": "evt-3", "question": "Q?"}
    await start_research_run(db, **start)
    await fail_research_run(db, research_id="res_itest_3", error="the provider was down")
    failed = await get_research_run(db, "res_itest_3")
    assert failed is not None
    assert (failed["status"], failed["eval_summary"]) == (
        "failed",
        {"error": "the provider was down"},
    )
    assert failed["completed_at"] is not None

    # found in review: the report was stored, then the last step failed for good
    start["research_id"] = "res_itest_4"
    await start_research_run(db, **start)
    await finish_research_run(
        db, research_id="res_itest_4", status="completed", report_text="A report [1]."
    )
    await fail_research_run(db, research_id="res_itest_4", error="could not send the event")
    kept = await get_research_run(db, "res_itest_4")
    assert kept is not None
    assert (kept["status"], kept["report"], kept["eval_summary"]) == (
        "completed",
        "A report [1].",
        {},
    )


def _digest(digest_id: uuid.UUID, start: datetime, text: str) -> dict[str, Any]:
    return {
        "digest_id": digest_id,
        "window_start": start,
        "window_end": start + timedelta(days=7),
        "title": "Week of test",
        "summary_markdown": text,
        "citations": [{"n": 1, "claim_id": "c1"}],
        "stats": {"article_count": 3, "claim_count": 9},
    }


async def test_a_window_has_one_digest(db: AsyncSession) -> None:
    digest_id = uuid.uuid4()
    far_future = datetime(2090, 1, 5, tzinfo=UTC)  # later than any real digest: it is "latest"
    await upsert_digest(db, **_digest(digest_id, far_future, "first draft [1]"))
    await upsert_digest(db, **_digest(digest_id, far_future, "second draft [1]"))

    count = select(func.count()).select_from(Digest).where(Digest.digest_id == digest_id)
    assert (await db.execute(count)).scalar_one() == 1
    latest = await get_latest_digest(db)
    assert latest is not None
    assert (latest["digest_id"], latest["summary_markdown"]) == (str(digest_id), "second draft [1]")
    assert latest["stats"] == {"article_count": 3, "claim_count": 9}
    assert latest["window_start"] == far_future.isoformat()


async def test_the_digest_window_falls_back_to_recent_articles(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    # the seeded article was published on 2026-11-17
    start = datetime(2026, 11, 17, tzinfo=UTC)
    in_window, fell_back = await get_articles_in_window(db, start, start + timedelta(days=1))
    assert not fell_back
    seeded = next(a for a in in_window if a["article_id"] == str(article_id))
    assert (seeded["publication"], seeded["title"]) == ("Integration Test", "A seeded post")
    end = (start + timedelta(days=1)).isoformat()
    assert all(start.isoformat() <= a["published_at"] < end for a in in_window)

    empty_start = datetime(2090, 1, 5, tzinfo=UTC)
    recent, fell_back = await get_articles_in_window(
        db, empty_start, empty_start + timedelta(days=7), fallback_limit=2
    )
    assert fell_back
    assert 1 <= len(recent) <= 2
    dates = [a["published_at"] for a in recent]
    assert dates == sorted(dates, reverse=True)  # newest first


def _claim(article_id: uuid.UUID, quote: str, confidence: float, kind: str) -> dict[str, Any]:
    text = f"The author says: {quote}"
    return {
        "claim_id": str(claim_id(article_id, "v1", quote, text)),
        "text": text,
        "quote_span": quote,
        "claim_type": kind,
        "confidence": confidence,
        "topics": [{"slug": "quokkazebra", "name": "Quokkazebra"}],
    }


async def test_window_claims_are_current_capped_per_article_and_carry_their_source(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    claims = [
        _claim(article_id, "quokkazebra is fine", 0.5, "opinion"),
        _claim(article_id, "quokkazebra took 3 ms", 0.9, "measurement"),
        _claim(article_id, "quokkazebra will win", 0.7, "prediction"),
    ]
    await persist_article_claims(db, article_id=article_id, claims=claims, now=NOW)

    best_two = await get_active_claims_for_articles(db, [article_id], per_article=2)
    assert [c["claim_type"] for c in best_two] == ["measurement", "prediction"]  # by confidence
    first = best_two[0]
    assert first["claim_text"] == "The author says: quokkazebra took 3 ms"
    assert (first["title"], first["publication"]) == ("A seeded post", "Integration Test")
    assert first["topics"] == ["quokkazebra"]

    # a claim from an older version of the article is not news
    await persist_article_claims(db, article_id=article_id, claims=claims[:1], now=NOW)
    assert len(await get_active_claims_for_articles(db, [article_id], per_article=5)) == 1
    assert await get_active_claims_for_articles(db, [], per_article=5) == []
