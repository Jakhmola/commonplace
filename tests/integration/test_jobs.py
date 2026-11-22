"""A job's progress is counted from what exists: stored articles and recorded failures."""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import Article, Failure
from commonplace.db.queries import (
    create_job,
    get_job_status,
    get_unseen_canonical_ids,
    record_failure,
    upsert_article,
)

DONE = "pub:integration-test:a-seeded-post"  # the article the fixture stores
MISSING = "pub:integration-test:never-arrived"


async def test_a_failed_article_is_counted_against_its_job(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    await create_job(db, event_id="evt-job-test", kind="backfill", canonical_ids=[DONE, MISSING])
    still_running = await get_job_status(db, "evt-job-test")
    assert still_running is not None
    assert (still_running["done"], still_running["failed"]) == (1, 0)
    assert still_running["status"] == "running"

    await record_failure(
        db,
        function_id="commonplace-ingest-article",
        run_id="run-1",
        error="robots.txt disallows fetching the page",
        event_data={"name": "cp/article.discovered", "data": {"canonical_id": MISSING}},
    )
    settled = await get_job_status(db, "evt-job-test")
    assert settled is not None
    assert (settled["done"], settled["failed"], settled["status"]) == (1, 1, "failed")
    assert settled["errors"] == ["robots.txt disallows fetching the page"]

    # The article is retried and stored: its old failure no longer counts.
    stored = await db.get(Article, article_id)
    assert stored is not None
    await upsert_article(
        db,
        canonical_id=MISSING,
        publication_id=stored.publication_id,
        url="https://integration-test.example/p/never-arrived",
        title="Arrived after all",
        author=None,
        published_at=stored.published_at,
        content_hash="h",
        text="t",
    )
    recovered = await get_job_status(db, "evt-job-test")
    assert recovered is not None
    assert (recovered["done"], recovered["failed"], recovered["status"]) == (2, 0, "completed")


async def test_a_failure_from_before_the_job_does_not_count_against_it(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    # The article failed during an earlier job; a new job is retrying it right now.
    failure_id = await record_failure(
        db,
        function_id="commonplace-ingest-article",
        run_id="run-old",
        error="an old failure",
        event_data={"name": "cp/article.discovered", "data": {"canonical_id": MISSING}},
    )
    await db.execute(
        update(Failure)
        .where(Failure.id == failure_id)
        .values(created_at=datetime.now(UTC) - timedelta(days=1))
    )
    await create_job(db, event_id="evt-retry", kind="backfill", canonical_ids=[DONE, MISSING])

    status = await get_job_status(db, "evt-retry")
    assert status is not None
    assert (status["done"], status["failed"], status["status"]) == (1, 0, "running")


async def test_polling_does_not_requeue_articles_that_failed_for_good(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    await record_failure(
        db,
        function_id="commonplace-ingest-article",
        run_id="run-1",
        error="no article text found",
        event_data={"name": "cp/article.discovered", "data": {"canonical_id": MISSING}},
    )
    brand_new = "pub:integration-test:brand-new"
    ids = [DONE, MISSING, brand_new]
    assert await get_unseen_canonical_ids(db, ids, skip_failed=True) == [brand_new]
    # a backfill is the explicit way to retry, so it still sees the failed one
    assert await get_unseen_canonical_ids(db, ids, skip_failed=False) == [MISSING, brand_new]
