"""A job's progress is counted from what exists: stored articles and recorded failures."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import Article
from commonplace.db.queries import create_job, get_job_status, record_failure, upsert_article

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
