"""Unit tests for freshness polling, change detection, failure capture and the MCP tools.

No Docker or keys needed. Tests that need Postgres live in tests/integration.
"""

import inngest
import pytest
import respx

from commonplace.functions.freshness import poll_feeds, publications_to_poll
from commonplace.functions.ingest import classify_change
from commonplace.functions.ops import failure_from_event, handle_function_failed
from commonplace.schemas.events import ARTICLE_UPDATED, ArticleUpdated
from commonplace.sources.allowlist import PublicationEntry
from commonplace.sources.rss import fetch_feed_conditional

FEED_URL = "https://www.latent.space/feed"
FEED_BODY = b"<rss><channel><title>Latent Space</title></channel></rss>"

PUBLICATIONS = [
    PublicationEntry(
        slug="latent-space",
        name="Latent Space",
        feed_url=FEED_URL,
        homepage="https://www.latent.space",
    ),
    PublicationEntry(
        slug="interconnects",
        name="Interconnects",
        feed_url="https://www.interconnects.ai/feed",
        homepage="https://www.interconnects.ai",
    ),
]

# --- conditional requests: an unchanged feed costs one tiny round trip ---


@respx.mock
async def test_an_unchanged_feed_answers_304_and_carries_no_body() -> None:
    route = respx.get(FEED_URL).respond(304)
    response = await fetch_feed_conditional(
        FEED_URL, etag='"abc"', last_modified="Tue, 17 Nov 2026 09:00:00 GMT"
    )
    sent = route.calls.last.request.headers
    assert sent["if-none-match"] == '"abc"'
    assert sent["if-modified-since"] == "Tue, 17 Nov 2026 09:00:00 GMT"
    assert (response.status, response.content) == (304, None)
    # the validators we already hold stay valid
    assert (response.etag, response.last_modified) == ('"abc"', "Tue, 17 Nov 2026 09:00:00 GMT")


@respx.mock
async def test_a_changed_feed_returns_its_new_validators() -> None:
    respx.get(FEED_URL).respond(
        200,
        content=FEED_BODY,
        headers={"etag": '"def"', "last-modified": "Wed, 18 Nov 2026 09:00:00 GMT"},
    )
    response = await fetch_feed_conditional(FEED_URL, etag='"abc"')
    assert (response.status, response.content) == (200, FEED_BODY)
    assert (response.etag, response.last_modified) == ('"def"', "Wed, 18 Nov 2026 09:00:00 GMT")


@respx.mock
async def test_a_feed_without_validators_is_fetched_in_full() -> None:
    route = respx.get(FEED_URL).respond(200, content=FEED_BODY)
    response = await fetch_feed_conditional(FEED_URL)
    sent = route.calls.last.request.headers
    assert "if-none-match" not in sent and "if-modified-since" not in sent
    assert (response.status, response.etag, response.last_modified) == (200, None, None)


@respx.mock
async def test_a_failing_feed_raises_so_the_step_retries() -> None:
    respx.get(FEED_URL).respond(503)
    with pytest.raises(Exception, match="503"):
        await fetch_feed_conditional(FEED_URL)


# --- the poller: on a schedule, and on demand ---


def test_poll_runs_on_a_schedule_and_on_demand() -> None:
    triggers = poll_feeds.get_config("http://localhost:8000").main.triggers
    assert {getattr(t, "cron", None) for t in triggers} == {"*/5 * * * *", None}
    assert {getattr(t, "event", None) for t in triggers} == {"cp/feed.poll", None}


def test_poll_can_be_limited_to_one_publication() -> None:
    assert publications_to_poll(PUBLICATIONS, None) == PUBLICATIONS
    assert [p.slug for p in publications_to_poll(PUBLICATIONS, "interconnects")] == [
        "interconnects"
    ]
    with pytest.raises(inngest.NonRetriableError, match="not in publications.yaml"):
        publications_to_poll(PUBLICATIONS, "no-such-feed")


# --- edits: decided from the content hash, announced as an update ---


def test_change_is_decided_from_the_content_hash() -> None:
    assert classify_change(None, "abc") == "new"
    assert classify_change("abc", "abc") == "unchanged"  # stops before the paid call
    assert classify_change("abc", "def") == "changed"


def test_article_updated_carries_the_new_hash_and_count() -> None:
    updated = ArticleUpdated(
        canonical_id="pub:latent-space:a-post",
        article_id="6f1c0a3e-9d5b-4c55-8f0e-2b7a4d1c9e10",
        content_hash="def",
        passage_count=7,
    )
    assert ARTICLE_UPDATED == "cp/article.updated"
    assert updated.model_dump(mode="json")["passage_count"] == 7


# --- failed runs: recorded from Inngest's own failure event ---

FAILED_EVENT = {
    "function_id": "commonplace-ingest-article",
    "run_id": "01M82RUN",
    "error": {
        "name": "NonRetriableError",
        "message": "URL is outside the allowlist (publications.yaml): 'https://example.com/p/x'",
        "stack": 'Traceback (most recent call last):\n  File "/home/me/project/ingest.py" ...',
    },
    "event": {
        "name": "cp/article.discovered",
        "data": {"canonical_id": "pub:latent-space:x", "url": "https://example.com/p/x"},
    },
}


def test_a_failure_event_becomes_a_row_without_the_stack_trace() -> None:
    failure = failure_from_event(FAILED_EVENT)
    assert failure == {
        "function_id": "commonplace-ingest-article",
        "run_id": "01M82RUN",
        "error": "URL is outside the allowlist (publications.yaml): 'https://example.com/p/x'",
        "event_data": FAILED_EVENT["event"],
    }
    # an error that is not a dict, and an event that is missing, still make a row
    bare = failure_from_event({"function_id": "f", "run_id": "r", "error": "boom"})
    assert bare == {"function_id": "f", "run_id": "r", "error": "boom", "event_data": None}


def test_the_failure_handler_ignores_its_own_failures() -> None:
    own = {**FAILED_EVENT, "function_id": "commonplace-handle-function-failed"}
    assert failure_from_event(own) is None  # or one failure would record itself forever


def test_the_failure_handler_listens_to_inngest_itself() -> None:
    (trigger,) = handle_function_failed.get_config("http://localhost:8000").main.triggers
    assert getattr(trigger, "event", None) == "inngest/function.failed"
