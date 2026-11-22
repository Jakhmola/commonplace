"""Unit tests for the pure building blocks of ingestion.

No Docker, database, Inngest or API key needed:

    url -> canonical id -> allowlist -> feed entry -> page text -> chunks -> packed vectors
"""

import hashlib
from array import array
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import inngest
import pytest
import respx

from commonplace import inngest_client
from commonplace.inngest_client import usage_from_output
from commonplace.main import job_state, run_summary
from commonplace.schemas.ids import canonical_id, normalize_url
from commonplace.sources.allowlist import PublicationEntry, assert_allowed
from commonplace.sources.chunking import chunk_text
from commonplace.sources.embedder import pack, unpack
from commonplace.sources.fetcher import FetchError, fetch_article
from commonplace.sources.parser import parse_html
from commonplace.sources.rss import parse_feed

# --- ids: every link variant of a post maps to one id ---


def test_tracking_params_and_trailing_slash_are_removed() -> None:
    messy = "http://WWW.Latent.Space/p/agent-labs/?utm_source=x&r=1#comments"
    assert normalize_url(messy) == "https://www.latent.space/p/agent-labs"


def test_canonical_id_is_publication_plus_post_slug() -> None:
    url = "https://www.latent.space/p/agent-labs?ref=share"
    assert canonical_id("latent-space", url) == "pub:latent-space:agent-labs"


def test_canonical_id_needs_a_path() -> None:
    with pytest.raises(ValueError, match="no path segment"):
        canonical_id("latent-space", "https://www.latent.space/")


# --- allowlist: only hosts listed in publications.yaml are fetched ---

LATENT_SPACE = PublicationEntry(
    slug="latent-space",
    name="Latent Space",
    feed_url="https://www.latent.space/feed",
    homepage="https://www.latent.space",
    archive_urls=["https://latent.space"],
)


def test_allowed_url_passes() -> None:
    assert assert_allowed("https://www.latent.space/p/agent-labs", [LATENT_SPACE]).slug == (
        "latent-space"
    )
    # the bare domain is listed as an archive host, so it passes too
    assert assert_allowed("https://latent.space/p/agent-labs", [LATENT_SPACE]) is LATENT_SPACE


@pytest.mark.parametrize(
    "url",
    [
        "https://random-blog.example.com/post",
        "https://evil-latent.space.example.com/p/agent-labs",  # lookalike host
        "https://sub.latent.space/p/agent-labs",  # unlisted subdomain
        "not a url",
    ],
)
def test_unknown_site_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="outside the allowlist"):
        assert_allowed(url, [LATENT_SPACE])


# --- chunking: long text becomes overlapping pieces, words stay whole ---

TEXT = "the quick brown fox jumps over the lazy dog while the cat watches"


def test_short_text_stays_in_one_chunk() -> None:
    assert chunk_text(TEXT, chunk_size=1000, chunk_overlap=10) == [TEXT]
    assert chunk_text("  \n ", chunk_size=1000, chunk_overlap=10) == []


def test_long_text_is_split_and_no_chunk_is_too_big() -> None:
    chunks = chunk_text(TEXT, chunk_size=30, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert chunks[0].startswith("the quick")
    assert chunks[-1].endswith("cat watches")


def test_neighbouring_chunks_share_some_words() -> None:
    chunks = chunk_text(TEXT, chunk_size=30, chunk_overlap=10)
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert set(first.split()) & set(second.split())


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_text(TEXT, chunk_size=10, chunk_overlap=10)


# --- feed: a list of pointers to articles ---

FEED = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
  <title>Demo</title>
  <item>
    <title>Agent labs</title>
    <link>https://www.latent.space/p/agent-labs?utm_source=rss&amp;utm_medium=feed#top</link>
    <dc:creator>Jane Writer</dc:creator>
    <pubDate>Tue, 17 Nov 2026 09:48:40 GMT</pubDate>
  </item>
  <item>
    <title>No date, so not usable</title>
    <link>https://www.latent.space/p/undated</link>
  </item>
  <item>
    <title>No link either</title>
    <pubDate>Mon, 16 Nov 2026 09:00:00 GMT</pubDate>
  </item>
</channel></rss>"""


def test_feed_entries_are_parsed() -> None:
    (entry,) = parse_feed(FEED)
    assert entry.title == "Agent labs"
    assert entry.published_at == datetime(2026, 11, 17, 9, 48, 40, tzinfo=UTC)


def test_feed_entry_keeps_author_and_clean_url() -> None:
    (entry,) = parse_feed(FEED)
    assert entry.author == "Jane Writer"
    assert entry.url == "https://www.latent.space/p/agent-labs"


def test_feed_skips_broken_entries_and_rejects_garbage() -> None:
    assert len(parse_feed(FEED)) == 1  # the two incomplete items are skipped, not fatal
    with pytest.raises(ValueError, match="did not parse"):
        parse_feed(b"<html><body>502 Bad Gateway</body></html")


# --- page: article text without the page chrome ---

PAGE = """<html><head><title>Why durable steps matter | Latent Space</title>
<meta name="author" content="Jane Writer"><meta property="article:published_time"
content="2026-11-17T09:48:40+00:00"></head><body>
<nav><a href="/">Home</a> <a href="/archive">Archive</a> <button>Subscribe now</button></nav>
<article><h1>Why durable steps matter</h1>
<p>A pipeline that fetches, parses, embeds and stores an article has four places to fail,
and the expensive one is the embedding call in the middle of it.</p>
<p>When a step fails, a durable engine replays the function from the top, but every step
that already finished returns its recorded result instead of running again.</p>
<p>That is why the page is fetched exactly once even if the embedding provider is down
for an hour: the retry starts at the step that failed, not at the beginning.</p>
</article><footer>Share this post. Subscribe now. Copyright 2026.</footer></body></html>"""


def test_parse_html_drops_page_chrome() -> None:
    fallback = datetime(2020, 1, 1, tzinfo=UTC)
    article = parse_html(PAGE, "https://www.latent.space/p/durable", fallback_published_at=fallback)
    assert "returns its recorded result" in article.text
    assert "Subscribe now" not in article.text
    assert article.published_at == datetime(2026, 11, 17, tzinfo=UTC)  # the page's date wins
    assert article.content_hash == hashlib.sha256(article.text.encode()).hexdigest()


def test_parse_html_without_text_is_an_error() -> None:
    with pytest.raises(ValueError, match="no article text"):
        parse_html(
            "<html><body></body></html>",
            "https://www.latent.space/p/x",
            fallback_published_at=datetime(2020, 1, 1, tzinfo=UTC),
        )


# --- vectors: packed small enough to cross an Inngest step boundary ---


def test_vector_pack_roundtrip_is_float32_exact_and_small() -> None:
    vector = [0.1 * i - 0.05 for i in range(1536)]
    assert unpack(pack(vector)) == array("f", vector).tolist()
    # 500 passages of a very long article must stay well under the 4 MB step output cap
    assert len(pack(vector)) < 8300


# --- cost capture: which step outputs count as a paid call ---


def test_only_complete_usage_dicts_are_recorded() -> None:
    usage = {"model": "m", "input_tokens": 10, "output_tokens": 0, "est_usd": 0.0001}
    assert usage_from_output({"vectors": [], "usage": usage}) == usage
    assert usage_from_output({"usage": {"model": "m"}}) is None  # incomplete
    assert usage_from_output({"html": "<p>usage</p>"}) is None
    assert usage_from_output(["usage"]) is None
    assert usage_from_output(None) is None


# --- job status: what Inngest says about the run ---


def test_a_run_between_retries_is_not_reported_as_failed() -> None:
    # The dev server labels a run "Failed" while it is only waiting to retry.
    retrying = {"status": "Failed", "ended_at": None, "output": None}
    assert run_summary([retrying]) == ("Running", [])


def test_a_failed_run_reports_its_message_without_the_stack_trace() -> None:
    dead = {
        "status": "Failed",
        "ended_at": "2026-11-22T20:14:45Z",
        "output": {"message": "Feed is outside the allowlist", "stack": "Traceback /home/me/..."},
    }
    assert run_summary([dead]) == ("Failed", ["Feed is outside the allowlist"])
    no_output = {"status": "Failed", "ended_at": "2026-11-22T20:14:45Z", "output": None}
    assert run_summary([no_output]) == ("Failed", ["unknown error"])
    assert run_summary([]) == (None, [])


async def test_a_failed_cost_write_does_not_fail_the_step(monkeypatch: pytest.MonkeyPatch) -> None:
    # If recording the cost raised, Inngest would discard the step's output and
    # run the paid call again.
    @asynccontextmanager
    async def database_is_down() -> AsyncIterator[None]:
        raise ConnectionError("database is down")
        yield

    monkeypatch.setattr(inngest_client, "get_session", database_is_down)
    usage = {"model": "m", "input_tokens": 10, "output_tokens": 0, "est_usd": 0.0001}
    step: Any = SimpleNamespace(id="embed")  # the middleware only checks that a step ran
    result = inngest.TransformOutputResult(error=None, output={"usage": usage}, step=step)
    middleware = inngest_client.CostCaptureMiddleware(inngest_client.client, raw_request=None)
    await middleware.transform_output(result)  # must not raise
    assert result.output == {"usage": usage}


# --- fetching: stay on the allowlist, treat a throttled robots.txt as temporary ---


@respx.mock
async def test_a_redirect_off_the_allowlist_is_never_followed() -> None:
    respx.get("https://www.latent.space/robots.txt").respond(404)
    respx.get("https://www.latent.space/p/moved").respond(
        302, headers={"location": "https://elsewhere.example/landing"}
    )
    elsewhere = respx.get("https://elsewhere.example/landing").respond(200, text="<p>hi</p>")

    def only_latent_space(url: str) -> None:
        assert_allowed(url, [LATENT_SPACE])

    with pytest.raises(ValueError, match="outside the allowlist"):
        await fetch_article("https://www.latent.space/p/moved", check_url=only_latent_space)
    assert not elsewhere.called


@respx.mock
async def test_a_throttled_robots_txt_is_retried_not_cached_as_allow_all() -> None:
    robots = respx.get("https://throttled.example/robots.txt")
    robots.respond(429)
    page = respx.get("https://throttled.example/p/a").respond(200, text="<p>hi</p>")
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_article("https://throttled.example/p/a")
    assert not page.called
    robots.respond(200, text="User-agent: *\nDisallow: /p/")
    with pytest.raises(FetchError, match="robots.txt disallows"):
        await fetch_article("https://throttled.example/p/a")


# --- dates, chunks and job states found in review ---


def test_the_feed_timestamp_is_kept_when_the_page_only_knows_the_day() -> None:
    feed_time = datetime(2026, 11, 17, 9, 48, 40, tzinfo=UTC)
    article = parse_html(PAGE, "https://www.latent.space/p/x", fallback_published_at=feed_time)
    assert article.published_at == feed_time


def test_a_very_long_token_does_not_produce_redundant_chunks() -> None:
    prose = " ".join(f"word{i:02d}" for i in range(60))
    text = f"{prose} {'x' * 900} tail"
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=150)
    for previous, chunk in zip(chunks, chunks[1:], strict=False):
        assert chunk not in previous  # every chunk adds text the one before lacked
    assert "x" * 900 in "".join(chunks) and chunks[-1].endswith("tail")


def test_job_state_is_lowercase_and_cancelled_counts_as_failed() -> None:
    assert job_state(None, "Running", []) == "running"
    assert job_state(None, "Cancelled", []) == "failed"
    assert job_state(None, None, []) == "queued"  # the event exists, its run has not started
    assert job_state({"status": "running"}, "Completed", []) == "running"
    assert job_state({"status": "completed"}, "Completed", []) == "completed"
    assert job_state({"status": "running"}, "Failed", ["boom"]) == "failed"
