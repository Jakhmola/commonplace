"""Unit tests for flow control, hybrid retrieval and answers. No Docker or keys needed."""

import json
import uuid
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from commonplace.functions import FUNCTIONS
from commonplace.functions.ask import cited, format_context, synthesize
from commonplace.functions.backfill import candidates, feed_pointers
from commonplace.functions.ingest import ingest_article
from commonplace.main import resolve_manual_article
from commonplace.retrieval.hybrid import compute_rrf
from commonplace.schemas.ids import canonical_id, snapshot_id
from commonplace.sources.allowlist import PublicationEntry, assert_allowed

# --- flow control is configuration on the function, enforced by Inngest's queue ---


def _config() -> Any:
    return ingest_article.get_config("http://localhost:8000").main


def test_concurrency_is_keyed_per_publication() -> None:
    (concurrency,) = _config().concurrency
    assert concurrency.key == "event.data.publication_id"
    assert concurrency.limit == 3


def test_runs_are_throttled_globally() -> None:
    throttle = _config().throttle
    assert throttle is not None
    assert throttle.key is None  # one embedding quota, shared by every publication
    assert throttle.limit == 50


def test_manual_runs_are_shifted_ahead_and_backfill_runs_behind() -> None:
    priority = _config().priority
    assert priority is not None
    assert priority.run == "event.data.via == 'manual' ? 600 : -600"


def test_no_idempotency_key_so_an_article_can_be_resent() -> None:
    assert _config().idempotency is None


def test_backfill_listens_for_added_publications() -> None:
    ids = {f.get_config("http://localhost:8000").main.id for f in FUNCTIONS}
    assert ids == {"commonplace-backfill-publication", "commonplace-ingest-article"}


# --- hybrid retrieval: fuse two rankings by position, never by score ---


def _hit(passage_id: str, rank: float = 0.5) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": f"text of {passage_id}",
        "title": "A post",
        "author": "Jane Writer",
        "url": "https://www.latent.space/p/a-post",
        "published_at": "2026-11-17T00:00:00+00:00",
        "rank": rank,
    }


def test_agreement_between_engines_beats_one_engines_favourite() -> None:
    sparse = [_hit("only-sparse"), _hit("both")]
    dense = [_hit("only-dense"), _hit("both")]
    fused = compute_rrf(sparse, dense)
    # second in both lists beats first in just one
    assert [c.passage_id for c in fused] == ["both", "only-dense", "only-sparse"]
    assert fused[0].source == "both"


def test_rrf_score_is_the_sum_of_one_over_k_plus_rank() -> None:
    fused = compute_rrf([_hit("p")], [_hit("x"), _hit("p")], k=60)
    scores = {c.passage_id: c.rrf_score for c in fused}
    assert scores["p"] == pytest.approx(1 / 61 + 1 / 62)  # first in sparse, second in dense
    assert scores["x"] == pytest.approx(1 / 61)


def test_ties_are_broken_by_passage_id_so_results_are_deterministic() -> None:
    fused = compute_rrf([_hit("b")], [_hit("a")])  # both rank 1 in one engine: equal scores
    assert [c.passage_id for c in fused] == ["a", "b"]
    assert fused[0].rrf_score == fused[1].rrf_score


def test_a_passage_found_by_one_engine_keeps_its_source() -> None:
    # A paraphrased question often matches nothing in full-text search.
    fused = compute_rrf([], [_hit("d1", rank=0.61), _hit("d2", rank=0.58)])
    assert [c.source for c in fused] == ["vector", "vector"]
    assert fused[0].sparse_rank is None and fused[0].dense_rank == 1
    assert fused[0].dense_score == pytest.approx(0.61)
    (only_sparse,) = compute_rrf([_hit("s1", rank=0.3)], [])
    assert only_sparse.source == "bm25" and only_sparse.dense_rank is None


# --- one publication on two domains, one id per post ---

AHEAD_OF_AI = PublicationEntry(
    slug="ahead-of-ai",
    name="Ahead of AI",
    feed_url="https://magazine.sebastianraschka.com/feed",
    homepage="https://magazine.sebastianraschka.com",
    archive_urls=["https://sebastianraschka.substack.com"],
)


def test_both_domains_of_a_publication_are_allowed() -> None:
    for url in (
        "https://magazine.sebastianraschka.com/p/a-post",
        "https://sebastianraschka.substack.com/p/a-post",
    ):
        assert assert_allowed(url, [AHEAD_OF_AI]).slug == "ahead-of-ai"


def test_the_same_post_gets_the_same_id_on_either_domain() -> None:
    assert canonical_id(
        "ahead-of-ai", "https://magazine.sebastianraschka.com/p/a-post"
    ) == canonical_id("ahead-of-ai", "https://sebastianraschka.substack.com/p/a-post/?r=1")


def test_snapshot_ids_are_prefixed_and_unique() -> None:
    ids = {snapshot_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("snap_") and len(i) == 17 for i in ids)


# --- manual articles: validated before anything is queued ---


def test_manual_article_url_is_validated() -> None:
    pub, cid = resolve_manual_article(
        "https://magazine.sebastianraschka.com/p/a-post?utm_source=x", [AHEAD_OF_AI]
    )
    assert (pub.slug, cid) == ("ahead-of-ai", "pub:ahead-of-ai:a-post")
    with pytest.raises(ValueError, match="outside the allowlist"):
        resolve_manual_article("https://example.com/p/a-post", [AHEAD_OF_AI])
    with pytest.raises(ValueError, match="no path segment"):
        resolve_manual_article("https://magazine.sebastianraschka.com/", [AHEAD_OF_AI])


# --- answers: numbered context in, citations for what was actually cited out ---

PASSAGES = [
    {**_hit("p1"), "title": "First post", "text": "Durable steps are memoized."},
    {**_hit("p2"), "title": "Second post", "author": None, "text": "Throttling limits run starts."},
    {**_hit("p3"), "title": "Third post", "text": "Priority is a time shift."},
]


def test_context_numbers_passages_from_one() -> None:
    context = format_context(PASSAGES)
    assert context.startswith("[1] (Title: First post | Author: Jane Writer | URL: https://")
    assert "[2] (Title: Second post | Author: Unknown |" in context
    assert context.count("\n\n") == 2  # one blank line between passages, none inside
    assert context.endswith("Priority is a time shift.")


def test_only_passages_the_answer_cites_become_citations() -> None:
    answer = "Steps are memoized [1]. Priority shifts arrival time [3]. See also [7]."
    citations = cited(answer, PASSAGES)
    assert [(c["citation_idx"], c["passage_id"]) for c in citations] == [(1, "p1"), (3, "p3")]
    assert citations[0].keys() == {"citation_idx", "passage_id", "title", "author", "url"}
    assert cited("I could not find this in the passages.", PASSAGES) == []


class _FailingLLM:
    """Stands in for the Anthropic client when the provider is unreachable."""

    def __init__(self) -> None:
        async def create(**_: Any) -> Any:
            raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


async def test_llm_failure_becomes_the_answer_text() -> None:
    answer = await synthesize(_FailingLLM(), "What is throttling?", PASSAGES)  # type: ignore[arg-type]
    assert answer.startswith("LLM synthesis failed (APIConnectionError)")
    assert cited(answer, PASSAGES) == []  # a failed answer cites nothing


# --- backfill: what the feed step hands on, and which entries become articles ---


def test_the_feed_step_passes_on_pointers_not_post_bodies() -> None:
    body = "<p>" + "a very long post " * 60_000 + "</p>"  # about 1 MB of content
    feed = f"""<?xml version="1.0"?>
    <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
      <item><title>Long post</title><link>https://www.latent.space/p/long-post</link>
        <pubDate>Tue, 17 Nov 2026 09:48:40 GMT</pubDate>
        <content:encoded><![CDATA[{body}]]></content:encoded></item>
    </channel></rss>""".encode()
    (pointer,) = feed_pointers(feed)
    assert pointer["url"] == "https://www.latent.space/p/long-post"
    assert len(json.dumps(pointer)) < 500  # a step output is capped at 4 MB


def test_backfill_skips_entries_it_cannot_identify_and_duplicate_ids() -> None:
    def entry(url: str) -> dict[str, Any]:
        return {"title": "t", "url": url, "author": None, "published_at": "2026-11-17T00:00:00Z"}

    pointers = [
        entry("https://magazine.sebastianraschka.com/p/a-post"),
        entry("https://magazine.sebastianraschka.com/"),  # the homepage: no post slug
        entry("https://magazine.sebastianraschka.com/podcast/a-post"),  # same id as the first
        entry("https://magazine.sebastianraschka.com/p/b-post"),
    ]
    found = candidates(pointers, AHEAD_OF_AI, uuid.uuid4())
    assert [c.canonical_id for c in found] == ["pub:ahead-of-ai:a-post", "pub:ahead-of-ai:b-post"]
    assert found[0].url.endswith("/p/a-post") and found[0].via == "backfill"
