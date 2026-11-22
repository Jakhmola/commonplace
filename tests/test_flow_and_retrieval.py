"""Unit tests for flow control, hybrid retrieval and answers. No Docker or keys needed."""

import json
import uuid
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from commonplace import operations
from commonplace.citations import cited_numbers
from commonplace.functions import FUNCTIONS
from commonplace.functions.ask import cited, format_context, synthesize
from commonplace.functions.backfill import candidates, feed_pointers
from commonplace.functions.ingest import ingest_article
from commonplace.operations import resolve_manual_article
from commonplace.retrieval.hybrid import compute_rrf
from commonplace.schemas.ids import canonical_id, snapshot_id
from commonplace.sources.allowlist import PublicationEntry, assert_allowed

# --- flow control is configuration on the function, enforced by Inngest's queue ---


def _config() -> Any:
    return ingest_article.get_config("http://localhost:8000").main


def test_concurrency_is_keyed_per_publication() -> None:
    limits = {c.key: c.limit for c in _config().concurrency}
    assert limits["event.data.publication_id"] == 3


def test_runs_for_the_same_article_never_overlap() -> None:
    # Two events for one article (a poll and a manual add, or a feed posted twice):
    # the second is skipped while the first is in flight, instead of embedding again.
    singleton = _config().singleton
    assert (singleton.key, singleton.mode) == ("event.data.canonical_id", "skip")


def test_runs_are_throttled_globally() -> None:
    throttle = _config().throttle
    assert throttle is not None
    assert throttle.key is None  # one embedding quota, shared by every publication
    assert throttle.limit == 50


def test_manual_runs_are_shifted_ahead_and_other_runs_are_not_delayed() -> None:
    priority = _config().priority
    assert priority is not None
    # A negative value is not a lower rank: Inngest holds the run that many seconds
    # even on an idle queue, so a backfill would start ten minutes late.
    assert priority.run == "event.data.via == 'manual' ? 600 : 0"


def test_no_idempotency_key_so_an_article_can_be_resent() -> None:
    assert _config().idempotency is None


def test_backfill_listens_for_added_publications() -> None:
    ids = {f.get_config("http://localhost:8000").main.id for f in FUNCTIONS}
    assert {"commonplace-backfill-publication", "commonplace-ingest-article"} <= ids
    assert "commonplace-add-publication" not in ids  # both would react to the same event


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
    answer, usage = await synthesize(_FailingLLM(), "What is throttling?", PASSAGES)  # type: ignore[arg-type]
    assert usage is None  # nothing was billed
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


# --- answers: found in review ---


def test_citations_in_lists_and_ranges_are_recognised() -> None:
    assert [c["citation_idx"] for c in cited("Both agree [1, 2].", PASSAGES)] == [1, 2]
    assert [c["citation_idx"] for c in cited("All three say so [1-3].", PASSAGES)] == [1, 2, 3]
    assert [c["citation_idx"] for c in cited("Stacked [2][3] and spaced [ 1 ].", PASSAGES)] == [
        1,
        2,
        3,
    ]
    assert cited("Passage [10] does not exist, and 1 is not cited.", PASSAGES) == []


def test_brackets_that_are_not_citations_do_not_break_the_parser() -> None:
    # found in review: each of the first two raised ValueError, after the model had been paid
    assert cited_numbers("Released on [2024-01-15], as [1] says.") == {1}
    assert cited_numbers("A version string [1-2-3] is not a citation.") == set()
    # A range that is backwards or implausibly wide is not expanded, but its ends
    # still count as cited, so a check on citation bounds can see them.
    assert cited_numbers("backwards [9-7], far apart [7-60]") == {7, 9, 60}
    assert cited_numbers("[ 1 , 3-4 ]") == {1, 3, 4}


class _LLM:
    """An Anthropic client stand-in whose one call either raises or returns `response`."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        async def create(**_: Any) -> Any:
            if error is not None:
                raise error
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


async def test_any_exception_from_the_llm_becomes_the_answer_text() -> None:
    # The SDK also raises plain errors, e.g. ValueError when max_tokens needs streaming.
    llm: Any = _LLM(error=ValueError("Streaming is required"))
    answer, _ = await synthesize(llm, "What is throttling?", PASSAGES)
    assert answer == "LLM synthesis failed (ValueError): Streaming is required"


async def test_a_response_without_text_is_reported_not_returned_empty() -> None:
    spent = SimpleNamespace(input_tokens=900, output_tokens=8000)
    llm: Any = _LLM(response=SimpleNamespace(stop_reason="max_tokens", content=[], usage=spent))
    answer, usage = await synthesize(llm, "What is throttling?", PASSAGES)
    assert "max_tokens" in answer and cited(answer, PASSAGES) == []
    assert usage is not None and usage["output_tokens"] == 8000  # billed even without an answer


def test_a_manual_url_must_be_http() -> None:
    with pytest.raises(ValueError, match="http"):
        resolve_manual_article("ftp://magazine.sebastianraschka.com/p/a-post", [AHEAD_OF_AI])


async def test_search_reports_an_embedding_outage_as_bad_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def search(*_: Any) -> list[dict[str, Any]]:
        raise openai.OpenAIError("provider is down")

    monkeypatch.setitem(operations._RETRIEVERS, "dense", SimpleNamespace(search=search))
    with pytest.raises(operations.OperationError) as raised:
        await operations.search("anything", mode="dense")
    assert raised.value.status == 502
