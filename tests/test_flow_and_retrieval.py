"""Unit tests for flow control, hybrid retrieval and answers. No Docker or keys needed."""

from typing import Any

import pytest

from commonplace.functions import FUNCTIONS
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
