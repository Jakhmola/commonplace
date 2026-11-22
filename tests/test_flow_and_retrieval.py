"""Unit tests for flow control, hybrid retrieval and answers. No Docker or keys needed."""

from typing import Any

from commonplace.functions import FUNCTIONS
from commonplace.functions.ingest import ingest_article

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
