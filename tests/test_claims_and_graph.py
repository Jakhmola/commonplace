"""Unit tests for the ontology, the quote guardrail, the agent layer and graph retrieval.

No Docker or keys needed. Tests that need Postgres or Memgraph live in tests/integration.
"""

import uuid

import pytest
from pydantic import ValidationError

from commonplace.graph.ontology import (
    EDGE_TYPES,
    NODE_LABELS,
    normalize_author,
    normalize_topic,
    normalize_topics,
    slugify,
)
from commonplace.schemas.entities import RawExtractedClaim
from commonplace.schemas.ids import claim_id
from commonplace.sources.quote_validator import find_verbatim, validate_and_normalize_claims

# --- ontology: closed, and every name is canonicalised before it becomes a node ---


def test_the_ontology_is_closed_at_five_and_five() -> None:
    assert {"Publication", "Author", "Article", "Claim", "Topic"} == NODE_LABELS
    assert {"PUBLISHED_IN", "WRITTEN_BY", "ASSERTS", "ABOUT", "SUPERSEDES"} == EDGE_TYPES


def test_synonyms_collapse_to_one_topic() -> None:
    variants = ["RAG", "retrieval augmented generation", "Retrieval-Augmented Generation", "rag."]
    assert {normalize_topic(v).slug for v in variants} == {"rag"}  # type: ignore[union-attr]
    assert normalize_topic("LLMs").name == "LLM"  # type: ignore[union-attr]
    assert normalize_topic("reinforcement learning").slug == normalize_topic("RL").slug  # type: ignore[union-attr]


def test_plurals_collapse_and_generic_words_are_rejected() -> None:
    assert normalize_topic("Coding Agents").slug == "coding-agent"  # type: ignore[union-attr]
    assert normalize_topic("coding agent").slug == "coding-agent"  # type: ignore[union-attr]
    assert normalize_topic("open weights").name == "Open Weight"  # type: ignore[union-attr]
    assert normalize_topic("analytics").slug == "analytics"  # type: ignore[union-attr]
    for generic in ("AI", "models", "system", "data", "  ", "x"):
        assert normalize_topic(generic) is None, generic


def test_topics_are_deduplicated_in_order() -> None:
    topics = normalize_topics(["Fine-tuning", "RAG", "fine tuning", "AI", "rag", "GPU clusters"])
    assert [t.slug for t in topics] == ["fine-tuning", "rag", "gpu-cluster"]
    assert topics[2].name == "GPU Cluster"


def test_author_names_normalize_with_a_fallback() -> None:
    assert normalize_author("  Sebastian   Raschka, PhD ").slug == "sebastian-raschka-phd"
    assert normalize_author("  Sebastian   Raschka, PhD ").name == "Sebastian Raschka, PhD"
    assert normalize_author(None).slug == normalize_author("  ").slug == "unknown-author"
    assert slugify("Zoë's Café: notes") == "zoe-s-cafe-notes"


# --- claim ids: one id per claim per article version ---

ARTICLE_ID = uuid.UUID("6f1c0a3e-9d5b-4c55-8f0e-2b7a4d1c9e10")


def test_claim_ids_ignore_whitespace_and_case_but_not_content() -> None:
    base = claim_id(ARTICLE_ID, "hash-1", "Durable steps are memoized.", "Steps are memoized")
    reflowed = claim_id(
        ARTICLE_ID, "hash-1", "durable  steps\nare memoized.", "  steps are   Memoized "
    )
    assert base == reflowed  # a replay converges on the same row and node
    assert base != claim_id(
        ARTICLE_ID, "hash-2", "Durable steps are memoized.", "Steps are memoized"
    )
    assert base != claim_id(ARTICLE_ID, "hash-1", "Durable steps are memoized.", "Steps are cached")
    assert base.version == 5


# --- the quote guardrail: a claim survives only if its quote is in the article ---

ARTICLE = (
    "Durable execution changes how you think about failure.\n\n"
    "When a step fails, the engine replays the function from the top, but every step that "
    "already finished returns its recorded result. That’s why the page is fetched “exactly "
    "once” – even if the embedding provider is down for an hour.\n"
    "In our tests the retry added 40 ms of overhead."
)


def test_exact_and_reflowed_quotes_pass() -> None:
    exact = "every step that already finished returns its recorded result"
    assert find_verbatim(ARTICLE, exact) == exact
    reflowed = "changes how you think about failure. When a step fails, the engine"
    assert find_verbatim(ARTICLE, reflowed.replace(" ", "\n", 3)) == reflowed


def test_typographic_variants_match_and_the_article_text_is_kept() -> None:
    typed = 'That\'s why the page is fetched "exactly once" - even if the embedding provider'
    found = find_verbatim(ARTICLE, typed)
    # the model wrote straight quotes and a hyphen; what is stored is the article's own text
    assert found == "That’s why the page is fetched “exactly once” – even if the embedding provider"


def test_paraphrased_or_stitched_quotes_are_dropped() -> None:
    assert find_verbatim(ARTICLE, "the engine re-runs the whole function from the start") is None
    stitched = "the engine replays the function ... the page is fetched exactly once"
    assert find_verbatim(ARTICLE, stitched) is None


def test_short_or_empty_quotes_are_dropped() -> None:
    assert find_verbatim(ARTICLE, "40 ms") is None  # real, but too short to be evidence
    assert find_verbatim(ARTICLE, "") is None
    assert find_verbatim("", "anything at all, long enough") is None


def _raw(quote: str, entities: list[str], text: str = "A claim about steps") -> RawExtractedClaim:
    return RawExtractedClaim(
        claim_text=text, quote_span=quote, claim_type="opinion", entities=entities
    )


def test_validation_keeps_valid_claims_and_counts_the_dropped() -> None:
    real = "every step that already finished returns its recorded result"
    raw = [
        _raw(real, ["Durable Execution", "AI"]),
        _raw(real, ["durable execution"]),  # the same claim again
        _raw("this sentence is not in the article at all", ["Inngest"]),
        _raw("In our tests the retry added 40 ms of overhead.", ["AI", "systems"]),  # no topic
    ]
    valid, dropped = validate_and_normalize_claims(ARTICLE_ID, "hash-1", ARTICLE, raw)
    assert [(c.quote_span, [t.slug for t in c.topics]) for c in valid] == [
        (real, ["durable-execution"])
    ]
    assert dropped == 2
    assert valid[0].claim_id == str(claim_id(ARTICLE_ID, "hash-1", real, "A claim about steps"))

    # with a fallback topic, a verbatim claim whose own topics were all generic is kept
    valid, dropped = validate_and_normalize_claims(
        ARTICLE_ID, "hash-1", ARTICLE, raw, fallback_topic="Latent Space"
    )
    assert (len(valid), dropped) == (2, 1)
    assert valid[1].topics[0].slug == "latent-space"


def test_claim_types_are_a_closed_set() -> None:
    with pytest.raises(ValidationError):
        RawExtractedClaim(
            claim_text="A claim about steps",
            quote_span="every step that already finished",
            claim_type="rumour",
            entities=["Inngest"],
        )
