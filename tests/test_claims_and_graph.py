"""Unit tests for the ontology, the quote guardrail, the agent layer and graph retrieval.

No Docker or keys needed. Tests that need Postgres or Memgraph live in tests/integration.
"""

import re
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from commonplace import operations
from commonplace.agents import core
from commonplace.agents.core import run_agent
from commonplace.agents.providers import estimate_cost_usd
from commonplace.agents.roles.extractor import (
    claim_extractor,
    format_extractor_prompt,
    heuristic_extract_claims,
)
from commonplace.functions import FUNCTIONS
from commonplace.functions.ask import cited, format_context
from commonplace.functions.enrich import extract_claims, should_extract
from commonplace.graph import cypher
from commonplace.graph.ontology import (
    EDGE_TYPES,
    NODE_LABELS,
    normalize_author,
    normalize_topic,
    normalize_topics,
    slugify,
)
from commonplace.retrieval import graph as graph_retrieval
from commonplace.retrieval.graph import extract_query_seed_topics
from commonplace.retrieval.hybrid import compute_rrf
from commonplace.schemas.entities import ClaimExtractionBatch, RawExtractedClaim
from commonplace.schemas.events import CLAIMS_EXTRACTED, ClaimsExtracted
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
    # names that merely end in "s" are left alone; plural acronyms are not
    assert normalize_topic("Synopsys").slug == "synopsys"  # type: ignore[union-attr]
    assert normalize_topic("GPUs").slug == "gpu"  # type: ignore[union-attr]
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


# --- the agent layer: one LLM call, inside one durable step, validated ---


class _Step:
    """Stands in for `ctx.step`: runs the callable and remembers the step ids."""

    def __init__(self) -> None:
        self.ids: list[str] = []

    async def run(self, step_id: str, fn: Any) -> Any:
        self.ids.append(step_id)
        return await fn()


async def test_run_agent_runs_inside_a_step_and_validates_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_call_llm(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        claim = {
            "claim_text": "Steps are memoized",
            "quote_span": "every step that already finished returns its recorded result",
            "claim_type": "measurement",
            "entities": ["Durable Execution"],
        }
        usage = {"model": "m", "input_tokens": 1200, "output_tokens": 80, "est_usd": 0.0064}
        return {"output": {"claims": [claim]}, "usage": usage, "stop_reason": "end_turn"}

    monkeypatch.setattr(core, "call_llm", fake_call_llm)
    step = _Step()
    batch = await run_agent(step, claim_extractor, "the article", "extract-claims-llm", ARTICLE_ID)

    assert step.ids == ["extract-claims-llm"]  # memoized by Inngest, and its usage is captured
    assert isinstance(batch, ClaimExtractionBatch)
    assert batch.claims[0].claim_type == "measurement"
    assert seen["output_schema"] is ClaimExtractionBatch
    assert seen["system_prompt"] == claim_extractor.system_prompt
    assert seen["article_id"] == ARTICLE_ID


async def test_run_agent_rejects_output_that_does_not_fit_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_llm(**_: Any) -> dict[str, Any]:
        return {"output": {"claims": [{"claim_text": "x"}]}, "usage": {}, "stop_reason": "end_turn"}

    monkeypatch.setattr(core, "call_llm", fake_call_llm)
    with pytest.raises(ValidationError):
        await run_agent(_Step(), claim_extractor, "the article", "extract-claims-llm")


def test_cost_is_estimated_from_the_model_price_table() -> None:
    # 1M input + 1M output at the listed prices
    assert estimate_cost_usd("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert estimate_cost_usd("claude-haiku-5-5", 2_000_000, 0) == pytest.approx(0.20)
    # an unknown model is priced like the default one rather than as free
    assert estimate_cost_usd("some-future-model", 1_000_000, 0) == pytest.approx(4.0)


def test_extractor_prompt_carries_title_author_and_text() -> None:
    prompt = format_extractor_prompt(
        title="Why durable steps matter",
        author=None,
        publication_name="Latent Space",
        article_text=ARTICLE,
    )
    assert "Publication: Latent Space" in prompt and "Title: Why durable steps matter" in prompt
    assert "Author: Unknown" in prompt and prompt.endswith(ARTICLE)


def test_the_heuristic_extractor_quotes_real_sentences() -> None:
    text = (
        "Retrieval augmented generation is still the default for grounding answers. "
        "Nothing here mentions a known topic at all, only weather. "
        "In our runs, fine-tuning cut latency by 40% on the same hardware. "
        "We think coding agents will write most pull requests within two years."
    )
    batch = heuristic_extract_claims(text)
    assert [c.claim_type for c in batch.claims] == ["opinion", "measurement", "prediction"]
    for claim in batch.claims:
        assert find_verbatim(text, claim.quote_span) == claim.quote_span  # passes the guardrail
    assert batch.claims[0].entities == ["RAG"]


# --- sources: an id from a citation leads back to the exact text ---


async def test_a_source_id_must_be_a_uuid() -> None:
    with pytest.raises(operations.OperationError) as raised:
        await operations.get_source("not-a-uuid")
    assert raised.value.status == 400


# --- cypher: the three rules every statement must keep ---


def _statements() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(cypher).items()
        if name.isupper() and isinstance(value, str)
    }


def test_cypher_is_parameterized_merges_only_and_reads_filter_validity() -> None:
    statements = _statements()
    assert len(statements) >= 6
    for name, statement in statements.items():
        # 1. values arrive as $parameters, never by string formatting
        assert "{}" not in statement and "%s" not in statement, name
        assert not re.search(r"\{[a-z_]+\}", statement), name
        # 2. writes MERGE on a key, so a replay cannot create a second node
        # (`ON CREATE SET` is part of a MERGE, not a CREATE)
        assert not re.search(r"(?<!ON )\bCREATE\b", statement), name
        # 3. anything that reads claims says which ones are current
        if "RETURN" in statement and ":Claim" in statement and "MERGE" not in statement:
            assert "valid_to IS NULL" in statement, name
    # the schema statements are the one place CREATE is allowed (indexes, constraints)
    assert all(s.startswith("CREATE ") for s in cypher.SCHEMA_STATEMENTS)


# --- the extract-claims function ---


def _extract_config() -> Any:
    return extract_claims.get_config("http://localhost:8000").main


def test_extraction_reacts_to_new_and_updated_articles() -> None:
    events = {getattr(t, "event", None) for t in _extract_config().triggers}
    assert events == {"cp/article.ingested", "cp/article.updated"}
    assert extract_claims in FUNCTIONS


def test_extraction_is_serialized_per_article_and_capped_globally() -> None:
    limits = {c.key: c.limit for c in _extract_config().concurrency}
    # two quick edits to one article must not interleave their writes
    assert limits["event.data.canonical_id"] == 1
    # and the model provider is called by at most this many runs at once
    assert limits[None] == 2


def test_claims_extracted_summarises_the_guardrail() -> None:
    summary = ClaimsExtracted(
        article_id=ARTICLE_ID,
        canonical_id="pub:latent-space:a-post",
        claim_count=6,
        dropped_spans=2,
    )
    assert CLAIMS_EXTRACTED == "cp/claims.extracted"
    assert summary.model_dump(mode="json") == {
        "article_id": str(ARTICLE_ID),
        "canonical_id": "pub:latent-space:a-post",
        "claim_count": 6,
        "dropped_spans": 2,
    }


# --- the graph as a third retriever ---


def _passage(passage_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": f"text of {passage_id}",
        "title": "A post",
        "author": "Jane Writer",
        "url": "https://www.latent.space/p/a-post",
        "published_at": "2026-11-17T00:00:00+00:00",
        "rank": 0.5,
        **extra,
    }


def test_three_way_fusion_rewards_agreement_across_all_engines() -> None:
    fused = compute_rrf(
        [_passage("only-sparse"), _passage("everywhere")],
        [_passage("only-dense"), _passage("everywhere")],
        [_passage("only-graph"), _passage("everywhere")],
    )
    best = fused[0]
    assert (best.passage_id, best.source) == ("everywhere", "all")
    assert best.rrf_score == pytest.approx(3 / 62)  # second in each of three lists
    assert (best.sparse_rank, best.dense_rank, best.graph_rank) == (2, 2, 2)
    assert [c.passage_id for c in fused[1:]] == ["only-dense", "only-graph", "only-sparse"]


def test_sources_are_labelled_by_the_engines_that_found_them() -> None:
    fused = compute_rrf(
        [_passage("s"), _passage("sd"), _passage("sg"), _passage("sdg")],
        [_passage("d"), _passage("sd"), _passage("dg"), _passage("sdg")],
        [_passage("g"), _passage("sg"), _passage("dg"), _passage("sdg")],
    )
    assert {c.passage_id: c.source for c in fused} == {
        "s": "bm25",
        "d": "vector",
        "g": "graph",
        "sd": "both",
        "sg": "bm25+graph",
        "dg": "vector+graph",
        "sdg": "all",
    }


def test_a_graph_hit_brings_its_claim_to_an_existing_candidate() -> None:
    claim = {"claim_id": "c-1", "claim_text": "Steps are memoized", "claim_type": "measurement"}
    second_claim = {"claim_id": "c-2", "claim_text": "Another", "claim_type": "opinion"}
    fused = compute_rrf(
        [_passage("p1")],
        [],
        # two claims quoted from the same passage: the passage counts once, first claim kept
        [_passage("p1", **claim), _passage("p1", **second_claim), _passage("p2", **second_claim)],
    )
    by_id = {c.passage_id: c for c in fused}
    assert (by_id["p1"].claim_id, by_id["p1"].claim_type) == ("c-1", "measurement")
    assert (by_id["p1"].sparse_rank, by_id["p1"].graph_rank) == (1, 1)
    assert by_id["p1"].rrf_score == pytest.approx(2 / 61)
    assert (by_id["p2"].source, by_id["p2"].claim_text, by_id["p2"].graph_rank) == (
        "graph",
        "Another",
        2,
    )


def test_two_way_fusion_still_works_without_a_graph() -> None:
    fused = compute_rrf([_passage("a")], [_passage("a")])
    assert (fused[0].source, fused[0].graph_rank, fused[0].claim_id) == ("both", None, None)


def test_seed_topics_come_from_aliases_and_phrases() -> None:
    slugs, terms = extract_query_seed_topics(
        "How do RAG and fine-tuning compare for coding agents in production?"
    )
    assert {"rag", "fine-tuning", "coding-agent"} <= set(slugs)
    assert "large-language-model" not in slugs and "llm" not in slugs
    assert not {"how", "do", "and", "for", "in"} & set(terms)
    assert {"rag", "coding", "agents"} <= set(terms)

    slugs, _ = extract_query_seed_topics("What do authors say about large language models?")
    assert "llm" in slugs  # a multi-word alias resolves to its canonical topic
    assert extract_query_seed_topics("what is it about?") == ([], [])


async def test_graph_leg_failure_degrades_to_two_way(monkeypatch: pytest.MonkeyPatch) -> None:
    async def memgraph_is_down(**_: Any) -> list[dict[str, Any]]:
        raise ConnectionError("memgraph is down")

    monkeypatch.setattr(graph_retrieval, "search_graph_by_topics", memgraph_is_down)
    session: Any = None  # never reached: the graph is asked first
    assert await graph_retrieval.search_graph_claims(session, "RAG evaluation") == []


# --- answers know about claims ---


def test_context_shows_the_attributed_claim_with_its_type() -> None:
    passages = [
        _passage("p1", claim_id="c-1", claim_text="Retries add 40 ms", claim_type="measurement"),
        _passage("p2"),
    ]
    context = format_context(passages)
    first, second = context.split("\n\n")
    assert first.splitlines()[1] == "Attributed claim (measurement): Retries add 40 ms"
    assert first.splitlines()[2] == "text of p1"
    assert "Attributed claim" not in second

    citations = cited("It adds 40 ms [1], see also [2].", passages)
    assert (citations[0]["claim_id"], citations[0]["claim_type"]) == ("c-1", "measurement")
    assert "claim_id" not in citations[1]


def test_an_article_is_not_extracted_twice_unless_it_changed() -> None:
    # A duplicate "ingested" event must not pay for the model again, or replace
    # the claims of an article whose text is the same.
    assert should_extract("cp/article.ingested", has_claims=False) is True
    assert should_extract("cp/article.ingested", has_claims=True) is False
    # an edit always re-extracts: its claims describe the old text
    assert should_extract("cp/article.updated", has_claims=True) is True
