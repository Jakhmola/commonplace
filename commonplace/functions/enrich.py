"""cp/article.ingested | cp/article.updated -> extract-claims.

Turns an article into claims that can be trusted to the extent that their
quotes are real:

1. `load-article`           the text and metadata from Postgres
2. `extract-claims-llm`     one model call proposes claims (memoized, so a
                            retry of a later step never pays for it again)
3. `validate-quote-spans`   plain code drops every claim whose quote is not in
                            the article, and canonicalises topics
4. `persist-claims-pg`      Postgres: close the previous version's claims,
                            upsert the new ones
5. `sync-claims-memgraph`   the same, merged into the graph
6. `emit`                   cp/claims.extracted with the counts

Postgres is written before the graph on purpose: the graph is derived from
it, so if the graph step fails and is retried, nothing is lost.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import inngest

from commonplace.agents.core import run_agent
from commonplace.agents.roles.extractor import (
    claim_extractor,
    format_extractor_prompt,
    heuristic_extract_claims,
)
from commonplace.config import settings
from commonplace.db.queries import (
    get_active_claims_for_article,
    get_article_with_publication,
    persist_article_claims,
)
from commonplace.db.session import get_session
from commonplace.graph.client import close_article_claims, sync_article_claims_to_graph
from commonplace.graph.ontology import normalize_author
from commonplace.inngest_client import client
from commonplace.schemas.entities import ClaimExtractionBatch
from commonplace.schemas.events import (
    ARTICLE_INGESTED,
    ARTICLE_UPDATED,
    CLAIMS_EXTRACTED,
    ArticleIngested,
    ClaimsExtracted,
    event,
)
from commonplace.sources.quote_validator import validate_and_normalize_claims


def should_extract(event_name: str, *, has_claims: bool) -> bool:
    """Whether this event warrants a (paid) extraction.

    An edited article is always extracted again: its claims describe text that
    no longer exists. A plain "ingested" event for an article that already has
    claims is a duplicate, and extracting again would pay for the model twice
    and replace claims that are still right with a different sample of them.
    """
    return event_name == ARTICLE_UPDATED or not has_claims


def should_retire_claims(failure: dict[str, Any]) -> bool:
    """After an extraction failed for good: are the article's current claims now stale?

    Yes when the run was for an edit. Its claims quote text that is gone, and
    with no new extraction to replace them they would stay "current" forever.
    A first extraction that fails leaves nothing behind to retire.
    """
    return (failure.get("event") or {}).get("name") == ARTICLE_UPDATED


def extraction_step_id(*, has_key: bool) -> str:
    """The model call and its no-key stand-in return different shapes, so they
    are different steps: a run replayed after the key changed must not read
    one's memo as the other's.
    """
    return "extract-claims-llm" if has_key else "extract-claims-heuristic"


async def retire_stale_claims(ctx: inngest.Context) -> None:
    """Runs when `extract-claims` has failed for good."""
    failure: dict[str, Any] = dict(ctx.event.data or {})
    if not should_retire_claims(failure):
        return
    original = ArticleIngested.model_validate(failure["event"]["data"])

    async def retire() -> int:
        now = datetime.now(UTC)
        async with get_session() as session:
            await persist_article_claims(
                session, article_id=original.article_id, claims=[], now=now
            )
            await session.commit()
        return await close_article_claims(original.canonical_id, now.isoformat())

    await ctx.step.run("retire-stale-claims", retire)


@client.create_function(
    fn_id="extract-claims",
    trigger=[
        inngest.TriggerEvent(event=ARTICLE_INGESTED),
        inngest.TriggerEvent(event=ARTICLE_UPDATED),
    ],
    # A cap across all articles, to stay inside the model provider's rate limit.
    concurrency=[inngest.Concurrency(limit=settings.concurrency_extract_claims)],
    # One extraction per article, and the newest event wins: if the article is
    # edited while it is being extracted, that run is cancelled and a new one
    # reads the new text. (Concurrency limits steps, not runs, so it cannot
    # keep two runs for one article from interleaving.)
    singleton=inngest.Singleton(key="event.data.canonical_id", mode="cancel"),
    on_failure=retire_stale_claims,
    retries=3,
)
async def extract_claims(ctx: inngest.Context) -> dict[str, object]:
    # cp/article.updated carries the same fields as cp/article.ingested.
    payload = ArticleIngested.model_validate(ctx.event.data)

    async def load_article() -> dict[str, Any] | None:
        async with get_session() as session:
            row = await get_article_with_publication(session, payload.article_id)
            if row is None or not row[0].text.strip():
                return None
            has_claims = bool(await get_active_claims_for_article(session, payload.article_id))
        article, publication = row
        return {
            "has_claims": has_claims,
            "article_id": str(article.id),
            "canonical_id": article.canonical_id,
            "url": article.url,
            "title": article.title,
            "author": article.author,
            "published_at": article.published_at.isoformat(),
            "content_hash": article.content_hash,
            "text": article.text,
            "publication": {
                "slug": publication.slug,
                "name": publication.name,
                "homepage": publication.homepage,
            },
        }

    article = await ctx.step.run("load-article", load_article)
    summary = ClaimsExtracted(
        article_id=payload.article_id, canonical_id=payload.canonical_id, claim_count=0
    )
    if article is None:
        return {**summary.model_dump(mode="json"), "status": "skipped_empty"}
    if not should_extract(ctx.event.name, has_claims=article["has_claims"]):
        return {**summary.model_dump(mode="json"), "status": "already_extracted"}
    publication: dict[str, str] = article["publication"]

    if settings.anthropic_api_key:
        batch = await run_agent(
            ctx.step,
            claim_extractor,
            format_extractor_prompt(
                title=article["title"],
                author=article["author"],
                publication_name=publication["name"],
                article_text=article["text"],
            ),
            extraction_step_id(has_key=True),
            payload.article_id,
        )
    else:
        # No key: a crude stand-in, so the rest of the pipeline can still be run.
        async def heuristic() -> dict[str, Any]:
            return heuristic_extract_claims(article["text"]).model_dump(mode="json")

        batch = ClaimExtractionBatch.model_validate(
            await ctx.step.run(extraction_step_id(has_key=False), heuristic)
        )

    async def validate_quote_spans() -> dict[str, Any]:
        assert isinstance(batch, ClaimExtractionBatch)
        valid, dropped = validate_and_normalize_claims(
            article["article_id"],
            article["content_hash"],
            article["text"],
            batch.claims,
            fallback_topic=publication["name"],
        )
        return {"claims": [c.model_dump(mode="json") for c in valid], "dropped_spans": dropped}

    validated = await ctx.step.run("validate-quote-spans", validate_quote_spans)
    claims: list[dict[str, Any]] = validated["claims"]

    async def persist_claims() -> dict[str, Any]:
        now = datetime.now(UTC)
        async with get_session() as session:
            pairs = await persist_article_claims(
                session, article_id=uuid.UUID(article["article_id"]), claims=claims, now=now
            )
            await session.commit()
        return {"supersedes_pairs": pairs, "now_iso": now.isoformat()}

    persisted = await ctx.step.run("persist-claims-pg", persist_claims)

    async def sync_graph() -> dict[str, int]:
        return await sync_article_claims_to_graph(
            publication=publication,
            # A post without a byline is attributed to the publication itself.
            author=normalize_author(article["author"] or publication["name"]),
            article={
                key: article[key]
                for key in (
                    "canonical_id",
                    "article_id",
                    "url",
                    "title",
                    "published_at",
                    "content_hash",
                )
            },
            claims=[{**c, "valid_from": persisted["now_iso"]} for c in claims],
            supersedes_pairs=persisted["supersedes_pairs"],
            now_iso=persisted["now_iso"],
        )

    await ctx.step.run("sync-claims-memgraph", sync_graph)

    summary = summary.model_copy(
        update={"claim_count": len(claims), "dropped_spans": validated["dropped_spans"]}
    )
    await ctx.step.send_event("emit", event(CLAIMS_EXTRACTED, summary))
    return summary.model_dump(mode="json")
