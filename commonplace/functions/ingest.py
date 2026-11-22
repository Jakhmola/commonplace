"""cp/article.discovered -> ingest-article.

Eight steps, each memoized by Inngest. If `embed` (the paid call) fails, a
retry replays the function but `fetch-html`, `parse` and `chunk` return their
recorded results instantly: the page is not fetched again and nothing upstream
is paid for twice.

After `parse`, the content hash decides what happens. An article whose text
has not changed stops there. A changed article has its current passages
closed (`valid_to`) and new ones inserted, so history is kept and searches
only ever see the current version.

Flow control is configuration on the function, enforced in Inngest's queue
before any of this code runs: concurrency per publication, a global throttle
on run starts, and a priority that lets manual articles overtake a backfill.

There is deliberately no idempotency key: it would refuse a re-send of the
same article for 24 hours, and re-sending is how a failed article is retried.
"""

from datetime import timedelta
from typing import Any, Literal

import inngest
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.config import settings
from commonplace.db.queries import (
    count_current_passages,
    get_active_claims_for_article,
    get_article_by_canonical_id,
    store_passages,
    upsert_article,
)
from commonplace.db.session import get_session
from commonplace.inngest_client import client
from commonplace.schemas.events import (
    ARTICLE_DISCOVERED,
    ARTICLE_INGESTED,
    ARTICLE_UPDATED,
    ArticleDiscovered,
    ArticleIngested,
    ArticleUpdated,
    event,
)
from commonplace.sources.allowlist import assert_allowed, load_publications
from commonplace.sources.chunking import chunk_text
from commonplace.sources.embedder import embed_texts, pack, unpack
from commonplace.sources.fetcher import FetchError, fetch_article
from commonplace.sources.parser import ParsedArticle, parse_html

Change = Literal["new", "unchanged", "changed"]


def classify_change(stored_hash: str | None, new_hash: str) -> Change:
    """What a freshly parsed article is, relative to the version already stored."""
    if stored_hash is None:
        return "new"
    return "unchanged" if stored_hash == new_hash else "changed"


async def persist_article(
    session: AsyncSession,
    payload: ArticleDiscovered,
    page_url: str,
    article: ParsedArticle,
    chunks: list[str],
    vectors: list[str],
) -> dict[str, Any]:
    """Store the article and make its passages the current version, in one transaction.

    Safe to run twice: if this exact version is already stored with its
    passages (the step's response was lost after the commit and Inngest runs it
    again), nothing is written, so the passages are not superseded by themselves.
    """
    existing = await get_article_by_canonical_id(session, payload.canonical_id)
    if existing is not None and existing.content_hash == article.content_hash:
        current = await count_current_passages(session, existing.id)
        if current:
            return {"article_id": str(existing.id), "passage_count": current}

    article_id = await upsert_article(
        session,
        canonical_id=payload.canonical_id,
        publication_id=payload.publication_id,
        url=page_url,
        # The feed's title and byline are cleaner than page metadata,
        # which appends the site name or a stray separator.
        title=payload.title or article.title,
        author=payload.author or article.author,
        published_at=article.published_at,
        content_hash=article.content_hash,
        text=article.text,
    )
    passage_count = await store_passages(
        session,
        article_id=article_id,
        passages=[
            {"chunk_index": i, "text": text, "embedding": unpack(vector)}
            for i, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
    await session.commit()
    return {"article_id": str(article_id), "passage_count": passage_count}


@client.create_function(
    fn_id="ingest-article",
    trigger=inngest.TriggerEvent(event=ARTICLE_DISCOVERED),
    # One limit per publication: never more than N requests at a time against one
    # site, and one large backfill cannot starve the other feeds.
    concurrency=[
        inngest.Concurrency(
            key="event.data.publication_id", limit=settings.concurrency_per_publication
        ),
    ],
    # One run per article at a time. Concurrency limits steps, not runs, so it
    # cannot do this: two runs for one article would interleave between steps
    # and both embed. A second event for an article that is being ingested (a
    # poll and a manual add, a feed submitted twice) is skipped.
    singleton=inngest.Singleton(key="event.data.canonical_id", mode="skip"),
    # No key: the embedding quota is shared, so the limit is too. Throttle limits
    # run starts, and each run embeds its whole article in about one request.
    throttle=inngest.Throttle(limit=settings.ingest_runs_per_minute, period=timedelta(minutes=1)),
    # Manual articles are queued as if they had arrived earlier than they did.
    priority=inngest.Priority(
        run=(
            f"event.data.via == 'manual' ? {settings.priority_manual_s} "
            f": {settings.priority_backfill_s}"
        )
    ),
    retries=3,
)
async def ingest_article(ctx: inngest.Context) -> dict[str, object]:
    payload = ArticleDiscovered.model_validate(ctx.event.data)

    async def check_allowlist() -> None:
        try:
            assert_allowed(payload.url, load_publications())
        except ValueError as exc:
            raise inngest.NonRetriableError(str(exc)) from exc

    await ctx.step.run("check-allowlist", check_allowlist)

    async def fetch_html() -> dict[str, object]:
        publications = load_publications()

        def stay_on_allowlist(url: str) -> None:
            assert_allowed(url, publications)

        try:
            # Checked for every hop, so a redirect cannot lead off the allowlist.
            page = await fetch_article(payload.url, check_url=stay_on_allowlist)
        except (FetchError, ValueError) as exc:
            raise inngest.NonRetriableError(str(exc)) from exc
        return {"url": page.url, "html": page.html, "status_code": page.status_code}

    page = await ctx.step.run("fetch-html", fetch_html)

    async def parse() -> dict[str, object]:
        return parse_html(
            str(page["html"]), str(page["url"]), fallback_published_at=payload.published_at
        ).model_dump(mode="json")

    article = ParsedArticle.model_validate(await ctx.step.run("parse", parse))

    async def check_content_change() -> dict[str, Any]:
        async with get_session() as session:
            stored = await get_article_by_canonical_id(session, payload.canonical_id)
            has_claims = bool(stored and await get_active_claims_for_article(session, stored.id))
        return {
            "change": classify_change(
                stored.content_hash if stored else None, article.content_hash
            ),
            "article_id": str(stored.id) if stored else None,
            "has_claims": has_claims,
        }

    checked = await ctx.step.run("check-content-change", check_content_change)
    change: Change = checked["change"]
    if change == "unchanged":
        # Same text as last time: stop here, before chunking and the paid embedding call.
        if not checked["has_claims"]:
            # Stored before claim extraction existed, or its extraction failed:
            # announce it again so `extract-claims` picks it up.
            unextracted = ArticleIngested(
                canonical_id=payload.canonical_id,
                article_id=checked["article_id"],
                content_hash=article.content_hash,
                passage_count=0,
            )
            await ctx.step.send_event("emit-for-enrichment", event(ARTICLE_INGESTED, unextracted))
        return {"article_id": checked["article_id"], "passage_count": 0, "status": "unchanged"}

    async def chunk() -> list[str]:
        return chunk_text(
            article.text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

    chunks = await ctx.step.run("chunk", chunk)

    async def embed() -> dict[str, Any]:
        result = await embed_texts(chunks)
        # Packed, or a long article's vectors would not fit in a step output.
        return {"vectors": [pack(v) for v in result["embeddings"]], "usage": result["usage"]}

    embedded = await ctx.step.run("embed", embed)

    async def persist() -> dict[str, Any]:
        async with get_session() as session:
            return await persist_article(
                session, payload, str(page["url"]), article, chunks, embedded["vectors"]
            )

    persisted = await ctx.step.run("persist", persist)

    # An edit is announced differently from a first ingest, so listeners can
    # tell "new article" from "this article's passages were replaced".
    name, model = (
        (ARTICLE_UPDATED, ArticleUpdated)
        if change == "changed"
        else (ARTICLE_INGESTED, ArticleIngested)
    )
    done = model(
        canonical_id=payload.canonical_id,
        article_id=persisted["article_id"],
        content_hash=article.content_hash,
        passage_count=persisted["passage_count"],
    )
    await ctx.step.send_event("emit", event(name, done))
    return {**persisted, "status": change}
