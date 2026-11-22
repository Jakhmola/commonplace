"""cp/article.discovered -> ingest-article.

Seven steps, each memoized by Inngest. If `embed` (the paid call) fails, a
retry replays the function but `fetch-html`, `parse` and `chunk` return their
recorded results instantly: the page is not fetched again and nothing upstream
is paid for twice. Every write is an upsert, so re-sending the event is safe.

Flow control is configuration on the function, enforced in Inngest's queue
before any of this code runs: concurrency per publication, a global throttle
on run starts, and a priority that lets manual articles overtake a backfill.

There is deliberately no idempotency key: it would refuse a re-send of the
same article for 24 hours, and re-sending is how a failed article is retried.
"""

from datetime import timedelta
from typing import Any

import inngest

from commonplace.config import settings
from commonplace.db.queries import replace_passages, upsert_article
from commonplace.db.session import get_session
from commonplace.inngest_client import client
from commonplace.schemas.events import (
    ARTICLE_DISCOVERED,
    ARTICLE_INGESTED,
    ArticleDiscovered,
    ArticleIngested,
    event,
)
from commonplace.sources.allowlist import assert_allowed, load_publications
from commonplace.sources.chunking import chunk_text
from commonplace.sources.embedder import embed_texts, pack, unpack
from commonplace.sources.fetcher import FetchError, fetch_article
from commonplace.sources.parser import ParsedArticle, parse_html


@client.create_function(
    fn_id="ingest-article",
    trigger=inngest.TriggerEvent(event=ARTICLE_DISCOVERED),
    # One limit per publication: never more than N requests at a time against one
    # site, and one large backfill cannot starve the other feeds.
    concurrency=[
        inngest.Concurrency(
            key="event.data.publication_id", limit=settings.concurrency_per_publication
        )
    ],
    # No key: the embedding quota is shared, so the limit is too. Throttle limits
    # run starts, and each run embeds its whole article in about one request.
    throttle=inngest.Throttle(limit=settings.ingest_runs_per_minute, period=timedelta(minutes=1)),
    # Manual articles are queued as if they had arrived earlier, backfill later.
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

    async def persist() -> dict[str, object]:
        async with get_session() as session:
            article_id = await upsert_article(
                session,
                canonical_id=payload.canonical_id,
                publication_id=payload.publication_id,
                url=str(page["url"]),
                # The feed's title and byline are cleaner than page metadata,
                # which appends the site name or a stray separator.
                title=payload.title or article.title,
                author=payload.author or article.author,
                published_at=article.published_at,
                content_hash=article.content_hash,
                text=article.text,
            )
            passage_count = await replace_passages(
                session,
                article_id=article_id,
                passages=[
                    {"chunk_index": i, "text": text, "embedding": unpack(vector)}
                    for i, (text, vector) in enumerate(
                        zip(chunks, embedded["vectors"], strict=True)
                    )
                ],
            )
            await session.commit()
        return {"article_id": str(article_id), "passage_count": passage_count}

    persisted = await ctx.step.run("persist", persist)

    done = ArticleIngested(
        canonical_id=payload.canonical_id,
        article_id=persisted["article_id"],
        content_hash=article.content_hash,
        passage_count=persisted["passage_count"],
    )
    await ctx.step.send_event("emit", event(ARTICLE_INGESTED, done))
    return persisted
