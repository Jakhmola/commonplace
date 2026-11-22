"""cp/article.discovered -> ingest-article.

Seven steps, each memoized by Inngest. If `embed` (the paid call) fails, a
retry replays the function but `fetch-html`, `parse` and `chunk` return their
recorded results instantly: the page is not fetched again and nothing upstream
is paid for twice. Every write is an upsert, so re-sending the event is safe.
"""

from typing import Any
from uuid import UUID

import inngest

from commonplace import db
from commonplace.config import settings
from commonplace.events import (
    ARTICLE_DISCOVERED,
    ARTICLE_INGESTED,
    ArticleDiscovered,
    ArticleIngested,
    event,
)
from commonplace.inngest_client import client
from commonplace.sources.allowlist import NotAllowed, assert_allowed, load_publications
from commonplace.sources.chunk import chunk
from commonplace.sources.embed import cost_usd, embed, pack, unpack
from commonplace.sources.fetch import FetchRefused, fetch
from commonplace.sources.parse import ArticleText, extract_article


@client.create_function(
    fn_id="ingest-article",
    trigger=inngest.TriggerEvent(event=ARTICLE_DISCOVERED),
    retries=3,
)
async def ingest_article(ctx: inngest.Context) -> dict[str, object]:
    d = ArticleDiscovered.model_validate(ctx.event.data)

    async def check_allowlist() -> None:
        try:
            assert_allowed(d.url, load_publications())
        except NotAllowed as e:
            raise inngest.NonRetriableError(str(e)) from e

    await ctx.step.run("check-allowlist", check_allowlist)

    async def fetch_html() -> dict[str, str]:
        try:
            final_url, body = await fetch(d.url)
        except FetchRefused as e:
            raise inngest.NonRetriableError(str(e)) from e
        return {"url": final_url, "html": body.decode("utf-8", errors="replace")}

    page = await ctx.step.run("fetch-html", fetch_html)

    async def parse() -> dict[str, object]:
        return extract_article(page["html"], page["url"]).model_dump(mode="json")

    art = ArticleText.model_validate(await ctx.step.run("parse", parse))

    async def chunk_step() -> list[str]:
        return chunk(art.text, settings.chunk_size, settings.chunk_overlap)

    chunks = await ctx.step.run("chunk", chunk_step)

    async def embed_step() -> dict[str, Any]:
        vectors, tokens = await embed(chunks)
        return {"vectors": [pack(v) for v in vectors], "tokens": tokens}

    embedded = await ctx.step.run("embed", embed_step)

    async def persist() -> str:
        tokens: int = embedded["tokens"]
        async with db.tx() as conn:
            article_id = await db.save_article(
                conn,
                canonical_id=d.canonical_id,
                publication_id=d.publication_id,
                url=page["url"],
                # The feed's title/author are cleaner than page metadata
                # (which appends the site name, or a stray "·" to the byline).
                title=d.title,
                author=d.author or art.author,
                published_at=art.published_at or d.published_at,
                content_hash=art.content_hash,
                body=art.text,
                chunks=chunks,
                vectors=[unpack(v) for v in embedded["vectors"]],
                embed_tokens=tokens,
                embed_usd=cost_usd(tokens),
            )
        return str(article_id)

    article_id = await ctx.step.run("persist", persist)

    done = ArticleIngested(
        canonical_id=d.canonical_id,
        article_id=UUID(article_id),
        content_hash=art.content_hash,
        passages=len(chunks),
    )
    await ctx.step.send_event("emit", event(ARTICLE_INGESTED, done))
    return {"article_id": article_id, "passages": len(chunks)}
