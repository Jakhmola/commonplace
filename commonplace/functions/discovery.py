"""cp/publication.added -> add-publication.

Reads the feed, takes the newest `discovery_limit` entries, and fans out one
cp/article.discovered per entry that is not stored yet. Sending the same
publication again therefore retries only the articles that failed before.
"""

from uuid import UUID

import inngest

from commonplace import db
from commonplace.config import settings
from commonplace.events import (
    ARTICLE_DISCOVERED,
    PUBLICATION_ADDED,
    ArticleDiscovered,
    PublicationAdded,
    event,
)
from commonplace.inngest_client import client
from commonplace.sources.allowlist import find_by_feed, load_publications
from commonplace.sources.fetch import fetch
from commonplace.sources.rss import FeedEntry, parse_feed
from commonplace.sources.urls import canonical_id


@client.create_function(
    fn_id="add-publication",
    trigger=inngest.TriggerEvent(event=PUBLICATION_ADDED),
    retries=3,
)
async def add_publication(ctx: inngest.Context) -> dict[str, int]:
    payload = PublicationAdded.model_validate(ctx.event.data)

    pub = find_by_feed(payload.feed_url, load_publications())
    if pub is None:
        raise inngest.NonRetriableError(f"{payload.feed_url!r} is not in publications.yaml")

    async def fetch_feed() -> list[dict[str, object]]:
        _, raw = await fetch(pub.feed_url)
        return [e.model_dump(mode="json") for e in parse_feed(raw)[: settings.discovery_limit]]

    entries = [FeedEntry.model_validate(e) for e in await ctx.step.run("fetch-feed", fetch_feed)]

    async def save_publication() -> str:
        async with db.tx() as conn:
            return str(await db.upsert_publication(conn, pub))

    publication_id = UUID(await ctx.step.run("save-publication", save_publication))

    discovered = [
        ArticleDiscovered(
            canonical_id=canonical_id(e.url),
            publication_id=publication_id,
            url=e.url,
            title=e.title,
            author=e.author,
            published_at=e.published_at,
            via="rss",
        )
        for e in entries
    ]

    async def find_new() -> list[str]:
        ids = [d.canonical_id for d in discovered]
        async with db.tx() as conn:
            await db.create_job(conn, ctx.event.id, publication_id, ids)
            stored = await db.stored_canonical_ids(conn, ids)
        return [i for i in ids if i not in stored]

    new_ids = set(await ctx.step.run("find-new", find_new))
    new = [d for d in discovered if d.canonical_id in new_ids]
    if new:
        await ctx.step.send_event("fan-out", [event(ARTICLE_DISCOVERED, d) for d in new])
    return {"entries": len(discovered), "queued": len(new)}
