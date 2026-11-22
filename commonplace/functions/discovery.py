"""cp/publication.added -> add-publication.

Reads the feed, takes the newest `discovery_limit` entries, and fans out one
cp/article.discovered per entry that is not stored yet. Sending the same
publication again therefore retries only the articles that failed before.
"""

import uuid

import inngest

from commonplace.config import settings
from commonplace.db.queries import create_job, get_existing_canonical_ids, upsert_publication
from commonplace.db.session import get_session
from commonplace.inngest_client import client
from commonplace.schemas.events import (
    ARTICLE_DISCOVERED,
    PUBLICATION_ADDED,
    ArticleDiscovered,
    PublicationAdded,
    event,
)
from commonplace.schemas.ids import canonical_id
from commonplace.sources.allowlist import (
    PublicationEntry,
    find_publication_for_url,
    load_publications,
)
from commonplace.sources.rss import fetch_feed, parse_feed


@client.create_function(
    fn_id="add-publication",
    trigger=inngest.TriggerEvent(event=PUBLICATION_ADDED),
    retries=3,
)
async def add_publication(ctx: inngest.Context) -> dict[str, int]:
    payload = PublicationAdded.model_validate(ctx.event.data)

    async def match_allowlist() -> dict[str, object]:
        pub = find_publication_for_url(payload.feed_url, load_publications())
        if pub is None:
            # Retrying cannot make an unlisted feed listed.
            raise inngest.NonRetriableError(
                f"Feed is outside the allowlist (publications.yaml): {payload.feed_url!r}"
            )
        return pub.model_dump()

    pub = PublicationEntry.model_validate(await ctx.step.run("match-allowlist", match_allowlist))

    async def fetch() -> str:
        return (await fetch_feed(pub.feed_url)).decode("utf-8", errors="replace")

    feed_text = await ctx.step.run("fetch-feed", fetch)

    async def save_publication() -> str:
        async with get_session() as session:
            publication_id = await upsert_publication(
                session, slug=pub.slug, name=pub.name, feed_url=pub.feed_url, homepage=pub.homepage
            )
            await session.commit()
        return str(publication_id)

    publication_id = uuid.UUID(await ctx.step.run("upsert-publication", save_publication))

    discovered = [
        ArticleDiscovered(
            canonical_id=canonical_id(pub.slug, entry.url),
            publication_id=publication_id,
            url=entry.url,
            title=entry.title,
            author=entry.author,
            published_at=entry.published_at,
            via="rss",
        )
        for entry in parse_feed(feed_text.encode())[: settings.discovery_limit]
    ]

    async def discover() -> list[str]:
        """Record the job, and return the ids that still need ingesting."""
        ids = [d.canonical_id for d in discovered]
        async with get_session() as session:
            await create_job(
                session,
                event_id=ctx.event.id,
                kind="discovery",
                canonical_ids=ids,
                publication_id=publication_id,
            )
            stored = await get_existing_canonical_ids(session, ids)
            await session.commit()
        return [i for i in ids if i not in stored]

    missing = set(await ctx.step.run("discover", discover))
    new = [d for d in discovered if d.canonical_id in missing]
    if new:
        # One batched send for the whole fan-out.
        await ctx.step.send_event("emit", [event(ARTICLE_DISCOVERED, d) for d in new])
    return {"count": len(discovered), "emitted": len(new)}
