"""cp/publication.added -> backfill-publication.

Takes every entry in the feed, drops the ones already stored, and fans out
one cp/article.discovered per new article. A feed only carries a
publication's recent posts (about 20), so this is "the whole feed", not the
whole archive.

The fan-out is cheap on purpose. The expensive work is in `ingest-article`,
where concurrency, throttle and priority decide when each article runs.
Sending the same publication again re-queues only what is still missing.
"""

import uuid
from typing import Any

import inngest

from commonplace.config import settings
from commonplace.db.queries import create_job, get_existing_canonical_ids, upsert_publication
from commonplace.db.session import get_session
from commonplace.inngest_client import client
from commonplace.schemas.events import (
    ARTICLE_DISCOVERED,
    PUBLICATION_ADDED,
    ArticleDiscovered,
    DiscoverySource,
    PublicationAdded,
)
from commonplace.schemas.ids import canonical_id
from commonplace.sources.allowlist import (
    PublicationEntry,
    find_publication_for_url,
    load_publications,
)
from commonplace.sources.rss import FeedEntry, fetch_feed, parse_feed


def feed_pointers(raw: bytes) -> list[dict[str, Any]]:
    """What the `fetch-feed` step hands on: title, link, author and date per entry.

    Never the raw feed. Feeds carry whole post bodies, and a step's output is
    capped at 4 MB: one feed here was already 2.9 MB.
    """
    return [entry.model_dump(mode="json") for entry in parse_feed(raw)]


def candidates(
    pointers: list[dict[str, Any]],
    pub: PublicationEntry,
    publication_id: uuid.UUID,
    via: DiscoverySource = "backfill",
) -> list[ArticleDiscovered]:
    """One candidate per identifiable post.

    An entry whose link has no post slug (a feed item pointing at the homepage)
    is skipped rather than failing the whole backfill, and when two links end
    in the same slug the first one wins, so the job's total stays reachable.
    """
    found: dict[str, ArticleDiscovered] = {}
    for pointer in pointers:
        entry = FeedEntry.model_validate(pointer)
        try:
            cid = canonical_id(pub.slug, entry.url)
        except ValueError:
            continue
        found.setdefault(
            cid,
            ArticleDiscovered(
                canonical_id=cid,
                publication_id=publication_id,
                url=entry.url,
                title=entry.title,
                author=entry.author,
                published_at=entry.published_at,
                via=via,
            ),
        )
    return list(found.values())


@client.create_function(
    fn_id="backfill-publication",
    trigger=inngest.TriggerEvent(event=PUBLICATION_ADDED),
    retries=3,
)
async def backfill_publication(ctx: inngest.Context) -> dict[str, int]:
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

    async def fetch() -> list[dict[str, Any]]:
        return feed_pointers(await fetch_feed(pub.feed_url))

    pointers = await ctx.step.run("fetch-feed", fetch)

    async def save_publication() -> str:
        async with get_session() as session:
            publication_id = await upsert_publication(
                session, slug=pub.slug, name=pub.name, feed_url=pub.feed_url, homepage=pub.homepage
            )
            await session.commit()
        return str(publication_id)

    publication_id = uuid.UUID(await ctx.step.run("upsert-publication", save_publication))

    async def find_new_articles() -> list[dict[str, Any]]:
        found = candidates(pointers, pub, publication_id)
        async with get_session() as session:
            stored = await get_existing_canonical_ids(session, [c.canonical_id for c in found])
            new = [c for c in found if c.canonical_id not in stored]
            # The job row is written before anything is queued, so progress can
            # be read from the moment the first article lands.
            await create_job(
                session,
                event_id=ctx.event.id,
                kind="backfill",
                canonical_ids=[c.canonical_id for c in new],
                publication_id=publication_id,
            )
            await session.commit()
        return [c.model_dump(mode="json") for c in new]

    new_articles = await ctx.step.run("find-new-articles", find_new_articles)

    batch = settings.backfill_batch_size
    for n, start in enumerate(range(0, len(new_articles), batch)):
        await ctx.step.send_event(
            f"emit-{n}",
            [
                inngest.Event(name=ARTICLE_DISCOVERED, data=data)
                for data in new_articles[start : start + batch]
            ],
        )
    return {"discovered": len(new_articles)}
