"""poll-feeds: keep the corpus fresh without re-reading feeds that have not changed.

Runs every five minutes and on demand (cp/feed.poll). Each feed gets one
conditional request. A 304 means no body, no parsing and no downstream work.
A 200 is parsed, and only articles that are not stored yet are queued.
"""

import uuid
from typing import Any

import inngest

from commonplace.db.queries import (
    get_cursor,
    get_existing_canonical_ids,
    record_poll,
    save_validators,
    upsert_publication,
)
from commonplace.db.session import get_session
from commonplace.functions.backfill import candidates, feed_pointers
from commonplace.inngest_client import client
from commonplace.schemas.events import ARTICLE_DISCOVERED, FEED_POLL, FeedPoll
from commonplace.sources.allowlist import PublicationEntry, load_publications
from commonplace.sources.rss import fetch_feed_conditional


def publications_to_poll(
    publications: list[PublicationEntry], only: str | None
) -> list[PublicationEntry]:
    """Every allowlisted feed, or just the one whose slug was asked for."""
    if only is None:
        return publications
    chosen = [p for p in publications if p.slug == only]
    if not chosen:
        raise inngest.NonRetriableError(f"Publication {only!r} is not in publications.yaml")
    return chosen


async def _poll(pub: PublicationEntry) -> dict[str, Any]:
    async with get_session() as session:
        publication_id = await upsert_publication(
            session, slug=pub.slug, name=pub.name, feed_url=pub.feed_url, homepage=pub.homepage
        )
        cursor = await get_cursor(session, publication_id)
        etag = cursor.etag if cursor else None
        last_modified = cursor.last_modified if cursor else None
        await session.commit()

    response = await fetch_feed_conditional(pub.feed_url, etag=etag, last_modified=last_modified)

    new: list[dict[str, Any]] = []
    async with get_session() as session:
        await record_poll(
            session, publication_id=publication_id, not_modified=response.status == 304
        )
        if response.content is not None:
            found = candidates(feed_pointers(response.content), pub, publication_id, via="poll")
            stored = await get_existing_canonical_ids(session, [c.canonical_id for c in found])
            new = [c.model_dump(mode="json") for c in found if c.canonical_id not in stored]
            if not new:
                # Nothing to queue, so this version of the feed is dealt with.
                await save_validators(
                    session,
                    publication_id=publication_id,
                    etag=response.etag,
                    last_modified=response.last_modified,
                )
        await session.commit()

    return {
        "slug": pub.slug,
        "publication_id": str(publication_id),
        "status": response.status,
        "etag": response.etag,
        "last_modified": response.last_modified,
        "new": new,
    }


@client.create_function(
    fn_id="poll-feeds",
    trigger=[inngest.TriggerCron(cron="*/5 * * * *"), inngest.TriggerEvent(event=FEED_POLL)],
    retries=3,
)
async def poll_feeds(ctx: inngest.Context) -> dict[str, object]:
    only = FeedPoll.model_validate(ctx.event.data or {}).publication
    results = []

    for pub in publications_to_poll(load_publications(), only):

        async def poll(pub: PublicationEntry = pub) -> dict[str, Any]:
            return await _poll(pub)

        polled = await ctx.step.run(f"poll-{pub.slug}", poll)

        if polled["new"]:
            await ctx.step.send_event(
                f"emit-{pub.slug}",
                [inngest.Event(name=ARTICLE_DISCOVERED, data=data) for data in polled["new"]],
            )

            # The new validators are saved only once the articles are queued. If
            # this run dies in between, the next poll sees the same 200 again
            # instead of a 304 that would hide the articles for good.
            async def remember(polled: dict[str, Any] = polled) -> None:
                async with get_session() as session:
                    await save_validators(
                        session,
                        publication_id=uuid.UUID(polled["publication_id"]),
                        etag=polled["etag"],
                        last_modified=polled["last_modified"],
                    )
                    await session.commit()

            await ctx.step.run(f"remember-{pub.slug}", remember)

        results.append(
            {"slug": pub.slug, "status": polled["status"], "new_articles": len(polled["new"])}
        )

    return {"polled": len(results), "results": results}
