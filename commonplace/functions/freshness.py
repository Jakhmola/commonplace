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
    get_unseen_canonical_ids,
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


async def check_feed(
    pub: PublicationEntry, etag: str | None, last_modified: str | None
) -> dict[str, Any]:
    """One conditional request to a feed, parsed. Reports a failure instead of raising.

    A feed that is down, throttling or serving garbage must not fail the run:
    the feeds after it would never be polled, and the next tick is five minutes
    away anyway.
    """
    try:
        response = await fetch_feed_conditional(
            pub.feed_url, etag=etag, last_modified=last_modified
        )
        pointers = feed_pointers(response.content) if response.content is not None else []
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "pointers": [],
            "etag": etag,
            "last_modified": last_modified,
        }
    return {
        "status": response.status,
        "error": None,
        "pointers": pointers,
        "etag": response.etag,
        "last_modified": response.last_modified,
    }


async def _poll(pub: PublicationEntry) -> dict[str, Any]:
    async with get_session() as session:
        publication_id = await upsert_publication(
            session, slug=pub.slug, name=pub.name, feed_url=pub.feed_url, homepage=pub.homepage
        )
        cursor = await get_cursor(session, publication_id)
        etag = cursor.etag if cursor else None
        last_modified = cursor.last_modified if cursor else None
        await session.commit()

    checked = await check_feed(pub, etag, last_modified)
    result = {
        "slug": pub.slug,
        "publication_id": str(publication_id),
        "status": checked["status"],
        "error": checked["error"],
        "etag": checked["etag"],
        "last_modified": checked["last_modified"],
        "new": [],
    }
    if checked["status"] == "error":
        return result

    async with get_session() as session:
        await record_poll(
            session, publication_id=publication_id, not_modified=checked["status"] == 304
        )
        if checked["status"] == 200:
            found = candidates(checked["pointers"], pub, publication_id, via="poll")
            unseen = set(
                await get_unseen_canonical_ids(
                    session, [c.canonical_id for c in found], skip_failed=True
                )
            )
            result["new"] = [c.model_dump(mode="json") for c in found if c.canonical_id in unseen]
            if not result["new"]:
                # Nothing to queue, so this version of the feed is dealt with.
                await save_validators(
                    session,
                    publication_id=publication_id,
                    etag=checked["etag"],
                    last_modified=checked["last_modified"],
                )
        await session.commit()
    return result


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

        summary = {"slug": pub.slug, "status": polled["status"], "new_articles": len(polled["new"])}
        if polled["error"]:
            summary["error"] = polled["error"]
        results.append(summary)

    return {"polled": len(results), "results": results}
