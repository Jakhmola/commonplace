"""Queue claim extraction for stored articles that have no claims yet.

    uv run python scripts/enrich_backlog.py            # every such article
    uv run python scripts/enrich_backlog.py --limit 3  # only the three oldest

New and edited articles are extracted automatically. This is for the articles
that were ingested before extraction existed, or whose extraction failed. It
sends cp/article.ingested for each one, which is what `extract-claims` listens
to; nothing is fetched or embedded again.
"""

import argparse
import asyncio

from commonplace.db.queries import get_articles_without_claims
from commonplace.db.session import engine, get_session
from commonplace.inngest_client import client
from commonplace.schemas.events import ARTICLE_INGESTED, ArticleIngested, event


async def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="queue at most this many")
    args = parser.parse_args()

    async with get_session() as session:
        articles = await get_articles_without_claims(session, args.limit)
    if articles:
        await client.send(
            [
                event(
                    ARTICLE_INGESTED,
                    ArticleIngested(
                        canonical_id=a.canonical_id,
                        article_id=a.id,
                        content_hash=a.content_hash,
                        passage_count=0,
                    ),
                )
                for a in articles
            ]
        )
    print(f"queued {len(articles)} articles for claim extraction")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
