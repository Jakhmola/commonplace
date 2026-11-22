"""Replay a backfill and show that it writes nothing new.

    uv run python scripts/replay_backfill.py --verify              # print the counts
    uv run python scripts/replay_backfill.py --all                 # replay every publication
    uv run python scripts/replay_backfill.py --slug latent-space   # replay one

`--all` and `--slug` re-send cp/publication.added, exactly as POST /publications
does. Run `--verify` before, and again once Inngest has finished: if ingestion
is idempotent the publication, article and passage counts are identical, and so
is the number of embedding calls in `costs`, because stored articles never
reach the `embed` step.
"""

import argparse
import asyncio
import sys

from sqlalchemy import func, select

from commonplace.config import settings
from commonplace.db.models import Article, Cost, Passage, Publication
from commonplace.db.session import engine, get_session
from commonplace.inngest_client import client
from commonplace.schemas.events import PUBLICATION_ADDED, PublicationAdded, event
from commonplace.sources.allowlist import load_publications


async def counts() -> dict[str, int]:
    queries = {
        "publications": select(func.count(Publication.id)),
        "articles": select(func.count(Article.id)),
        "passages": select(func.count(Passage.id)),
        "embedding_calls": select(func.count(Cost.id)).where(
            Cost.model == settings.embedding_model
        ),
    }
    async with get_session() as session:
        return {name: (await session.execute(q)).scalar_one() for name, q in queries.items()}


async def replay(slug: str | None) -> None:
    publications = [p for p in load_publications() if slug in (None, p.slug)]
    if not publications:
        sys.exit(f"No publication with slug {slug!r} in publications.yaml")
    ids = await client.send(
        [event(PUBLICATION_ADDED, PublicationAdded(feed_url=p.feed_url)) for p in publications]
    )
    for publication, event_id in zip(publications, ids, strict=True):
        print(f"{publication.slug}: {settings.inngest_base_url}/event/{event_id}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true", help="print the current counts")
    group.add_argument("--all", action="store_true", help="replay every publication")
    group.add_argument("--slug", help="replay a single publication")
    args = parser.parse_args()

    if args.verify:
        print("  ".join(f"{name}={value}" for name, value in (await counts()).items()))
    else:
        await replay(args.slug)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
