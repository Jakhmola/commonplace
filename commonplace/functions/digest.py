"""The weekly digest: what the library took in, as a short cited briefing.

`generate-weekly-digest` runs every Monday at 08:00 UTC and on request
(cp/digest.requested):

1. `collect-weekly-window`     the articles published in the window and their
                               most confident current claims
2. `synthesize-weekly-digest`  a model writes the briefing, citing claims by number
3. `verify-and-persist-digest` the citations are checked in code and the digest
                               is stored, one row per window
4. `emit`                      cp/digest.published

The digest is written from extracted claims, not from article text: every
claim already carries a verbatim quote that was checked against its article,
so each `[n]` in the briefing leads to something verifiable.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import inngest

from commonplace.agents.core import run_agent_step
from commonplace.agents.roles.researcher import digest_writer
from commonplace.citations import cited_numbers
from commonplace.config import settings
from commonplace.db.queries import (
    get_active_claims_for_articles,
    get_articles_in_window,
    upsert_digest,
)
from commonplace.db.session import get_session
from commonplace.inngest_client import client
from commonplace.schemas.events import (
    DIGEST_PUBLISHED,
    DIGEST_REQUESTED,
    DigestPublished,
    DigestRequested,
    event,
)

_DIGEST_NAMESPACE = uuid.UUID("8b1f6c0a-2d47-4e9b-a3c5-5f0e7d9126b4")
_NO_CLAIMS = "No claims were extracted from the articles in this window."
_SECTIONS = (
    ("measurement", "## What was measured"),
    ("opinion", "## What authors argued"),
    ("prediction", "## Predictions to watch"),
)


def digest_window(now: datetime, lookback_days: int) -> tuple[datetime, datetime]:
    """The `lookback_days` complete UTC days before today: [start, end).

    Whole days make the window the same at any hour of the day, so a digest
    written twice on one day is for the same window, and the Monday run covers
    exactly the Monday to Sunday before it.
    """
    end = datetime(now.year, now.month, now.day, tzinfo=UTC)
    return end - timedelta(days=lookback_days), end


def digest_id_for(start: datetime, end: datetime) -> uuid.UUID:
    """One id per window, so writing a window's digest again replaces it."""
    return uuid.uuid5(_DIGEST_NAMESPACE, f"{start.date()}:{end.date()}")


def digest_evidence(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The claims as the writer cites them: numbered from 1, each with its source."""
    fields = ("claim_id", "claim_text", "claim_type", "title", "author", "publication", "url")
    return [
        {"n": n, **{f: claim[f] for f in fields}, "topics": list(claim.get("topics") or [])}
        for n, claim in enumerate(claims, start=1)
    ]


def shared_topics(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topics that claims from more than one publication are about, widest first."""
    by_topic: dict[str, dict[str, Any]] = {}
    for item in evidence:
        for topic in item["topics"]:
            entry = by_topic.setdefault(topic, {"publications": set(), "claims": []})
            entry["publications"].add(item["publication"])
            entry["claims"].append(item["n"])
    shared = [
        {"topic": topic, "publications": sorted(e["publications"]), "claims": e["claims"]}
        for topic, e in by_topic.items()
        if len(e["publications"]) > 1
    ]
    return sorted(shared, key=lambda t: (-len(t["publications"]), t["topic"]))


def format_digest_input(
    title: str, evidence: list[dict[str, Any]], shared: list[dict[str, Any]]
) -> str:
    claims = "\n".join(
        f"[{e['n']}] ({e['claim_type']}) {e['claim_text']} - "
        f'{e["author"] or "Unknown"}, {e["publication"]}, "{e["title"]}"'
        for e in evidence
    )
    topics = "\n".join(
        f"{t['topic']}: {', '.join(t['publications'])} (claims {', '.join(map(str, t['claims']))})"
        for t in shared
    )
    return (
        f"{title}\n\nClaims:\n{claims}\n\n"
        f"Topics that more than one publication wrote about:\n{topics or '(none)'}"
    )


def plain_digest(evidence: list[dict[str, Any]]) -> str:
    """The digest without a model: the claims listed under their type, each cited."""
    if not evidence:
        return _NO_CLAIMS
    sections = []
    for claim_type, heading in _SECTIONS:
        lines = [
            f"- {e['claim_text']} ({e['author'] or e['publication']}, {e['title']}) [{e['n']}]"
            for e in evidence
            if e["claim_type"] == claim_type
        ]
        if lines:
            sections.append("\n".join([heading, "", *lines]))
    return "\n\n".join(sections)


def citations_in_bounds(markdown: str, evidence: list[dict[str, Any]]) -> bool:
    """Every `[n]` in the digest is the number of a claim it was given, and a
    digest that had claims to work from cites at least one.
    """
    numbers = cited_numbers(markdown)
    if not evidence:
        return not numbers
    return bool(numbers) and all(1 <= n <= len(evidence) for n in numbers)


def _title(start: datetime, end: datetime, fell_back: bool) -> str:
    last_day = end - timedelta(days=1)
    span = f"{start:%Y-%m-%d} to {last_day:%Y-%m-%d}"
    if fell_back:
        return f"Most recent articles (nothing was published {span})"
    return f"Week of {span}" if (end - start).days == 7 else f"Digest for {span}"


@client.create_function(
    fn_id="generate-weekly-digest",
    trigger=[
        inngest.TriggerCron(cron=settings.digest_cron),
        inngest.TriggerEvent(event=DIGEST_REQUESTED),
    ],
    retries=2,
)
async def generate_weekly_digest(ctx: inngest.Context) -> dict[str, object]:
    # A cron run carries no payload of ours: it gets the configured lookback.
    data = ctx.event.data if ctx.event.name == DIGEST_REQUESTED else {}
    given = {key: value for key, value in (data or {}).items() if value is not None}
    payload = DigestRequested.model_validate(
        {"lookback_days": settings.digest_lookback_days, **given}
    )

    async def collect() -> dict[str, Any]:
        # The window is fixed inside a step, so a replay on another day keeps it.
        start, end = digest_window(datetime.now(UTC), payload.lookback_days)
        async with get_session() as session:
            articles, fell_back = await get_articles_in_window(session, start, end)
            claims = await get_active_claims_for_articles(
                session, [uuid.UUID(a["article_id"]) for a in articles]
            )
        return {
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "fell_back": fell_back,
            "article_count": len(articles),
            "publications": sorted({a["publication"] for a in articles}),
            "claims": claims,
        }

    window = await ctx.step.run("collect-weekly-window", collect)
    start = datetime.fromisoformat(window["window_start"])
    end = datetime.fromisoformat(window["window_end"])
    title = _title(start, end, window["fell_back"])
    evidence = digest_evidence(window["claims"])
    shared = shared_topics(evidence)

    cost_usd = 0.0
    if evidence and settings.anthropic_api_key:
        written = await run_agent_step(
            ctx.step,
            digest_writer,
            format_digest_input(title, evidence, shared),
            "synthesize-weekly-digest",
        )
        markdown = str(written["output"] or "") or plain_digest(evidence)
        cost_usd = written["usage"]["est_usd"]
    else:
        markdown = plain_digest(evidence)

    numbers = cited_numbers(markdown)
    digest_id = digest_id_for(start, end)
    stats = {
        "article_count": window["article_count"],
        "claim_count": len(evidence),
        "publications": window["publications"],
        "shared_topics": shared,
        # True: nothing was published in the window, so the most recent articles were used.
        "fell_back_to_recent": window["fell_back"],
        # False: the digest cites a number it was not given, or cites nothing.
        "citations_valid": citations_in_bounds(markdown, evidence),
        "cost_usd": cost_usd,
    }

    async def persist() -> str:
        async with get_session() as session:
            await upsert_digest(
                session,
                digest_id=digest_id,
                window_start=start,
                window_end=end,
                title=title,
                summary_markdown=markdown,
                citations=[e for e in evidence if e["n"] in numbers],
                stats=stats,
            )
            await session.commit()
        return str(digest_id)

    await ctx.step.run("verify-and-persist-digest", persist)

    published = DigestPublished(
        digest_id=digest_id,
        window_start=start,
        window_end=end,
        article_count=window["article_count"],
        claim_count=len(evidence),
    )
    await ctx.step.send_event("emit", event(DIGEST_PUBLISHED, published))
    return {**published.model_dump(mode="json"), "citations_valid": stats["citations_valid"]}
