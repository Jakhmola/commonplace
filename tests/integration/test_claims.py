"""Claims in Postgres: replays converge, edits keep history, and everything traces to a source."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import Claim, Passage
from commonplace.db.queries import (
    count_active_claims,
    find_matching_passage_for_quote,
    get_active_claims_for_article,
    get_source_provenance,
    persist_article_claims,
    store_passages,
)
from commonplace.schemas.ids import claim_id
from tests.integration.conftest import passages

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)


def _claim(
    article_id: uuid.UUID, version: str, quote: str, topic: str, kind: str = "opinion"
) -> dict[str, Any]:
    text = f"The author says: {quote}"
    return {
        "claim_id": str(claim_id(article_id, version, quote, text)),
        "article_id": str(article_id),
        "text": text,
        "quote_span": quote,
        "claim_type": kind,
        "confidence": 0.9,
        "topics": [{"slug": topic, "name": topic.title()}],
    }


async def _rows(session: AsyncSession, article_id: uuid.UUID) -> list[Claim]:
    return list(
        (await session.execute(select(Claim).where(Claim.article_id == article_id))).scalars()
    )


async def test_replaying_extraction_changes_nothing(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    claims = [
        _claim(article_id, "v1", "quokkazebra scales linearly", "quokkazebra"),
        _claim(article_id, "v1", "wombatokapi is slower", "wombatokapi", "measurement"),
    ]
    before = await count_active_claims(db)
    assert await persist_article_claims(db, article_id=article_id, claims=claims, now=NOW) == []
    assert await persist_article_claims(db, article_id=article_id, claims=claims, now=NOW) == []

    assert len(await _rows(db, article_id)) == 2
    assert await count_active_claims(db) == before + 2
    stored = {c.quote_span: c for c in await get_active_claims_for_article(db, article_id)}
    assert stored["wombatokapi is slower"].claim_type == "measurement"
    assert stored["wombatokapi is slower"].entities == [
        {"slug": "wombatokapi", "name": "Wombatokapi"}
    ]


async def test_an_edit_closes_old_claims_and_links_the_new_ones(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    old_a = _claim(article_id, "v1", "quokkazebra scales linearly", "quokkazebra")
    old_b = _claim(article_id, "v1", "wombatokapi is slower", "wombatokapi")
    await persist_article_claims(db, article_id=article_id, claims=[old_a, old_b], now=NOW)

    new_a = _claim(article_id, "v2", "quokkazebra scales sublinearly", "quokkazebra")
    new_c = _claim(article_id, "v2", "numbatibis is new", "numbatibis")
    later = datetime(2026, 11, 24, tzinfo=UTC)
    pairs = await persist_article_claims(
        db, article_id=article_id, claims=[new_a, new_c], now=later
    )

    # the new claim about the same topic supersedes the old one; the unrelated one has no link
    assert pairs == [
        {
            "new_claim_id": new_a["claim_id"],
            "old_claim_id": old_a["claim_id"],
            "at": later.isoformat(),
        }
    ]
    rows = {str(c.id): c for c in await _rows(db, article_id)}
    assert len(rows) == 4  # history is kept
    assert rows[old_a["claim_id"]].valid_to == later and rows[old_b["claim_id"]].valid_to == later
    assert rows[new_a["claim_id"]].valid_to is None
    assert str(rows[new_a["claim_id"]].supersedes_id) == old_a["claim_id"]
    assert rows[new_c["claim_id"]].supersedes_id is None

    # a replay of the new version must not erase the link it recorded the first time
    await persist_article_claims(db, article_id=article_id, claims=[new_a, new_c], now=later)
    await db.refresh(rows[new_a["claim_id"]])
    assert str(rows[new_a["claim_id"]].supersedes_id) == old_a["claim_id"]


async def test_a_claim_resolves_to_the_passage_holding_its_quote(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    assert await find_matching_passage_for_quote(db, article_id=article_id, quote_span="x") is None

    await store_passages(
        db,
        article_id=article_id,
        passages=passages(
            "an introduction about nothing in particular",
            "the quokkazebra framework scales\nlinearly with cores",
            "closing remarks",
        ),
    )
    found = await find_matching_passage_for_quote(
        db, article_id=article_id, quote_span="Quokkazebra framework   scales linearly"
    )
    assert found is not None and found["text"].startswith("the quokkazebra framework")
    assert found["title"] == "A seeded post"

    # a quote no passage contains (it straddled a chunk boundary) falls back to the first passage
    fallback = await find_matching_passage_for_quote(
        db, article_id=article_id, quote_span="text that is in no single passage"
    )
    assert fallback is not None and fallback["text"].startswith("an introduction")


async def test_provenance_for_a_claim_and_for_a_passage(
    db: AsyncSession, article_id: uuid.UUID
) -> None:
    claim = _claim(article_id, "v1", "quokkazebra scales linearly", "quokkazebra", "measurement")
    await persist_article_claims(db, article_id=article_id, claims=[claim], now=NOW)
    await store_passages(
        db, article_id=article_id, passages=passages("quokkazebra scales linearly")
    )
    passage_id = (
        await db.execute(select(Passage.id).where(Passage.article_id == article_id))
    ).scalar_one()

    of_claim = await get_source_provenance(db, uuid.UUID(claim["claim_id"]))
    assert of_claim is not None
    assert (of_claim["kind"], of_claim["claim_type"]) == ("claim", "measurement")
    assert of_claim["verbatim_quote"] == "quokkazebra scales linearly"
    assert of_claim["publication"]["slug"] == "integration-test"
    assert of_claim["article"]["title"] == "A seeded post"
    assert of_claim["topics"] == [{"slug": "quokkazebra", "name": "Quokkazebra"}]
    assert of_claim["validity"]["is_active"] is True and of_claim["validity"]["valid_to"] is None

    of_passage = await get_source_provenance(db, passage_id)
    assert of_passage is not None
    assert (of_passage["kind"], of_passage["chunk_index"]) == ("passage", 0)
    assert of_passage["verbatim_quote"] == "quokkazebra scales linearly"

    assert await get_source_provenance(db, uuid.uuid4()) is None
