"""Every query in the project. Functions and endpoints call these; they never build SQL."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Row, Select, func, select, true, update
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.db.models import (
    Article,
    Claim,
    Cost,
    Cursor,
    EvalResult,
    EvalRun,
    Failure,
    Job,
    Passage,
    Publication,
    Snapshot,
)
from commonplace.sources.quote_validator import fold

# --- publications ---


async def upsert_publication(
    session: AsyncSession, *, slug: str, name: str, feed_url: str, homepage: str
) -> uuid.UUID:
    fields = {"name": name, "feed_url": feed_url, "homepage": homepage}
    stmt = (
        insert(Publication)
        .values(id=uuid.uuid4(), slug=slug, **fields)
        .on_conflict_do_update(index_elements=[Publication.slug], set_=fields)
        .returning(Publication.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def get_publication_by_slug(session: AsyncSession, slug: str) -> Publication | None:
    return (
        await session.execute(select(Publication).where(Publication.slug == slug))
    ).scalar_one_or_none()


async def list_publications(session: AsyncSession) -> list[Publication]:
    return list((await session.execute(select(Publication).order_by(Publication.slug))).scalars())


async def list_publications_with_stats(session: AsyncSession) -> list[dict[str, Any]]:
    """Every publication with its article count and polling state."""
    stmt = (
        select(Publication, func.count(Article.id), Cursor)
        .outerjoin(Article, Article.publication_id == Publication.id)
        .outerjoin(Cursor, Cursor.publication_id == Publication.id)
        .group_by(Publication.id, Cursor.publication_id)
        .order_by(Publication.slug)
    )
    return [
        {
            "id": str(pub.id),
            "slug": pub.slug,
            "name": pub.name,
            "feed_url": pub.feed_url,
            "homepage": pub.homepage,
            "active": pub.active,
            "article_count": article_count,
            "last_checked_at": (
                cursor.last_checked_at.isoformat() if cursor and cursor.last_checked_at else None
            ),
            "poll_count": cursor.poll_count if cursor else 0,
            "not_modified_count": cursor.not_modified_count if cursor else 0,
        }
        for pub, article_count, cursor in (await session.execute(stmt)).all()
    ]


# --- polling cursors ---


async def get_cursor(session: AsyncSession, publication_id: uuid.UUID) -> Cursor | None:
    return await session.get(Cursor, publication_id)


async def record_poll(
    session: AsyncSession, *, publication_id: uuid.UUID, not_modified: bool
) -> None:
    """Count one poll of a feed, and whether it found the feed unchanged."""
    unchanged = 1 if not_modified else 0
    stmt = (
        insert(Cursor)
        .values(
            publication_id=publication_id,
            last_checked_at=func.now(),
            poll_count=1,
            not_modified_count=unchanged,
        )
        .on_conflict_do_update(
            index_elements=[Cursor.publication_id],
            set_={
                "last_checked_at": func.now(),
                "poll_count": Cursor.poll_count + 1,
                "not_modified_count": Cursor.not_modified_count + unchanged,
            },
        )
    )
    await session.execute(stmt)


async def save_validators(
    session: AsyncSession,
    *,
    publication_id: uuid.UUID,
    etag: str | None,
    last_modified: str | None,
) -> None:
    """Remember the feed version we have fully dealt with. The cursor row must exist."""
    cursor = await session.get(Cursor, publication_id)
    if cursor is not None:
        cursor.etag, cursor.last_modified = etag, last_modified


# --- articles and passages ---


async def get_existing_canonical_ids(session: AsyncSession, canonical_ids: list[str]) -> set[str]:
    stmt = select(Article.canonical_id).where(Article.canonical_id.in_(canonical_ids))
    return set((await session.execute(stmt)).scalars())


async def upsert_article(
    session: AsyncSession,
    *,
    canonical_id: str,
    publication_id: uuid.UUID,
    url: str,
    title: str,
    author: str | None,
    published_at: datetime,
    content_hash: str,
    text: str,
) -> uuid.UUID:
    """Keyed on `canonical_id`, the dedupe key shared by every ingestion path."""
    fields = {
        "url": url,
        "title": title,
        "author": author,
        "published_at": published_at,
        "content_hash": content_hash,
        "text": text,
    }
    stmt = (
        insert(Article)
        .values(id=uuid.uuid4(), canonical_id=canonical_id, publication_id=publication_id, **fields)
        .on_conflict_do_update(
            index_elements=[Article.canonical_id], set_={**fields, "updated_at": func.now()}
        )
        .returning(Article.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def get_article_by_canonical_id(session: AsyncSession, canonical_id: str) -> Article | None:
    return (
        await session.execute(select(Article).where(Article.canonical_id == canonical_id))
    ).scalar_one_or_none()


async def count_current_passages(session: AsyncSession, article_id: uuid.UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(Passage)
        .where(Passage.article_id == article_id, Passage.valid_to.is_(None))
    )
    return (await session.execute(stmt)).scalar_one()


async def store_passages(
    session: AsyncSession, *, article_id: uuid.UUID, passages: list[dict[str, Any]]
) -> int:
    """Make `passages` the article's current version.

    Whatever was current is closed with `valid_to = now()`, never deleted or
    overwritten, so the history of an edited article stays queryable. For a
    new article there is nothing to close.
    """
    await session.execute(
        update(Passage)
        .where(Passage.article_id == article_id, Passage.valid_to.is_(None))
        .values(valid_to=func.now())
    )
    if passages:
        await session.execute(
            insert(Passage).values(
                [
                    {
                        "id": uuid.uuid4(),
                        "article_id": article_id,
                        "chunk_index": p["chunk_index"],
                        "text": p["text"],
                        "embedding": p["embedding"],
                    }
                    for p in passages
                ]
            )
        )
    return len(passages)


async def count_articles(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(Article))).scalar_one()


async def count_passages(session: AsyncSession) -> int:
    """Passages that are current, which is what a search can return."""
    stmt = select(func.count()).select_from(Passage).where(Passage.valid_to.is_(None))
    return (await session.execute(stmt)).scalar_one()


async def count_articles_for_publication(session: AsyncSession, publication_id: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(Article).where(Article.publication_id == publication_id)
    return (await session.execute(stmt)).scalar_one()


# --- jobs ---


async def create_job(
    session: AsyncSession,
    *,
    event_id: str,
    kind: str,
    canonical_ids: list[str],
    publication_id: uuid.UUID | None = None,
) -> None:
    stmt = (
        insert(Job)
        .values(
            event_id=event_id,
            kind=kind,
            # A replay that finds nothing new is finished the moment it is recorded.
            status="running" if canonical_ids else "completed",
            total=len(canonical_ids),
            done=0,
            publication_id=publication_id,
            canonical_ids=canonical_ids,
        )
        .on_conflict_do_nothing(index_elements=[Job.event_id])
    )
    await session.execute(stmt)


async def get_job(session: AsyncSession, event_id: str) -> Job | None:
    return await session.get(Job, event_id)


async def get_job_status(session: AsyncSession, event_id: str) -> dict[str, Any] | None:
    """Progress counted from what actually exists, then written back.

    `ingest-article` never learns which job asked for it, so nothing increments
    a counter. `done` is how many of the job's expected articles are stored, and
    `failed` how many of the rest have a recorded failure. A job is settled once
    every expected article is one or the other. The caller commits to keep the
    repaired `done`/`status`.
    """
    job = await get_job(session, event_id)
    if job is None:
        return None
    stored = await get_existing_canonical_ids(session, job.canonical_ids)
    missing = [cid for cid in job.canonical_ids if cid not in stored]
    errors = await get_failures_for(session, missing, since=job.created_at)

    job.done = len(stored)
    if job.done >= job.total:
        job.status = "completed"
    elif job.done + len(errors) >= job.total:
        job.status = "failed"
    else:
        job.status = "running"
    return {
        "event_id": job.event_id,
        "kind": job.kind,
        "status": job.status,
        "total": job.total,
        "done": job.done,
        "failed": len(errors),
        "errors": list(errors.values()),
    }


# --- failures ---


async def record_failure(
    session: AsyncSession,
    *,
    function_id: str,
    run_id: str,
    error: str,
    event_data: dict[str, Any] | None = None,
) -> uuid.UUID:
    failure = Failure(
        id=uuid.uuid4(), function_id=function_id, run_id=run_id, error=error, event_data=event_data
    )
    session.add(failure)
    await session.flush()
    return failure.id


async def get_failures_for(
    session: AsyncSession, canonical_ids: list[str], since: datetime | None = None
) -> dict[str, str]:
    """canonical id -> latest error, for articles whose ingest run failed for good.

    `since` limits this to failures recorded from that moment on, so a job is
    not blamed for a failure that happened before it existed.
    """
    if not canonical_ids:
        return {}
    failed_id = Failure.event_data["data"]["canonical_id"].astext
    stmt = select(failed_id, Failure.error).where(failed_id.in_(canonical_ids))
    if since is not None:
        stmt = stmt.where(Failure.created_at >= since)
    rows = (await session.execute(stmt.order_by(Failure.created_at))).all()
    return {cid: error for cid, error in rows}


async def get_unseen_canonical_ids(
    session: AsyncSession, canonical_ids: list[str], *, skip_failed: bool
) -> list[str]:
    """The ids, in order, that are not stored yet.

    With `skip_failed`, ids whose ingest already failed for good are left out
    too: a poller that kept re-queueing an article that can never be ingested
    would retry it on every poll, forever. An explicit backfill still retries it.
    """
    stored = await get_existing_canonical_ids(session, canonical_ids)
    unseen = [cid for cid in canonical_ids if cid not in stored]
    if skip_failed:
        failed = await get_failures_for(session, unseen)
        unseen = [cid for cid in unseen if cid not in failed]
    return unseen


# --- costs ---


async def insert_cost(
    session: AsyncSession,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    est_usd: float,
    article_id: uuid.UUID | None = None,
) -> None:
    session.add(
        Cost(
            article_id=article_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            est_usd=est_usd,
        )
    )


# --- search ---


def _passage_hits(score: Any) -> Select[Any]:
    """Current passages joined to their article, with `score` as `rank` (higher is better)."""
    return (
        select(
            Passage.id,
            Passage.text,
            Article.title,
            Article.author,
            Article.url,
            Article.published_at,
            Article.canonical_id,
            score.label("rank"),
        )
        .join(Article, Article.id == Passage.article_id)
        .where(Passage.valid_to.is_(None))
    )


def _hit(row: Row[Any]) -> dict[str, Any]:
    return {
        "passage_id": str(row.id),
        "text": row.text,
        "title": row.title,
        "author": row.author,
        "url": row.url,
        "published_at": row.published_at.isoformat(),
        "canonical_id": row.canonical_id,
        "rank": float(row.rank),
    }


# Each term of the query that a passage contains adds that term's rarity (inverse
# document frequency, in BM25's form) to the passage's score. A passage matches
# if it has any of the terms, so a question-shaped query still finds something,
# and a rare term outweighs any number of common ones.
# ponytail: term frequency and passage length are ignored (passages are all about
# the same size); use a real BM25 index if ranking within one term starts to matter.
_KEYWORD_SEARCH = sql_text("""
    WITH corpus AS (
        SELECT count(*)::float AS total FROM passages WHERE valid_to IS NULL
    ), terms AS (
        SELECT quote_literal(lexeme)::tsquery AS term
        FROM unnest(tsvector_to_array(to_tsvector('english', :query))) AS lexeme
    ), weighted AS (
        SELECT term, ln(1 + (total - df + 0.5) / (df + 0.5)) AS idf
        FROM corpus, terms, LATERAL (
            SELECT count(*) AS df FROM passages WHERE valid_to IS NULL AND tsv @@ term
        ) AS frequency
        WHERE df > 0
    )
    SELECT p.id, p.text, a.title, a.author, a.url, a.published_at, a.canonical_id,
           sum(w.idf) AS rank
    FROM weighted w
    JOIN passages p ON p.valid_to IS NULL AND p.tsv @@ w.term
    JOIN articles a ON a.id = p.article_id
    GROUP BY p.id, a.id
    ORDER BY rank DESC, p.id
    LIMIT :limit
""")


async def search_passages_bm25(
    session: AsyncSession, query: str, limit: int
) -> list[dict[str, Any]]:
    """Keyword search over the generated `tsv` column (GIN index).

    A query with double quotes is taken literally: phrases must match exactly
    and every term is required. Anything else is ranked by term rarity.
    """
    if '"' in query:
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank(Passage.tsv, tsquery)
        stmt = (
            _passage_hits(rank)
            .where(Passage.tsv.op("@@")(tsquery))
            .order_by(rank.desc(), Passage.id)
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
    else:
        rows = (await session.execute(_KEYWORD_SEARCH, {"query": query, "limit": limit})).all()
    return [_hit(row) for row in rows]


async def search_passages_dense(
    session: AsyncSession, query_embedding: list[float], limit: int
) -> list[dict[str, Any]]:
    """Nearest passages by cosine distance (HNSW index). `rank` is cosine similarity."""
    distance = Passage.embedding.cosine_distance(query_embedding)
    stmt = (
        _passage_hits(1 - distance)
        .where(Passage.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(limit)
    )
    return [_hit(row) for row in (await session.execute(stmt)).all()]


# --- snapshots ---


async def save_snapshot(
    session: AsyncSession,
    *,
    snapshot_id: str,
    question: str,
    answer_text: str,
    citations: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    retriever_variant: str,
    as_of: datetime | None,
    latency_ms: int,
) -> None:
    session.add(
        Snapshot(
            snapshot_id=snapshot_id,
            question=question,
            answer_text=answer_text,
            citations=citations,
            retrieved=retrieved,
            retriever_variant=retriever_variant,
            as_of=as_of,
            latency_ms=latency_ms,
        )
    )


async def get_snapshot(session: AsyncSession, snapshot_id: str) -> Snapshot | None:
    return await session.get(Snapshot, snapshot_id)


# --- claims ---


async def get_article_with_publication(
    session: AsyncSession, article_id: uuid.UUID
) -> tuple[Article, Publication] | None:
    stmt = (
        select(Article, Publication)
        .join(Publication, Publication.id == Article.publication_id)
        .where(Article.id == article_id)
    )
    row = (await session.execute(stmt)).first()
    return (row[0], row[1]) if row else None


async def count_active_claims(session: AsyncSession) -> int:
    stmt = select(func.count()).select_from(Claim).where(Claim.valid_to.is_(None))
    return (await session.execute(stmt)).scalar_one()


async def get_active_claims_for_article(
    session: AsyncSession, article_id: uuid.UUID
) -> list[Claim]:
    stmt = select(Claim).where(Claim.article_id == article_id, Claim.valid_to.is_(None))
    return list((await session.execute(stmt)).scalars())


async def get_articles_without_claims(
    session: AsyncSession, limit: int | None = None
) -> list[Article]:
    """Stored articles that have no current claims, oldest first."""
    has_claims = select(Claim.id).where(Claim.article_id == Article.id, Claim.valid_to.is_(None))
    stmt = select(Article).where(~has_claims.exists()).order_by(Article.published_at).limit(limit)
    return list((await session.execute(stmt)).scalars())


async def persist_article_claims(
    session: AsyncSession, *, article_id: uuid.UUID, claims: list[dict[str, Any]], now: datetime
) -> list[dict[str, str]]:
    """Make `claims` the article's current claims.

    Current claims that are not in the new set belong to an older version of
    the article and are closed. The new set is upserted by id, so a replay
    writes nothing new. Returns `[{new_claim_id, old_claim_id, at}]`: a new
    claim supersedes a closed one when they share a topic.
    """
    incoming = {uuid.UUID(c["claim_id"]) for c in claims}
    stale = [
        old
        for old in await get_active_claims_for_article(session, article_id)
        if old.id not in incoming
    ]
    if stale:
        await session.execute(
            update(Claim).where(Claim.id.in_([old.id for old in stale])).values(valid_to=now)
        )
    closed_by_topic = {entity["slug"]: old.id for old in stale for entity in old.entities}

    pairs: list[dict[str, str]] = []
    for claim in claims:
        cid = uuid.UUID(claim["claim_id"])
        topics = [{"slug": t["slug"], "name": t["name"]} for t in claim["topics"]]
        supersedes = next(
            (closed_by_topic[t["slug"]] for t in topics if t["slug"] in closed_by_topic), None
        )
        if supersedes is not None:
            pairs.append(
                {"new_claim_id": str(cid), "old_claim_id": str(supersedes), "at": now.isoformat()}
            )
        fields = {
            "text": claim["text"],
            "quote_span": claim["quote_span"],
            "claim_type": claim["claim_type"],
            "entities": topics,
            "confidence": claim.get("confidence", 0.9),
        }
        stmt = insert(Claim).values(
            id=cid, article_id=article_id, valid_from=now, supersedes_id=supersedes, **fields
        )
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[Claim.id],
                set_={
                    **fields,
                    "valid_to": None,
                    # On a replay nothing is stale any more, so there is no link to
                    # compute: keep the one recorded the first time.
                    "supersedes_id": func.coalesce(
                        stmt.excluded.supersedes_id, Claim.supersedes_id
                    ),
                },
            )
        )
    return pairs


async def find_matching_passage_for_quote(
    session: AsyncSession, *, article_id: uuid.UUID, quote_span: str
) -> dict[str, Any] | None:
    """The article's current passage that contains `quote_span`.

    Passages are cut at about 1000 characters, so a quote can straddle two of
    them: then the passage holding the start of the quote is used, and failing
    that the article's first passage. None only if the article has no passages.
    """
    stmt = (
        _passage_hits(Passage.chunk_index)
        .where(Passage.article_id == article_id)
        .order_by(Passage.chunk_index)
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        return None

    # Compared the way the quote guardrail compares: a quote that differs from
    # the passage only in typography must still land on its passage.
    quote = fold(quote_span).lower()
    texts = [fold(row.text).lower() for row in rows]
    for needle in (quote, quote[:60]):
        for row, haystack in zip(rows, texts, strict=True):
            if needle in haystack:
                return _hit(row) | {"rank": 1.0}
    return _hit(rows[0]) | {"rank": 1.0}


def _article_source(article: Article, publication: Publication) -> dict[str, Any]:
    return {
        "article": {
            "article_id": str(article.id),
            "canonical_id": article.canonical_id,
            "title": article.title,
            "url": article.url,
            "published_at": article.published_at.isoformat(),
            "content_hash": article.content_hash,
        },
        "publication": {
            "slug": publication.slug,
            "name": publication.name,
            "homepage": publication.homepage,
        },
        "author": article.author or "Unknown",
    }


def _validity(row: Claim | Passage, supersedes_id: uuid.UUID | None = None) -> dict[str, Any]:
    return {
        "valid_from": row.valid_from.isoformat(),
        "valid_to": row.valid_to.isoformat() if row.valid_to else None,
        "is_active": row.valid_to is None,
        "supersedes_id": str(supersedes_id) if supersedes_id else None,
    }


async def get_source_provenance(
    session: AsyncSession, target_id: uuid.UUID
) -> dict[str, Any] | None:
    """Where a claim or a passage comes from: article, publication, author, exact text, validity."""
    sources = (Article, Publication)
    joins = (
        (Article, Article.id == Claim.article_id),
        (Publication, Publication.id == Article.publication_id),
    )
    claim_stmt = (
        select(Claim, *sources).join(*joins[0]).join(*joins[1]).where(Claim.id == target_id)
    )
    if row := (await session.execute(claim_stmt)).first():
        claim, article, publication = row
        return {
            "kind": "claim",
            "id": str(claim.id),
            **_article_source(article, publication),
            "verbatim_quote": claim.quote_span,
            "claim_text": claim.text,
            "claim_type": claim.claim_type,
            "confidence": float(claim.confidence),
            "topics": list(claim.entities),
            "validity": _validity(claim, claim.supersedes_id),
        }

    passage_stmt = (
        select(Passage, *sources)
        .join(Article, Article.id == Passage.article_id)
        .join(*joins[1])
        .where(Passage.id == target_id)
    )
    if found := (await session.execute(passage_stmt)).first():
        passage, article, publication = found
        return {
            "kind": "passage",
            "id": str(passage.id),
            "chunk_index": passage.chunk_index,
            **_article_source(article, publication),
            "verbatim_quote": passage.text,
            "validity": _validity(passage),
        }
    return None


# --- evaluation ---


def _uuids(values: list[str]) -> list[uuid.UUID]:
    """The values that are UUIDs; anything else cannot be a stored id and is dropped."""
    parsed = []
    for value in values:
        try:
            parsed.append(uuid.UUID(str(value)))
        except ValueError:
            continue
    return parsed


async def get_superseded_ids(
    session: AsyncSession, *, passage_ids: list[str], claim_ids: list[str]
) -> set[str]:
    """Which of these passages and claims are no longer current."""
    superseded: set[str] = set()
    for model, ids in ((Passage, _uuids(passage_ids)), (Claim, _uuids(claim_ids))):
        if ids:
            stmt = select(model.id).where(model.id.in_(ids), model.valid_to.is_not(None))
            superseded |= {str(found) for found in (await session.execute(stmt)).scalars()}
    return superseded


async def upsert_eval_result(
    session: AsyncSession,
    *,
    snapshot_id: str,
    deterministic_passed: bool,
    checks: dict[str, bool],
    faithfulness_verdict: str,
    faithfulness_critique: str,
    disagreement_verdict: str,
    disagreement_critique: str,
    failure_codes: list[str],
    composite_score: float,
) -> uuid.UUID:
    """Record how a snapshot scored, replacing any earlier score, and put the
    composite score on the snapshot as its confidence.
    """
    fields = {
        "deterministic_passed": deterministic_passed,
        "checks": checks,
        "faithfulness_verdict": faithfulness_verdict,
        "faithfulness_critique": faithfulness_critique,
        "disagreement_verdict": disagreement_verdict,
        "disagreement_critique": disagreement_critique,
        "failure_codes": failure_codes,
        "composite_score": composite_score,
        # the moment of evaluation, not of the transaction's start
        "created_at": func.clock_timestamp(),
    }
    stmt = (
        insert(EvalResult)
        .values(id=uuid.uuid4(), snapshot_id=snapshot_id, **fields)
        .on_conflict_do_update(index_elements=[EvalResult.snapshot_id], set_=fields)
        .returning(EvalResult.id)
    )
    eval_id = (await session.execute(stmt)).scalar_one()
    await session.execute(
        update(Snapshot)
        .where(Snapshot.snapshot_id == snapshot_id)
        .values(confidence=composite_score)
    )
    return eval_id


async def upsert_eval_run(
    session: AsyncSession,
    *,
    run_id: str,
    suite_name: str,
    variants_summary: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> str:
    fields = {
        "suite_name": suite_name,
        "variants_summary": variants_summary,
        "case_results": case_results,
        "created_at": func.clock_timestamp(),
    }
    stmt = (
        insert(EvalRun)
        .values(run_id=run_id, **fields)
        .on_conflict_do_update(index_elements=[EvalRun.run_id], set_=fields)
    )
    await session.execute(stmt)
    return run_id


def _rate(passed: int, judged: int) -> float | None:
    return round(passed / judged, 4) if judged else None


async def get_latest_eval_report(session: AsyncSession, limit: int = 10) -> dict[str, Any]:
    """Where the system stands: how answers score, which failure modes occur, why
    the latest failures failed, and how the retrieval variants compared last time.
    """

    def count_where(*conditions: Any) -> Any:
        return func.count().filter(*conditions)

    faith, disagree = EvalResult.faithfulness_verdict, EvalResult.disagreement_verdict
    judged = ("PASS", "FAIL")  # SKIPPED and NOT_APPLICABLE are not judgements
    totals = (
        await session.execute(
            select(
                func.count().label("evaluated"),
                count_where(EvalResult.deterministic_passed).label("deterministic_passed"),
                count_where(faith.in_(judged)).label("faith_judged"),
                count_where(faith == "PASS").label("faith_passed"),
                count_where(disagree.in_(judged)).label("disagree_judged"),
                count_where(disagree == "PASS").label("disagree_passed"),
                func.avg(EvalResult.composite_score).label("mean_confidence"),
            ).select_from(EvalResult)
        )
    ).one()

    # One row per failure code per answer: the array is unnested next to its row.
    each = (
        func.jsonb_array_elements_text(EvalResult.failure_codes)
        .table_valued("code")
        .render_derived()
    )
    codes = await session.execute(
        select(each.c.code, func.count())
        .select_from(EvalResult)
        .join(each, true())
        .group_by(each.c.code)
        .order_by(each.c.code)
    )

    failed = await session.execute(
        select(EvalResult, Snapshot.question)
        .join(Snapshot, Snapshot.snapshot_id == EvalResult.snapshot_id)
        .where(EvalResult.composite_score < 1)
        .order_by(EvalResult.created_at.desc())
        .limit(limit)
    )
    recent_failures = []
    for result, question in failed.all():
        if result.faithfulness_verdict == "FAIL":
            critique = result.faithfulness_critique
        elif result.disagreement_verdict == "FAIL":
            critique = result.disagreement_critique
        else:
            broken = sorted(name for name, ok in result.checks.items() if not ok)
            critique = f"Deterministic checks failed: {', '.join(broken)}"
        recent_failures.append(
            {
                "snapshot_id": result.snapshot_id,
                "question": question,
                "failure_codes": result.failure_codes,
                "critique": critique,
                "evaluated_at": result.created_at.isoformat(),
            }
        )

    run = (
        await session.execute(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1))
    ).scalar_one_or_none()
    by_function = await session.execute(
        select(Failure.function_id, func.count()).group_by(Failure.function_id)
    )
    mean = totals.mean_confidence
    return {
        "answers": {
            "evaluated": totals.evaluated,
            "deterministic_pass_rate": _rate(totals.deterministic_passed, totals.evaluated),
            "faithfulness_judged": totals.faith_judged,
            "faithfulness_pass_rate": _rate(totals.faith_passed, totals.faith_judged),
            "disagreement_judged": totals.disagree_judged,
            "disagreement_pass_rate": _rate(totals.disagree_passed, totals.disagree_judged),
            "mean_confidence": round(float(mean), 4) if mean is not None else None,
        },
        "failure_modes": {name: count for name, count in codes.all()},
        "recent_failures": recent_failures,
        "latest_run": (
            {
                "run_id": run.run_id,
                "suite_name": run.suite_name,
                "created_at": run.created_at.isoformat(),
                "variants_summary": run.variants_summary,
            }
            if run
            else None
        ),
        "recorded_failures": {function_id: count for function_id, count in by_function.all()},
    }
