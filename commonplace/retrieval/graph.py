"""Graph retrieval: from the topics in a question to claims, and from claims to passages.

The other two retrievers match text. This one follows structure: topics named
in the question lead to the current claims about them, whatever words those
claims use and whichever publication made them. Each claim is then mapped
back to the passage that holds its quote, so the result is a ranking of
passages like the other two and can be fused with them by rank.

The graph is optional at read time. If Memgraph cannot be reached this
retriever returns nothing and search carries on with the other two.
"""

import logging
import re
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from commonplace.config import settings
from commonplace.db.queries import find_matching_passage_for_quote
from commonplace.graph.client import search_graph_by_topics
from commonplace.graph.ontology import STOP_TOPICS, normalize_topic

logger = logging.getLogger(__name__)

# Words that shape a question without being its subject.
_QUESTION_WORDS = frozenset(
    [
        "a",
        "about",
        "across",
        "agree",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "can",
        "compare",
        "could",
        "did",
        "disagree",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "over",
        "production",
        "say",
        "says",
        "should",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "to",
        "under",
        "use",
        "was",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
    ]
)
_NOT_A_TOPIC = STOP_TOPICS | _QUESTION_WORDS


def extract_query_seed_topics(query: str) -> tuple[list[str], list[str]]:
    """(topic slugs, plain search terms) to start a graph traversal from.

    Slugs come from every run of one to three content words, canonicalised the
    same way topics are when they are stored, so "large language models" in a
    question finds the `llm` topic. Search terms are the single content words,
    matched against whole words of a topic's slug.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _NOT_A_TOPIC]
    slugs: dict[str, None] = {}
    for size in (3, 2, 1):
        for start in range(len(words) - size + 1):
            topic = normalize_topic(" ".join(words[start : start + size]))
            if topic is not None:
                slugs.setdefault(topic.slug)
    terms = list(dict.fromkeys(w for w in words if len(w) >= 3))[:8]
    return list(slugs), terms


async def search_graph_claims(
    session: AsyncSession, query: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Passages reached through the claim graph, best first, each with the claim that led to it."""
    topic_slugs, search_terms = extract_query_seed_topics(query)
    if not topic_slugs and not search_terms:
        return []
    try:
        claims = await search_graph_by_topics(
            topic_slugs=topic_slugs,
            search_terms=search_terms,
            limit=limit or settings.graph_retrieval_limit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph retrieval skipped, Memgraph unavailable: %s", exc)
        return []

    results: dict[str, dict[str, Any]] = {}
    for claim in claims:
        passage = await find_matching_passage_for_quote(
            session, article_id=uuid.UUID(claim["article_id"]), quote_span=claim["quote_span"]
        )
        if passage is None:
            continue  # the article has no current passages: nothing to cite
        # Several claims can be quoted from one passage. It is retrieved once,
        # carrying the best-ranked of them.
        results.setdefault(
            passage["passage_id"],
            {
                **passage,
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "claim_type": claim["claim_type"],
                "quote_span": claim["quote_span"],
                "matched_topics": int(claim.get("matched_topics") or 1),
                "rank": float(claim.get("confidence") or 0.0),
            },
        )
    return list(results.values())
