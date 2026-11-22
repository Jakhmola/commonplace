"""The graph's vocabulary, decided before any code writes to it.

The ontology is closed. Five kinds of node and five kinds of edge:

    (Article)-[:PUBLISHED_IN]->(Publication)
    (Article)-[:WRITTEN_BY]->(Author)
    (Article)-[:ASSERTS]->(Claim)
    (Claim)-[:ABOUT]->(Topic)
    (Claim)-[:SUPERSEDES]->(Claim)

Topics are what connect one publication's claims to another's, so they only
work if "RAG", "retrieval-augmented generation" and "RAGs" are one node. Every
topic and author string is canonicalised here, deterministically, before it
reaches Postgres or the graph.
"""

import re
import unicodedata
from typing import NamedTuple

from commonplace.schemas.entities import ExtractedTopic

NODE_LABELS = frozenset({"Publication", "Author", "Article", "Claim", "Topic"})
EDGE_TYPES = frozenset({"PUBLISHED_IN", "WRITTEN_BY", "ASSERTS", "ABOUT", "SUPERSEDES"})


class AuthorRef(NamedTuple):
    slug: str
    name: str


UNKNOWN_AUTHOR = AuthorRef("unknown-author", "Unknown Author")

# slug of a phrase (singular) -> (canonical slug, display name)
TOPIC_ALIASES: dict[str, tuple[str, str]] = {
    "rag": ("rag", "RAG"),
    "retrieval-augmented-generation": ("rag", "RAG"),
    "llm": ("llm", "LLM"),
    "large-language-model": ("llm", "LLM"),
    "slm": ("small-language-model", "Small Language Model"),
    "small-language-model": ("small-language-model", "Small Language Model"),
    "mcp": ("mcp", "Model Context Protocol (MCP)"),
    "model-context-protocol": ("mcp", "Model Context Protocol (MCP)"),
    "rl": ("reinforcement-learning", "Reinforcement Learning"),
    "reinforcement-learning": ("reinforcement-learning", "Reinforcement Learning"),
    "fine-tuning": ("fine-tuning", "Fine-Tuning"),
    "finetuning": ("fine-tuning", "Fine-Tuning"),
    "fine-tune": ("fine-tuning", "Fine-Tuning"),
    "eval": ("evaluation", "Evaluation"),
    "evaluation": ("evaluation", "Evaluation"),
    "llm-evaluation": ("evaluation", "Evaluation"),
    "agent": ("agent", "Agent"),
    "ai-agent": ("agent", "Agent"),
    "coding-agent": ("coding-agent", "Coding Agent"),
    "ai-coding-agent": ("coding-agent", "Coding Agent"),
    "agi": ("agi", "AGI"),
    "artificial-general-intelligence": ("agi", "AGI"),
    "rrf": ("reciprocal-rank-fusion", "Reciprocal Rank Fusion"),
    "reciprocal-rank-fusion": ("reciprocal-rank-fusion", "Reciprocal Rank Fusion"),
    "hybrid-search": ("hybrid-retrieval", "Hybrid Retrieval"),
    "hybrid-retrieval": ("hybrid-retrieval", "Hybrid Retrieval"),
    "reranker": ("reranker", "Reranker"),
    "re-ranker": ("reranker", "Reranker"),
    "reranking": ("reranker", "Reranker"),
    "test-time-compute": ("test-time-compute", "Test-Time Compute"),
    "inference-time-compute": ("test-time-compute", "Test-Time Compute"),
    "durable-execution": ("durable-execution", "Durable Execution"),
    "open-source-model": ("open-model", "Open Model"),
    "open-weight-model": ("open-model", "Open Model"),
    "open-model": ("open-model", "Open Model"),
}

# Too generic to connect anything: as topics these would become hubs that link
# every claim to every other claim.
STOP_TOPICS = frozenset(
    [
        "ai",
        "system",
        "code",
        "article",
        "author",
        "data",
        "model",
        "software",
        "tool",
        "technology",
        "thing",
        "method",
        "approach",
        "use",
        "way",
        "paper",
        "post",
        "newsletter",
        "company",
        "team",
        "people",
        "work",
        "idea",
        "problem",
    ]
)

_ACRONYMS = frozenset(
    [
        "ai",
        "api",
        "bm25",
        "cpu",
        "gpt",
        "gpu",
        "kv",
        "llm",
        "lora",
        "mcp",
        "ml",
        "moe",
        "nlp",
        "rag",
        "rl",
        "rlhf",
        "rrf",
        "sdk",
        "sql",
        "ui",
    ]
)

# Words ending in "s" that are not plurals.
_NOT_PLURAL = frozenset(
    [
        "analysis",
        "bias",
        "class",
        "corpus",
        "kubernetes",
        "pass",
        "postgres",
        "redis",
        "series",
        "status",
    ]
)


def slugify(raw: str) -> str:
    """Lowercase ASCII with runs of anything else collapsed to single hyphens."""
    ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def _singular(word: str) -> str:
    if len(word) <= 3 or word in _NOT_PLURAL or word.endswith(("ss", "us", "is", "ics")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def normalize_author(raw: str | None) -> AuthorRef:
    name = " ".join((raw or "").split())
    slug = slugify(name)
    return AuthorRef(slug, name) if slug else UNKNOWN_AUTHOR


def normalize_topic(raw: str) -> ExtractedTopic | None:
    """The canonical topic for a raw entity string, or None if it is too generic to keep."""
    words = slugify(unicodedata.normalize("NFKC", raw)).split("-")
    words[-1] = _singular(words[-1])  # "coding agents" and "coding agent" are one topic
    slug = "-".join(w for w in words if w)
    if len(slug) < 2 or slug in STOP_TOPICS:
        return None
    if slug in TOPIC_ALIASES:
        canonical, name = TOPIC_ALIASES[slug]
        return ExtractedTopic(slug=canonical, name=name)
    name = " ".join(w.upper() if w in _ACRONYMS else w.capitalize() for w in slug.split("-"))
    return ExtractedTopic(slug=slug, name=name)


def normalize_topics(raw_entities: list[str]) -> list[ExtractedTopic]:
    """Canonical topics in their original order, without duplicates or rejected ones."""
    topics: dict[str, ExtractedTopic] = {}
    for raw in raw_entities:
        topic = normalize_topic(raw)
        if topic is not None:
            topics.setdefault(topic.slug, topic)
    return list(topics.values())
