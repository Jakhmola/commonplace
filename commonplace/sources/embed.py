from openai import AsyncOpenAI

from commonplace.config import settings

_BATCH = 256  # well under the API's per-request input limit


async def embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """Returns (vectors in input order, total tokens)."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    vectors: list[list[float]] = []
    tokens = 0
    for i in range(0, len(texts), _BATCH):
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=texts[i : i + _BATCH],
            dimensions=settings.embedding_dim,
        )
        vectors += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        tokens += resp.usage.total_tokens
    return vectors, tokens


def cost_usd(tokens: int) -> float:
    return tokens / 1_000_000 * settings.embedding_usd_per_mtok
