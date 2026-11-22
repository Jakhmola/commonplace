import base64
from array import array

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


def pack(v: list[float]) -> str:
    """float32 + base64: ~8 KB per 1536-d vector instead of ~33 KB of JSON.

    Inngest caps a step's output at 4 MB; as JSON floats a 114-passage article
    already hit 3.8 MB. Packed, the ceiling is roughly 490 passages per article.
    """
    return base64.b64encode(array("f", v).tobytes()).decode()


def unpack(s: str) -> list[float]:
    return array("f", base64.b64decode(s)).tolist()
