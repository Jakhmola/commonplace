"""OpenAI embeddings, batched.

The result carries a top-level `usage` dict. Any Inngest step that returns
one gets its cost recorded by `CostCaptureMiddleware` (inngest_client.py),
so call sites never write to the `costs` table themselves.
"""

import base64
from array import array
from typing import Any

from openai import AsyncOpenAI

from commonplace.config import settings

_client = AsyncOpenAI(api_key=settings.openai_api_key or "unset")


async def embed_texts(texts: list[str]) -> dict[str, Any]:
    """`{"embeddings": [...], "usage": {...}}`, embeddings in input order."""
    embeddings: list[list[float]] = []
    tokens = 0
    for start in range(0, len(texts), settings.embedding_batch_size):
        response = await _client.embeddings.create(
            model=settings.embedding_model,
            input=texts[start : start + settings.embedding_batch_size],
            dimensions=settings.embedding_dimensions,
        )
        # The API does not promise to answer in input order.
        embeddings += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        tokens += response.usage.total_tokens

    usd = tokens / 1_000_000 * settings.embedding_price_per_million_tokens
    return {
        "embeddings": embeddings,
        "usage": {
            "model": settings.embedding_model,
            "input_tokens": tokens,
            "output_tokens": 0,
            "est_usd": round(usd, 6),
        },
    }


def pack(vector: list[float]) -> str:
    """float32 + base64: about 8 KB per 1536-d vector instead of 33 KB of JSON floats.

    Inngest caps a step's output at 4 MB. As JSON, a 114-passage article was
    already at 3.8 MB; packed, the ceiling is roughly 490 passages.
    """
    return base64.b64encode(array("f", vector).tobytes()).decode()


def unpack(packed: str) -> list[float]:
    return array("f", base64.b64decode(packed)).tolist()
