"""The one place that talks to the LLM provider.

Every call returns its result together with a `usage` dict shaped for
`CostCaptureMiddleware`, so a call made inside an Inngest step has its cost
recorded without the caller doing anything.
"""

import uuid
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

from commonplace.config import settings

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# USD per million tokens (input, output).
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
_DEFAULT_PRICE = MODEL_PRICES["claude-opus-5-5"]

_llm: anthropic.AsyncAnthropic | None = None


class ModelRefusal(RuntimeError):
    """The provider declined the request (and its fallback did too). Trying the
    same request again will usually be declined again.
    """


def get_llm() -> anthropic.AsyncAnthropic | None:
    """The Anthropic client, or None when no key is configured."""
    global _llm
    if not settings.anthropic_api_key:
        return None
    if _llm is None:
        _llm = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _llm


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """What a call cost at list price. An unknown model is priced like the default, not as free."""
    input_price, output_price = MODEL_PRICES.get(model, _DEFAULT_PRICE)
    return round((input_tokens * input_price + output_tokens * output_price) / 1_000_000, 6)


async def call_llm(
    *,
    system_prompt: str,
    user_input: str,
    model: str,
    max_tokens: int,
    effort: Effort = "medium",
    output_schema: type[BaseModel] | None = None,
    article_id: uuid.UUID | None = None,
    llm: anthropic.AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """One model call: `{"output", "usage", "stop_reason"}`.

    With `output_schema` the response is constrained to that schema and
    `output` is the validated object as a dict (None if the model produced
    nothing parseable); otherwise `output` is the text. Provider errors are
    raised, so inside a step they are retried.
    """
    llm = llm or get_llm()
    if llm is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "output_config": {"effort": effort},
        # If the model declines, the API re-runs the request on a fallback model.
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_input}],
    }
    output: Any
    if output_schema is None:
        response = await llm.beta.messages.create(**request)
        output = "\n\n".join(block.text for block in response.content if block.type == "text")
    else:
        response = await llm.beta.messages.parse(**request, output_format=output_schema)
        parsed = response.parsed_output
        if parsed is None:
            # Raised here, inside the step, so the step is retried with a fresh
            # call. Returning None would be memoized, and every replay of the
            # function would then fail on the same empty result.
            error = ModelRefusal if response.stop_reason == "refusal" else RuntimeError
            raise error(
                f"The model returned no parseable output (stop reason: {response.stop_reason})"
            )
        output = parsed.model_dump(mode="json")

    input_tokens, output_tokens = response.usage.input_tokens, response.usage.output_tokens
    # After a server-side fallback the answer came from another model, at its price.
    served_by = getattr(response, "model", None) or model
    return {
        "output": output,
        "usage": {
            "model": served_by,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "est_usd": estimate_cost_usd(served_by, input_tokens, output_tokens),
            "article_id": str(article_id) if article_id else None,
        },
        "stop_reason": response.stop_reason,
    }
