"""The Inngest client, and the middleware that records what paid calls cost.

A step that calls a paid API returns a dict with a top-level `usage` key
(`model`, `input_tokens`, `output_tokens`, `est_usd`). The middleware sees
every step output on its way back to Inngest and writes one `costs` row for
each such dict, so no call site touches the `costs` table.

This cannot double count on replay: a finished step is answered from
Inngest's memo and its body never runs again, so its output never passes
through here a second time. A step that failed and is retried does call the
API again, and is correctly recorded again.
"""

import logging
import uuid
from typing import Any

import inngest

from commonplace.config import settings
from commonplace.db.queries import insert_cost
from commonplace.db.session import get_session

logger = logging.getLogger(__name__)

_USAGE_KEYS = {"model", "input_tokens", "output_tokens", "est_usd"}


def usage_from_output(output: object) -> dict[str, Any] | None:
    """The `usage` dict of a step output, if it has a complete one."""
    if not isinstance(output, dict):
        return None
    usage = output.get("usage")
    if isinstance(usage, dict) and usage.keys() >= _USAGE_KEYS:
        return usage
    return None


class CostCaptureMiddleware(inngest.Middleware):
    async def transform_output(self, result: inngest.TransformOutputResult) -> None:
        if result.step is None:
            return  # the function's own return value, not a step that just ran
        usage = usage_from_output(result.output)
        if usage is None:
            return
        article_id = usage.get("article_id")
        try:
            async with get_session() as session:
                await insert_cost(
                    session,
                    model=usage["model"],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    est_usd=usage["est_usd"],
                    article_id=uuid.UUID(article_id) if article_id else None,
                )
                await session.commit()
        except Exception:
            # Bookkeeping must never fail the step: an exception here would make
            # Inngest discard the step's output and run the paid call again.
            logger.exception("Could not record cost %s", usage)


client = inngest.Inngest(
    app_id="commonplace",
    is_production=not settings.inngest_dev,
    api_base_url=settings.inngest_base_url if settings.inngest_dev else None,
    event_api_base_url=settings.inngest_base_url if settings.inngest_dev else None,
    # Inngest Cloud: the event key authorises sending, the signing key verifies
    # that requests to /api/inngest really come from Inngest.
    event_key=settings.inngest_event_key or None,
    signing_key=settings.inngest_signing_key or None,
    logger=logging.getLogger("uvicorn"),
    middleware=[CostCaptureMiddleware],
)
