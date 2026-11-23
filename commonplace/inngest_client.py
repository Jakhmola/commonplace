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


def inngest_origins(*, dev: bool, base_url: str) -> tuple[str, str]:
    """(API address, event API address) for the client.

    Always passed explicitly. Left unset, the SDK reads `INNGEST_BASE_URL`
    from the environment, and `make` exports `.env`: a cloud-mode client
    would then send every event to the local dev server address.
    """
    if not dev and base_url == "https://api.inngest.com":
        return base_url, "https://inn.gs"  # Inngest Cloud takes events on its own host
    return base_url, base_url  # the dev server, or a self-hosted Inngest: one address


def inngest_event_key(*, dev: bool, key: str) -> str | None:
    """The event key for the client, never left for the SDK to read itself.

    Given None, the SDK falls back to `INNGEST_EVENT_KEY` from the environment,
    and `make` exports `.env`, where the key is empty in dev mode. An empty key
    sends events to `/e/`, which the dev server answers with an empty redirect.
    The dev server accepts any key, so it gets the SDK's own placeholder.
    """
    if key:
        return key
    return "NO_EVENT_KEY_SET" if dev else None


_api_origin, _event_api_origin = inngest_origins(
    dev=settings.inngest_dev, base_url=settings.inngest_base_url
)

client = inngest.Inngest(
    app_id="commonplace",
    is_production=not settings.inngest_dev,
    api_base_url=_api_origin,
    event_api_base_url=_event_api_origin,
    # Inngest Cloud: the event key authorises sending, the signing key verifies
    # that requests to /api/inngest really come from Inngest.
    event_key=inngest_event_key(dev=settings.inngest_dev, key=settings.inngest_event_key),
    signing_key=settings.inngest_signing_key or None,
    logger=logging.getLogger("uvicorn"),
    middleware=[CostCaptureMiddleware],
)
