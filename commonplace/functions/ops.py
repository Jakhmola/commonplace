"""inngest/function.failed -> handle-function-failed.

Inngest sends this event when a run has failed for good (retries exhausted,
or a non-retriable error). Each one becomes a row in `failures`: it is how a
job learns that one of its articles will never arrive, and real failures are
worth keeping for judging the system later.
"""

from typing import Any

import inngest

from commonplace.db.queries import record_failure
from commonplace.db.session import get_session
from commonplace.inngest_client import client

FUNCTION_FAILED = "inngest/function.failed"
_OWN_ID = "handle-function-failed"


def failure_from_event(data: dict[str, Any]) -> dict[str, Any] | None:
    """The `failures` row for a failure event, or None when it must not be recorded.

    Only the error message is kept: the full error also carries a stack trace
    with local paths. This handler's own failures are skipped, or a failure to
    record a failure would record itself, forever.
    """
    function_id = str(data.get("function_id") or "unknown")
    if function_id.endswith(_OWN_ID):
        return None
    error = data.get("error")
    message = (error.get("message") or error.get("name")) if isinstance(error, dict) else error
    original = data.get("event")
    return {
        "function_id": function_id,
        "run_id": str(data.get("run_id") or "unknown"),
        "error": str(message or "unknown error"),
        "event_data": original if isinstance(original, dict) else None,
    }


@client.create_function(
    fn_id=_OWN_ID,
    trigger=inngest.TriggerEvent(event=FUNCTION_FAILED),
    retries=1,
)
async def handle_function_failed(ctx: inngest.Context) -> dict[str, object]:
    failure = failure_from_event(dict(ctx.event.data or {}))
    if failure is None:
        return {"status": "ignored"}

    async def record() -> str:
        async with get_session() as session:
            failure_id = await record_failure(session, **failure)
            await session.commit()
        return str(failure_id)

    return {"status": "recorded", "failure_id": await ctx.step.run("record", record)}
