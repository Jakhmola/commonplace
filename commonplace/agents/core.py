"""A hand-written agent layer on Inngest steps. No framework.

An `Agent` is only a description: a role, a prompt, a model and the shape of
its answer. `run_agent` makes one model call for it inside `step.run`, which
is what gives the call its durability: the result is memoized, a retry of the
function never pays for the same call twice, and the call's `usage` passes
through the cost middleware.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from commonplace.agents.providers import Effort, call_llm


@dataclass(frozen=True)
class Agent[T: BaseModel]:
    name: str
    role: str
    system_prompt: str
    model: str
    output_schema: type[T] | None = None  # None: the agent answers in plain text
    max_tokens: int = 16000
    effort: Effort = "medium"


async def run_agent_step[T: BaseModel](
    step: Any,
    agent: Agent[T],
    user_input: str,
    step_id: str,
    article_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Run `agent` once as the step `step_id`: `{"output", "usage", "stop_reason"}`.

    For callers that need to know what the call cost, such as a loop with a budget.
    """

    async def call() -> dict[str, Any]:
        return await call_llm(
            system_prompt=agent.system_prompt,
            user_input=user_input,
            model=agent.model,
            max_tokens=agent.max_tokens,
            effort=agent.effort,
            output_schema=agent.output_schema,
            article_id=article_id,
        )

    result: dict[str, Any] = await step.run(step_id, call)
    return result


async def run_agent[T: BaseModel](
    step: Any,
    agent: Agent[T],
    user_input: str,
    step_id: str,
    article_id: uuid.UUID | None = None,
) -> T | str:
    """Run `agent` once as the step `step_id`; return its validated output."""
    result = await run_agent_step(step, agent, user_input, step_id, article_id)
    if agent.output_schema is None:
        return str(result["output"] or "")
    # Validated again here: on a replay the output comes from Inngest's memo as plain JSON.
    return agent.output_schema.model_validate(result["output"])
