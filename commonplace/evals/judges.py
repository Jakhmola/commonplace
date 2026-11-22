"""Level 2: narrow binary judges, for what only reading can decide.

Four rules shape them:

- A verdict is `PASS` or `FAIL`. A 1-5 scale invites a judge to answer "3"
  and tells nobody what to fix.
- The critique comes first in the schema. The model fills fields in order, so
  it has to go through the evidence before it commits to a verdict.
- One judge per kind of failure. A judge asked about faithfulness and about
  disagreement at once does both worse and cannot be calibrated on either.
- A judge is a classifier and is treated as one: its agreement with labelled
  traces is measured (`scripts/calibrate_judges.py`) before its verdicts count.

Without an API key no judge runs and the verdict is recorded as `SKIPPED`.
There is deliberately no rule-based stand-in: a keyword heuristic labelled
"judge" would produce numbers that mean nothing.
"""

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from commonplace.agents.core import Agent
from commonplace.agents.prompts import (
    DISAGREEMENT_JUDGE_SYSTEM_PROMPT,
    FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
)
from commonplace.config import settings
from commonplace.evals.deterministic import prompt_passages

Verdict = Literal["PASS", "FAIL"]
JudgeFailureCode = Literal["NONE", "FM-3", "FM-4", "FM-5"]

SKIPPED = "SKIPPED"  # no API key: the judge did not run
NOT_APPLICABLE = "NOT_APPLICABLE"  # the judge had nothing to judge


class BinaryVerdict(BaseModel):
    critique: str = Field(
        min_length=10,
        description="The evidence, statement by statement, written before deciding the verdict",
    )
    verdict: Verdict = Field(description="PASS only if there is no violation at all")
    failure_code: JudgeFailureCode = Field(
        default="NONE", description="Which failure mode a FAIL is; NONE for a PASS"
    )


def _judge(name: str, role: str, system_prompt: str) -> Agent[BinaryVerdict]:
    return Agent(
        name=name,
        role=role,
        system_prompt=system_prompt,
        model=settings.eval_judge_model,
        output_schema=BinaryVerdict,
        max_tokens=settings.eval_judge_max_tokens,
        effort=settings.eval_judge_effort,
    )


faithfulness_judge = _judge(
    "faithfulness-judge",
    "Is every statement in the answer supported by the passage it cites?",
    FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
)
disagreement_judge = _judge(
    "disagreement-judge",
    "Does the answer hide a real disagreement between its sources?",
    DISAGREEMENT_JUDGE_SYSTEM_PROMPT,
)


def judge_version(
    judges: tuple[Agent[BinaryVerdict], ...] = (faithfulness_judge, disagreement_judge),
) -> str:
    """`<model>:<hash of the prompts>`, stored with every verdict.

    A verdict means what its judge meant. When a prompt or the model changes,
    earlier verdicts were given by a different judge and should not be
    averaged with new ones as if nothing had happened.
    """
    prompts = "\x00".join(judge.system_prompt for judge in judges)
    return f"{judges[0].model}:{hashlib.sha256(prompts.encode()).hexdigest()[:8]}"


def requires_disagreement_check(retrieved: list[dict[str, Any]], prompt_limit: int = 5) -> bool:
    """Two or more known authors among the passages the model was shown.

    With a single author there is nobody to disagree with, and the judge is
    not asked.
    """
    authors = {
        str(p["author"]).strip().lower()
        for p in prompt_passages(retrieved, prompt_limit)
        if p.get("author")
    }
    return len(authors) >= 2


def format_judge_input(
    *, question: str, retrieved: list[dict[str, Any]], answer_text: str, prompt_limit: int = 5
) -> str:
    """One trace as a judge reads it: the question, exactly the passages the
    model saw (numbered as the answer cites them), and the answer.
    """
    blocks = [
        f"[{n}] Author: {p.get('author') or 'Unknown'} | Title: {p.get('title') or 'Unknown'} | "
        f"Claim type: {p.get('claim_type') or 'none'}\n"
        + (f"Claim: {p['claim_text']}\n" if p.get("claim_text") else "")
        + f"Text: {p.get('text', '')}"
        for n, p in enumerate(prompt_passages(retrieved, prompt_limit), start=1)
    ]
    passages = "\n\n".join(blocks) or "(no passages were retrieved)"
    return (
        f"Question: {question}\n\n"
        f"Passages shown to the model:\n{passages}\n\n"
        f"Answer:\n{answer_text}"
    )
