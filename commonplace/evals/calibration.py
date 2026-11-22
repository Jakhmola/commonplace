"""Measuring the evaluators against the labelled traces.

Each trace in `evals/labeled_traces.jsonl` says what verdict each evaluator
should reach on it, or null where that evaluator is not being tested. Running
an evaluator over the traces it is scored on and comparing gives its true
positive rate (failures caught) and true negative rate (good answers
cleared). An evaluator below the thresholds is not ready to be trusted: fix
its prompt or its rule, and measure again.
"""

from typing import Any

from commonplace.agents.core import Agent
from commonplace.agents.providers import ModelRefusal, call_llm
from commonplace.evals.deterministic import run_deterministic_checks
from commonplace.evals.judges import (
    BinaryVerdict,
    disagreement_judge,
    faithfulness_judge,
    format_judge_input,
)
from commonplace.evals.metrics import CalibrationReport, compute_tpr_tnr

EVALUATORS = ("deterministic", "faithfulness", "disagreement")
_JUDGES: dict[str, Agent[BinaryVerdict]] = {
    "faithfulness": faithfulness_judge,
    "disagreement": disagreement_judge,
}


def in_scope(trace: dict[str, Any], evaluator: str) -> bool:
    """Whether the trace carries an expected verdict for this evaluator."""
    return trace["expected"].get(evaluator) is not None


def predict_deterministic(trace: dict[str, Any]) -> str:
    result = run_deterministic_checks(
        answer_text=trace["answer_text"],
        citations=trace["citations"],
        retrieved=trace["retrieved"],
        superseded_ids=frozenset(trace.get("superseded_ids", [])),
    )
    return "PASS" if result.passed else "FAIL"


def calibrate_deterministic(traces: list[dict[str, Any]]) -> CalibrationReport:
    """The checks in code against their labels. Free: no model is called."""
    return compute_tpr_tnr(
        [
            (trace["expected"]["deterministic"], predict_deterministic(trace), trace)
            for trace in traces
            if in_scope(trace, "deterministic")
        ]
    )


async def predict_judge(evaluator: str, trace: dict[str, Any]) -> BinaryVerdict | None:
    """One real judge call on one trace (not inside a step: this is a measurement run).

    None when the provider refuses the request twice.
    """
    judge = _JUDGES[evaluator]
    for _attempt in range(2):
        try:
            result = await call_llm(
                system_prompt=judge.system_prompt,
                user_input=format_judge_input(
                    question=trace["question"],
                    retrieved=trace["retrieved"],
                    answer_text=trace["answer_text"],
                ),
                model=judge.model,
                max_tokens=judge.max_tokens,
                effort=judge.effort,
                output_schema=BinaryVerdict,
            )
        except ModelRefusal:
            continue
        return BinaryVerdict.model_validate(result["output"])
    return None


async def calibrate_judge(
    evaluator: str, traces: list[dict[str, Any]]
) -> tuple[CalibrationReport, dict[str, BinaryVerdict]]:
    """A judge against its labels: the report, and each verdict with its critique.

    A trace the judge could not be run on counts as a wrong verdict: a judge
    that does not answer has not caught the failure or cleared the answer.
    """
    verdicts: dict[str, BinaryVerdict] = {}
    pairs: list[tuple[str, str, dict[str, Any]]] = []
    for trace in traces:
        if not in_scope(trace, evaluator):
            continue
        expected = trace["expected"][evaluator]
        verdict = await predict_judge(evaluator, trace)
        if verdict is None:
            verdict = BinaryVerdict(
                critique="The provider refused to run the judge on this trace.",
                verdict="PASS" if expected == "FAIL" else "FAIL",
            )
        verdicts[trace["trace_id"]] = verdict
        pairs.append((expected, verdict.verdict, trace))
    return compute_tpr_tnr(pairs), verdicts
