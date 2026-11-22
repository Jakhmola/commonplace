"""Measure the evaluators against the labelled traces.

    uv run python scripts/calibrate_judges.py --mode offline   # checks in code only, free
    uv run python scripts/calibrate_judges.py --mode live      # also the two judges (paid)

For each evaluator this prints how many labelled failures it caught (true
positive rate) and how many good answers it cleared (true negative rate), and
lists every trace where it disagreed with the label, with its reasoning.
Exits 1 if any evaluator that ran is below the thresholds (EVAL_MIN_TPR,
EVAL_MIN_TNR, both 0.90 by default).

`live` makes one model call per trace per judge: about 25 calls.
"""

import argparse
import asyncio
import sys

from commonplace.config import settings
from commonplace.evals.calibration import calibrate_deterministic, calibrate_judge
from commonplace.evals.judges import BinaryVerdict
from commonplace.evals.metrics import CalibrationReport, load_labeled_traces


def _rate(value: float | None) -> str:
    return "n/a (no examples)" if value is None else f"{value:.2f}"


def show(
    name: str, report: CalibrationReport, verdicts: dict[str, BinaryVerdict] | None = None
) -> None:
    print(f"\n{name}: {report.total} traces")
    print(
        f"  labelled FAIL {report.labeled_fail}: caught {report.true_positives}, "
        f"missed {report.false_negatives}   -> TPR {_rate(report.tpr)}"
    )
    print(
        f"  labelled PASS {report.labeled_pass}: cleared {report.true_negatives}, "
        f"flagged {report.false_positives}  -> TNR {_rate(report.tnr)}"
    )
    print(f"  aligned (TPR >= {settings.eval_min_tpr}, TNR >= {settings.eval_min_tnr}): ", end="")
    print("yes" if report.aligned else "NO")
    for miss in report.disagreements:
        print(f"  - {miss['trace_id']}: labelled {miss['labeled']}, got {miss['predicted']}")
        if verdicts and miss["trace_id"] in verdicts:
            print(f"      {verdicts[miss['trace_id']].critique[:600]}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    args = parser.parse_args()

    traces = load_labeled_traces()
    print(f"{len(traces)} labelled traces from {settings.eval_labeled_traces_file}")

    reports = {"deterministic": calibrate_deterministic(traces)}
    show("deterministic checks", reports["deterministic"])

    if args.mode == "live":
        if not settings.anthropic_api_key:
            sys.exit("--mode live needs ANTHROPIC_API_KEY")
        for evaluator in ("faithfulness", "disagreement"):
            reports[evaluator], verdicts = await calibrate_judge(evaluator, traces)
            show(f"{evaluator} judge ({settings.eval_judge_model})", reports[evaluator], verdicts)

    if not all(report.aligned for report in reports.values()):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
