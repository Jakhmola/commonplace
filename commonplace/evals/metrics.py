"""Numbers that say whether an evaluator, or a retriever, can be trusted.

Calibration: an evaluator is itself a classifier ("is this answer a
failure?"), so before its verdicts mean anything they are compared with
labelled examples. Two rates matter and they fail differently: a low true
positive rate means failures slip through, a low true negative rate means
good answers get flagged. Accuracy would hide which one is wrong.

Retrieval: recall@k and reciprocal rank of the expected article.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from commonplace.config import settings

_REPO_ROOT = Path(__file__).resolve().parents[2]

FAIL, PASS = "FAIL", "PASS"


@dataclass(frozen=True)
class CalibrationReport:
    total: int
    labeled_fail: int
    labeled_pass: int
    true_positives: int  # failures the evaluator caught
    false_negatives: int  # failures it let through
    true_negatives: int  # good answers it cleared
    false_positives: int  # good answers it flagged
    tpr: float | None  # None: no failing example to measure on
    tnr: float | None  # None: no passing example to measure on
    aligned: bool
    disagreements: list[dict[str, Any]]


def compute_tpr_tnr(
    pairs: list[tuple[str, str, dict[str, Any]]],
    *,
    min_tpr: float = settings.eval_min_tpr,
    min_tnr: float = settings.eval_min_tnr,
) -> CalibrationReport:
    """Compare `(labeled, predicted, trace)` triples. The positive class is FAIL.

    A rate with no examples behind it is undefined, and an evaluator with an
    undefined rate is not aligned: nothing has shown it can do that half of
    the job.
    """
    counts = {(lab, pred): 0 for lab in (FAIL, PASS) for pred in (FAIL, PASS)}
    disagreements = []
    for labeled, predicted, trace in pairs:
        counts[(labeled, predicted)] += 1
        if labeled != predicted:
            disagreements.append(
                {"trace_id": trace.get("trace_id"), "labeled": labeled, "predicted": predicted}
            )
    tp, fn = counts[(FAIL, FAIL)], counts[(FAIL, PASS)]
    tn, fp = counts[(PASS, PASS)], counts[(PASS, FAIL)]
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return CalibrationReport(
        total=len(pairs),
        labeled_fail=tp + fn,
        labeled_pass=tn + fp,
        true_positives=tp,
        false_negatives=fn,
        true_negatives=tn,
        false_positives=fp,
        tpr=tpr,
        tnr=tnr,
        aligned=tpr is not None and tnr is not None and tpr >= min_tpr and tnr >= min_tnr,
        disagreements=disagreements,
    )


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """The share of the expected ids that appear in the first `k` results."""
    if not expected_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return sum(1 for expected in expected_ids if expected in top) / len(expected_ids)


def mrr(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """1 / rank of the first expected id within the first `k` results, else 0."""
    expected = set(expected_ids)
    for rank, retrieved in enumerate(retrieved_ids[:k], start=1):
        if retrieved in expected:
            return 1 / rank
    return 0.0


def _path(name: str, path: str | Path | None) -> Path:
    file = Path(path or name)
    return file if file.is_absolute() else _REPO_ROOT / file


def load_taxonomy(path: str | Path | None = None) -> list[dict[str, Any]]:
    data = yaml.safe_load(_path(settings.eval_taxonomy_file, path).read_text())
    return list(data["failure_modes"])


def load_labeled_traces(path: str | Path | None = None) -> list[dict[str, Any]]:
    lines = _path(settings.eval_labeled_traces_file, path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.startswith("#")]
