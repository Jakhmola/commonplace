"""Unit tests for the evaluation suite: checks in code, judges, metrics, the benchmark.

No Docker or keys needed.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from commonplace.evals.deterministic import is_abstain_answer, run_deterministic_checks
from commonplace.evals.judges import (
    BinaryVerdict,
    disagreement_judge,
    faithfulness_judge,
    format_judge_input,
    requires_disagreement_check,
)
from commonplace.evals.metrics import compute_tpr_tnr, load_taxonomy, mrr, recall_at_k

# --- the taxonomy: every failure mode has a name and someone responsible for it ---


def test_the_taxonomy_names_six_failure_modes_with_an_evaluator_each() -> None:
    modes = load_taxonomy()
    assert [m["code"] for m in modes] == ["FM-1", "FM-2", "FM-3", "FM-4", "FM-5", "FM-6"]
    for mode in modes:
        assert mode["slug"] and mode["evaluator"] and mode["description"].strip(), mode["code"]
    assert len({m["slug"] for m in modes}) == 6


# --- level 1: what plain code can check about an answer ---


def _passage(n: int, **extra: Any) -> dict[str, Any]:
    return {
        "passage_id": f"p{n}",
        "text": f"text of passage {n}",
        "title": "A post",
        "author": "Jane Writer",
        "url": "https://www.latent.space/p/a-post",
        "in_prompt": n <= 3,  # three passages reached the prompt, the rest are candidates only
        **extra,
    }


def _citation(n: int, **extra: Any) -> dict[str, Any]:
    return {"citation_idx": n, "passage_id": f"p{n}", "title": "A post", **extra}


RETRIEVED = [_passage(n) for n in range(1, 7)]


def _check(answer: str, citations: list[dict[str, Any]], **kwargs: Any) -> Any:
    return run_deterministic_checks(
        answer_text=answer,
        citations=citations,
        retrieved=kwargs.pop("retrieved", RETRIEVED),
        **kwargs,
    )


def test_a_clean_cited_answer_passes_every_check() -> None:
    result = _check("Steps are memoized [1] and retried alone [3].", [_citation(1), _citation(3)])
    assert result.passed and result.failure_codes == [] and result.score == 1.0
    assert set(result.checks) == {
        "citation_bounds_valid",
        "citations_subset_of_retrieved",
        "no_superseded_sources_cited",
        "abstain_or_cite_discipline",
    }


def test_a_citation_outside_the_prompt_is_a_phantom() -> None:
    # [5] exists among the candidates, but only three passages were shown to the model
    result = _check("It is fast [1] and cheap [5].", [_citation(1)])
    assert result.checks["citation_bounds_valid"] is False
    assert result.failure_codes == ["FM-1"] and result.score == 0.75


def test_a_citation_to_a_passage_that_was_not_retrieved_is_dangling() -> None:
    result = _check("It is fast [1].", [_citation(1, passage_id="not-among-the-candidates")])
    assert result.checks["citations_subset_of_retrieved"] is False
    assert result.failure_codes == ["FM-1"]
    # a citation whose number and passage do not line up is the same failure
    swapped = _check("It is fast [1].", [_citation(1, passage_id="p2")])
    assert swapped.checks["citations_subset_of_retrieved"] is False


def test_citation_lists_and_ranges_are_checked_like_ask_parses_them() -> None:
    fine = _check("All three agree [1-3].", [_citation(1), _citation(2), _citation(3)])
    assert fine.passed
    out_of_range = _check("Both say so [2, 4].", [_citation(2)])
    assert out_of_range.failure_codes == ["FM-1"]


def test_a_superseded_source_fails_fm2() -> None:
    cited = [_citation(1), _citation(2, claim_id="c-9")]
    assert _check("Yes [1][2].", cited).passed
    stale_passage = _check("Yes [1][2].", cited, superseded_ids=frozenset({"p1"}))
    assert stale_passage.failure_codes == ["FM-2"]
    stale_claim = _check("Yes [1][2].", cited, superseded_ids=frozenset({"c-9"}))
    assert stale_claim.checks["no_superseded_sources_cited"] is False


def test_empty_retrieval_must_abstain() -> None:
    abstained = _check(
        "No relevant passages were found in the knowledge base for this question.",
        [],
        retrieved=[],
    )
    assert abstained.passed
    confident = _check("Ingress controllers route external traffic to services.", [], retrieved=[])
    assert confident.failure_codes == ["FM-3"]


def test_an_uncited_answer_over_real_passages_fails_fm3() -> None:
    uncited = _check("Durable execution is generally a good idea for pipelines.", [])
    assert uncited.failure_codes == ["FM-3"]
    # saying the passages do not cover the question is the right answer, and needs no citation
    for abstention in (
        "The retrieved passages do not cover Kubernetes ingress controllers.",
        "Retrieved 5 relevant passages with hybrid search. Set ANTHROPIC_API_KEY to get an answer.",
    ):
        assert is_abstain_answer(abstention)
        assert _check(abstention, []).passed


# --- calibration: does an evaluator agree with the labels? ---


def _pairs(*labeled_predicted: tuple[str, str]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(lab, pred, {"trace_id": f"t{i}"}) for i, (lab, pred) in enumerate(labeled_predicted)]


def test_tpr_and_tnr_count_failures_as_the_positive_class() -> None:
    report = compute_tpr_tnr(
        _pairs(
            ("FAIL", "FAIL"),
            ("FAIL", "FAIL"),
            ("FAIL", "FAIL"),
            ("FAIL", "PASS"),  # a failure the evaluator missed
            ("PASS", "PASS"),
            ("PASS", "PASS"),
            ("PASS", "FAIL"),  # a false alarm
        )
    )
    assert (report.true_positives, report.false_negatives) == (3, 1)
    assert (report.true_negatives, report.false_positives) == (2, 1)
    assert report.tpr == pytest.approx(0.75) and report.tnr == pytest.approx(2 / 3)
    assert report.aligned is False
    assert [d["trace_id"] for d in report.disagreements] == ["t3", "t6"]

    perfect = compute_tpr_tnr(_pairs(("FAIL", "FAIL"), ("PASS", "PASS")))
    assert (perfect.tpr, perfect.tnr, perfect.aligned) == (1.0, 1.0, True)


def test_a_metric_with_no_examples_is_not_a_perfect_score() -> None:
    only_passes = compute_tpr_tnr(_pairs(("PASS", "PASS"), ("PASS", "PASS")))
    assert only_passes.tpr is None and only_passes.tnr == 1.0
    assert only_passes.aligned is False  # nothing shows it can catch a failure
    assert compute_tpr_tnr([]).aligned is False


# --- retrieval metrics ---


def test_recall_and_mrr_at_k() -> None:
    retrieved = ["a", "b", "c", "d", "e", "f"]
    assert recall_at_k(retrieved, ["c"], k=5) == 1.0
    assert recall_at_k(retrieved, ["c", "f"], k=5) == 0.5  # f is sixth: outside the top five
    assert recall_at_k(retrieved, ["zzz"], k=5) == 0.0
    assert mrr(retrieved, ["c", "a"], k=5) == 1.0  # the first relevant result is at rank 1
    assert mrr(retrieved, ["c"], k=5) == pytest.approx(1 / 3)
    assert mrr(retrieved, ["f"], k=5) == 0.0


# --- level 2: narrow binary judges ---


def test_a_verdict_puts_the_critique_before_the_verdict() -> None:
    # The model fills fields in schema order: it has to reason before it decides.
    assert list(BinaryVerdict.model_json_schema()["properties"]) == [
        "critique",
        "verdict",
        "failure_code",
    ]
    assert faithfulness_judge.output_schema is BinaryVerdict
    assert disagreement_judge.output_schema is BinaryVerdict
    assert faithfulness_judge.system_prompt != disagreement_judge.system_prompt


def test_verdicts_are_binary() -> None:
    ok = BinaryVerdict(critique="Every statement is supported by [1].", verdict="PASS")
    assert ok.failure_code == "NONE"
    for not_binary in ("MAYBE", "4/5", "pass"):
        with pytest.raises(ValidationError):
            BinaryVerdict(critique="Some reasoning here.", verdict=not_binary)  # type: ignore[arg-type]


def test_judge_input_numbers_passages_and_shows_claim_types() -> None:
    retrieved = [
        _passage(1, claim_text="Retries add 40 ms", claim_type="measurement"),
        _passage(2, author=None),
        _passage(5),  # a candidate that never reached the prompt
    ]
    text = format_judge_input(
        question="How much do retries cost?", retrieved=retrieved, answer_text="About 40 ms [1]."
    )
    assert text.startswith("Question: How much do retries cost?")
    assert "[1] Author: Jane Writer | Title: A post | Claim type: measurement" in text
    assert "Claim: Retries add 40 ms" in text and "Text: text of passage 1" in text
    assert "[2] Author: Unknown | Title: A post | Claim type: none" in text
    assert "text of passage 5" not in text
    assert text.endswith("Answer:\nAbout 40 ms [1].")


def test_judge_input_says_when_nothing_was_retrieved() -> None:
    text = format_judge_input(question="Anything?", retrieved=[], answer_text="No idea.")
    assert "(no passages were retrieved)" in text


def test_disagreement_is_only_judged_when_two_authors_were_in_the_prompt() -> None:
    same_author = [_passage(1), _passage(2)]
    assert requires_disagreement_check(same_author) is False
    two_authors = [_passage(1), _passage(2, author="Sam Critic")]
    assert requires_disagreement_check(two_authors) is True
    # a second author who was only a candidate, never shown to the model, does not count
    only_candidate = [_passage(1), _passage(6, author="Sam Critic")]
    assert requires_disagreement_check(only_candidate) is False
    assert requires_disagreement_check([_passage(1), _passage(2, author=None)]) is False
