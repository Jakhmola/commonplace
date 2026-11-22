"""Ask real questions and print each answer as a trace, one JSON object per line.

    uv run python scripts/export_traces.py "What is RoPE?" "How do judges work?" > traces.jsonl

A trace is what an evaluator needs to judge an answer: the question, exactly
the passages the model was shown, the answer and its citations. This is the
raw material for `evals/labeled_traces.jsonl`: read each trace against its
passages, add `expected_verdict`, `failure_code` and a `note`, and it becomes
a calibration example.

Each question is answered through the normal path, so it costs one model call
and leaves a snapshot behind like any other question.
"""

import asyncio
import json
import sys

from commonplace import operations
from commonplace.db.queries import get_snapshot
from commonplace.db.session import engine, get_session
from commonplace.evals.deterministic import prompt_passages

_PASSAGE_FIELDS = (
    "passage_id",
    "title",
    "author",
    "url",
    "published_at",
    "text",
    "claim_id",
    "claim_text",
    "claim_type",
)


async def main(questions: list[str]) -> None:
    for number, question in enumerate(questions, start=1):
        answer = await operations.ask(question)
        async with get_session() as session:
            snapshot = await get_snapshot(session, answer["snapshot_id"])
        assert snapshot is not None
        shown = prompt_passages(snapshot.retrieved)
        trace = {
            "trace_id": f"real-{number:02d}",
            "origin": "real",
            "snapshot_id": snapshot.snapshot_id,
            "question": question,
            "retrieved": [
                {**{f: p.get(f) for f in _PASSAGE_FIELDS}, "in_prompt": True} for p in shown
            ],
            "citations": snapshot.citations,
            "answer_text": snapshot.answer_text,
        }
        print(json.dumps(trace, ensure_ascii=False))
    # let the cp/answer.produced announcements go out before the loop closes
    await asyncio.sleep(1)
    await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1:]))
