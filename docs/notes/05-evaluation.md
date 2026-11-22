# 05 - Knowing when it is wrong

## Goal

The system answers questions with citations. Until now, the only evidence that the
answers were good was me reading a few. This phase replaces that with something
that runs on every answer and produces numbers I can watch.

## Start from the failures, not from metrics

A generic "quality score" tells nobody what to fix. So the first artefact is a list
of the specific ways an answer from this system can be wrong
(`evals/taxonomy.yaml`), made by reading real traces and grouping what looked off:

| Code | Failure | Caught by |
|---|---|---|
| FM-1 | A citation that points nowhere: a number that was not in the prompt, or a passage that was not retrieved | code |
| FM-2 | A cited passage or claim that has since been superseded | code |
| FM-3 | Nothing relevant was retrieved and the answer responds anyway from general knowledge | code, then the faithfulness judge |
| FM-4 | A statement its cited passage does not support: a wrong number, a wrong attribution, an opinion reported as a measurement | faithfulness judge |
| FM-5 | The sources show the question is contested and the answer gives one side as settled | disagreement judge |
| FM-6 | The right article exists but was not retrieved | retrieval metrics on the benchmark |

Each failure gets the cheapest evaluator that can catch it.

## What I built

| Piece | Where |
|---|---|
| Failure taxonomy | `evals/taxonomy.yaml` |
| Level 1: checks in code | `commonplace/evals/deterministic.py` |
| Level 2: two binary judges | `commonplace/evals/judges.py`, prompts in `commonplace/agents/prompts.py` |
| Calibration (TPR / TNR against labels) | `commonplace/evals/calibration.py`, `metrics.py`, `scripts/calibrate_judges.py` |
| Labelled traces | `evals/labeled_traces.jsonl`, exported with `scripts/export_traces.py` |
| Online scoring of every answer | `evaluate-answer` in `commonplace/functions/evals.py` |
| Golden benchmark and variant suite | `evals/golden_benchmark.yaml`, `commonplace/evals/benchmark.py`, `run-eval-suite` |
| `eval_results`, `eval_runs` | `alembic/versions/0005_eval_tables.py` |
| Report | `GET /evals/latest`, `POST /evals/run`, MCP `cp_eval_report` |

### Level 1: code

Four assertions over a snapshot, free and instant, run on every answer:
citations are in bounds and listed; each citation points at the retrieved passage
with that number; nothing cited is superseded; and the answer either cites or says
the passages do not cover the question. Two small changes made this checkable:
snapshots now mark which candidates actually reached the prompt, and the answer
prompt has one fixed way to abstain.

### Level 2: judges

Two judges, both on the agent layer from the previous phase:

- **Binary.** `PASS` or `FAIL`. A 1-5 scale invites a "3" and tells nobody what to fix.
- **Critique first.** The schema puts the reasoning before the verdict, so the model
  goes through the evidence before it commits.
- **One job each.** Faithfulness and disagreement are separate judges with separate
  prompts. The disagreement judge only runs when two authors were in the prompt.
- **Same view as the answerer.** A judge sees exactly the passages the model saw,
  numbered the same way, with claim types.

Without an API key the judges are recorded as `SKIPPED`. There is no rule-based
stand-in: a keyword heuristic called a judge would produce numbers that mean nothing.

### Online scoring

Every `/ask` already emitted `cp/answer.produced`. `evaluate-answer` listens to it:
checks in code, faithfulness judge, disagreement judge if applicable, then one
`eval_results` row and a composite score written to `snapshots.confidence`
(0 if a check or faithfulness failed, 0.5 if only disagreement did, 1 otherwise).
The person who asked never waits for any of it. Each judge is its own step, so a
retry does not pay for the first judge twice, and judge calls show up in `costs`.

### The benchmark

25 questions written from this corpus, in four categories, each naming the
article(s) that answer it. `run-eval-suite` puts every question to four retrieval
variants **against the live corpus** and scores recall@5 and reciprocal rank per
article (three passages of one long post count once).

## Results

### Retrieval variants (25 questions, k = 5)

| Variant | Recall@5 | MRR | Keyword | Identifier | Paraphrase | Cross-publication |
|---|---|---|---|---|---|---|
| `sparse_bm25` | 0.82 | 0.77 | 1.00 | 1.00 | 0.62 | 0.70 |
| `dense_pgvector` | 0.96 | 0.94 | 1.00 | 1.00 | 1.00 | 0.80 |
| `hybrid_2way_rrf_k60` | 0.96 | 0.87 | 1.00 | 1.00 | 0.94 | 0.90 |
| `hybrid_3way_rrf_k60` | 0.94 | 0.89 | 1.00 | 1.00 | 0.94 | 0.80 |

(Category columns are recall@5.)

What this says, without dressing it up:

- **Keyword search alone is the weak one**, and exactly where expected: paraphrased
  questions (0.62).
- **Vector search alone is as good as either hybrid on this benchmark**, and ranks
  the right article higher (MRR 0.94). Fusing in the keyword ranking pulls some
  right answers down a place.
- **The graph leg did not help here.** Three-way is no better than two-way, and on
  the five cross-publication cases, the ones the graph was added for, it found one
  expected article fewer.

The honest reading is that on a 60-article corpus, with questions I wrote, there is
not enough headroom for hybrid retrieval or the graph to show a benefit, and 25
cases cannot separate variants that differ by one case. The graph still earns its
place for what it stores (claims, types, sources, supersession), but its value as a
retriever is not demonstrated. That is worth knowing, and it is the kind of thing
this suite exists to tell me.

### Calibration of the evaluators (21 labelled traces)

| Evaluator | Traces | TPR (failures caught) | TNR (good answers cleared) |
|---|---|---|---|
| Checks in code | 21 | 1.00 (5 of 5) | 1.00 (16 of 16) |
| Faithfulness judge | 19 | 1.00 (8 of 8) | 0.91 (10 of 11) |
| Disagreement judge | 6 | 1.00 (2 of 2) | 1.00 (4 of 4) |

All three are above the 0.90 thresholds. These numbers come with caveats that
matter more than the numbers:

- **The first run was worse, and that was the point of running it.** With the
  original prompts the faithfulness judge caught 7 of 8 failures and the
  disagreement judge 0 of 2. The disagreement prompt defined a conflict as one
  between two authors of the passages, while my failing examples had one author
  reporting another party's opposing position. I widened the definition (in the
  prompt and in the taxonomy) and told the faithfulness judge not to accept a name
  the passage leaves unnamed. The table is the second run.
- **The prompts were tuned on the same traces they are measured on.** There is no
  held-out set, so the table is optimistic.
- **The sample is tiny.** Two failing examples for the disagreement judge say
  almost nothing about its real miss rate.
- **I labelled the traces.** Ten are real answers that I read against their
  passages statement by statement; eleven are failures made by changing a real
  trace on purpose (the file says which, and what was changed). They are not human
  labels until someone reviews them.
- **The judge and the answerer are the same model**, which is known to be lenient
  towards its own output. A different judge model can be set with `EVAL_JUDGE_MODEL`.

The remaining disagreement is instructive. On one real answer the judge says FAIL
where I said PASS: the answer cites passage [4] for a sentence whose name and title
are actually in passage [3], and credits a blog post with a repositioning the
passage credits to the company. I read that as acceptable because the information
was in the prompt; the judge read it as a mis-citation. I left my label alone
rather than move it to agree with the judge.

One of the ten real answers was a genuine failure (it named a product where the
cited excerpt, cut off mid-sentence, never did). A 10% failure rate on a ten-answer
sample is not a statistic, but it is a reminder that "reads fine" is not evidence.

### Online, so far

12 answers scored: every one passed the checks in code, 11 of 12 passed the
faithfulness judge, mean confidence 0.92.

## Problems I hit (and fixed)

1. **The provider refused a judge call** on a trace about cyber security. My code
   treated a refusal as fatal, which would have aborted a calibration run and left
   production answers unscored after retries. A refused judge is now recorded as
   skipped in production and counted against the judge in calibration.
2. **A circular import that only showed when a script started cold.** The checks
   imported citation parsing from the answering code, whose package imports the
   functions that import the checks. Tests passed because of import order. The
   parser now lives in its own module, and a test imports the evaluators in a fresh
   interpreter.
3. From the review of the previous phase: concurrency keys do not serialise runs
   (replaced with singletons), a failed re-extraction left stale claims current,
   stored quotes were cut from a normalised copy of the article, a hung Memgraph
   blocked reads for a minute, and multi-word topics were lost when seeding the
   graph from a question. All fixed.

## Verification

| Check | Result |
|---|---|
| `make check` | pass, 170 tests |
| `calibrate_judges.py --mode offline` | checks in code aligned, exit 0 |
| `calibrate_judges.py --mode live` | all three evaluators aligned (table above), exit 0 |
| `POST /ask` | `evaluate-answer` ran; `eval_results` row written; `snapshots.confidence` went from NULL to 1.0; `cp/answer.evaluated` sent |
| Snapshot that does not exist | `status: skipped_missing_snapshot` |
| `POST /evals/run` | four variants scored against the live corpus, stored in `eval_runs`, no expected article missing |
| `GET /evals/latest`, `cp_eval_report` | pass rates, failure-mode counts, recent failures with the judge's reason, latest variant comparison |

## Next

A research agent that plans several searches for a hard question and knows when to
stop, a weekly digest of what was ingested, and a production deployment.
