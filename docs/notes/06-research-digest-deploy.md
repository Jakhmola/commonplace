# 06 - A research agent with brakes, a weekly digest, and a deployable image

## Goal

`/ask` does one retrieval and answers from five passages. Some questions do not fit
that: "how do these authors differ on X, and what evidence does each give?" needs
several searches, and what to search next depends on what the first search found.
That is a loop a model steers, and a loop a model steers needs limits the model
cannot move. This phase builds that loop, a weekly digest, and the packaging to
run the whole thing as a container.

## The research agent

Three roles, each one model call with one job, on the agent layer from phase 4
(no framework):

| Role | Output | Job |
|---|---|---|
| Planner | `{rationale, subqueries, target_topics}` | Turn the question into one to three searches |
| Coverage critic | `{rationale, is_sufficient, missing_aspects, followup_queries}` | Read what was found: enough, or what to search next? |
| Synthesizer | text with `[n]` citations | Write the report from the best of the evidence |

`run-research-agent` is the loop, written directly on Inngest steps:

```
record-research-start
plan-subqueries                      planner
hop-1-retrieve                       each search through the three-way retrieval
hop-1-reflect                        critic
hop-2-retrieve / hop-2-reflect ...   while no bound says stop
synthesize-report                    from the 12 best passages of the pool
score-deterministic, judge-*         the same evaluators as any answer
persist-research-run, emit
```

Each hop's passages are merged into one pool, one entry per passage with its best
score, so the same passage found by three searches counts once.

### The brakes

Whether the loop goes on is decided in plain code
(`stop_reason_after_hop`, `fresh_queries` in `agents/roles/researcher.py`):

| Stop reason | When |
|---|---|
| `sufficient_coverage` | the critic says the evidence is enough |
| `converged` | a hop added no passage that was not already in the pool |
| `no_followup_queries` | everything the critic proposed had already been searched |
| `max_iterations_reached` | the cap on hops (default 3, a request may ask for 1 to 5) |
| `budget_exceeded` | the cap on dollars (default 0.75) |

The last four do not depend on the model's judgement, and the checks run before
the critic is asked, so a run that has to stop does not pay for one more opinion.
Every model call is its own step: a retry never pays twice, and the cost of each
call is read from its result and added up.

**What the dollar cap does and does not bound.** It bounds planning and
reflection, the part that can run away. Once it is reached nothing more is
searched; the report is still written from what was gathered and still scored,
which adds roughly $0.15. So a run can cost more than `max_usd`; its status is
then `budget_exceeded` and both numbers are stored (`total_cost_usd`, and
`search_usd` in the evaluation summary).

### Asymmetry again

`POST /research` returns a `research_id` in milliseconds; `GET /research/{id}` is a
plain read. The API writes the run's row as `queued` right after sending the
event, so the very first poll finds it. A run that fails for good is marked
`failed` by an `on_failure` handler instead of reading as `running` forever.

## The weekly digest

`generate-weekly-digest` runs on a cron (Mondays 08:00 UTC) and on request:

1. Collect the articles published in the last seven complete days and the three
   most confident current claims of each.
2. A model writes a briefing from those numbered claims: what was measured, what
   authors argued, what several publications wrote about, predictions.
3. Code checks that every `[n]` is a claim it was given, and stores one row per
   window (the id is derived from the window, so a second run replaces the first).

It is written from claims, not article text: each claim already has a verbatim
quote that was checked against its article, so every number in the digest leads
to something verifiable through `/sources/{claim_id}`. When nothing was published
in the window it falls back to the fifteen most recent articles and says so.

## Production packaging

- `Dockerfile`: dependencies from the lockfile, migrations before the server
  starts, runs as a non-root user.
- `infra/docker-compose.prod.yml`: the app, Postgres and Memgraph with health
  checks. It refuses to start without the database password, both Inngest keys,
  the OpenAI key and an API token. The databases are not published to the host.
- `INNGEST_DEV=0` switches from the local dev server to Inngest Cloud: the keys
  are passed to the client, and `make up` leaves the dev server out.
- `API_TOKEN`: when set, every route except `/health` and `/api/inngest` requires
  a bearer token. This was not in my plan. I added it when I saw that the deployed
  API would otherwise let anyone who can reach it start paid research runs.

## Interfaces

| | |
|---|---|
| `POST /research`, `GET /research/{id}` | start a run, read it |
| `POST /digests/run`, `GET /digests/latest` | write the digest now, read the latest |
| MCP `cp_research`, `cp_research_result`, `cp_weekly_digest` | the same, for an agent (13 tools in total) |
| Events | `cp/research.requested`, `cp/research.completed`, `cp/digest.requested`, `cp/digest.published` |
| Tables | `research_runs`, `digests` (`alembic/versions/0006_research_and_digests.py`) |

## What happened when I ran it

Three research runs against the live corpus (60 articles):

| Question | Bounds | Stopped because | Hops | Cost | Confidence |
|---|---|---|---|---|---|
| How do the authors differ on whether open-weight models are catching up, and what evidence does each give? | defaults | `max_iterations_reached` | 3 | $0.23 | 0.0 |
| Trade-offs between LLM judges and verifiers | `max_usd` 0.02 | `budget_exceeded` | 1 | $0.18 | 1.0 |
| What do the authors say about evaluating coding agents? (via MCP) | `max_iterations` 1 | `max_iterations_reached` | 1 | $0.17 | 1.0 |

Each took 45 to 65 seconds.

The first run is the interesting one. The critic noticed after hop 1 that nearly
all the evidence came from one author, asked for the others by name, and after
hop 2 said the same again; the cap ended it. The report says plainly that the
passages do not show the authors disagreeing and lists what is not covered. It
has about thirty cited statements. The faithfulness judge confirmed all but one:
a sentence that names "US open-weight efforts" as the subject of a passage that
begins mid-sentence with "they". That is exactly the rule added to the judge in
the previous phase, and because the verdict is binary, one guessed subject among
thirty statements gives the report a confidence of 0. The judge is right about
the sentence and the score is too harsh for a long report. I tightened the report
prompt on that point; the scoring is left as it is and noted below.

The digest for 2026-11-15 to 2026-11-21: 10 articles, 30 claims, 27 cited, all
citations valid, $0.04. Running it again left one row for the window.

## Problems I hit

1. **The first poll after `POST /research` was a 404**, because the row was written
   by the function's first step. Now the API writes it as `queued`, and the row
   only ever moves forward (`queued` to `running`), whichever side writes first.
2. **The evaluation code lived inside `evaluate-answer`.** Scoring a report would
   have meant a second copy. It is now one helper used by both.
3. **The models write longer than asked**: about 600 words where the prompt says
   500, and 550 for the digest where it says 400. Not fixed.

## Verification

| Check | Result |
|---|---|
| `make check` | pass, 221 tests |
| Loop bounds (fake step, fake model) | each of the five stop reasons reached; no model call after a hop that adds nothing |
| `POST /research` defaults | 3 hops, `max_iterations_reached`, report with 12 citations from 9 articles |
| `max_usd` 0.02 | stops after hop 1, status `budget_exceeded`, report still written |
| `max_iterations` 6, unknown id | 422, 404 |
| `POST /digests/run` twice | one row for the window, `citations_valid: true` |
| MCP over stdio | 13 tools; `cp_research` returns a handle, `cp_research_result` reads it, bad input is a tool error |
| Production compose | refuses to start without secrets; image builds; stack healthy with placeholder keys; 401 without the token, 200 with it; migrations applied; runs as non-root; unsigned requests to `/api/inngest` rejected |

**Not verified:** an actual sync with Inngest Cloud. I had no cloud keys, so cloud
mode is tested up to the point where the app starts in production mode and
rejects unsigned requests. The `trace_url` returned by the API is also only
meaningful for the local dev server.

## What I would do next

- **Cap passages per article in a prompt.** The benchmark shows cross-publication
  questions are where retrieval is weakest, because five passages often come from
  one or two articles. This would help `/ask`, the research agent and the
  benchmark at once.
- **Score reports per statement, not as a whole.** One unsupported sentence in
  thirty should lower confidence, not zero it.
- **A held-out set of labelled traces**, labelled by a person, and several judge
  calls per trace, before trusting the judges' numbers.
- **More sources**: books and PDFs go through the same ingest pipeline once they
  have a parser; the claim graph and the evaluators do not care where text came from.
