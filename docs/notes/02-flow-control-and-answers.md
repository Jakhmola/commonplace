# 02 - Three feeds, flow control, hybrid search, answers with citations

## Goal

Go from one newsletter and five posts to three newsletters and their whole feeds,
without hammering anyone's server or my embedding quota, and then ask questions of
the result and get answers I can check.

## What goes through Inngest, and what does not

Writes are events. `POST /publications` and `POST /articles` send an event and
return a handle; the work happens in durable functions where a crash or a provider
outage costs nothing.

Reads are not. `GET /search` and `POST /ask` talk to Postgres (and Claude) directly.
Someone is waiting for them, and a retry in three minutes is no use to that person.

```
POST /publications --> cp/publication.added --> backfill-publication
                                                  match-allowlist
                                                  fetch-feed          (entry pointers only)
                                                  upsert-publication
                                                  find-new-articles   (skip stored, record job)
                                                  emit-<n>            (batches of 25)
                                                       |
POST /articles  ----- via=manual ----------------------+--> cp/article.discovered
                                                              |
                                                              v
                                                        ingest-article
                                                        concurrency 3 per publication
                                                        throttle 50 run starts / minute
                                                        priority: manual +600 s

POST /ask --> hybrid search --> Claude --> snapshot row --> cp/answer.produced
```

## What I built

| Piece | Where |
|---|---|
| Backfill of a whole feed (replaces the five-newest discovery) | `commonplace/functions/backfill.py` |
| Concurrency, throttle and priority on ingest | `commonplace/functions/ingest.py` |
| Manual article ingestion, `POST /articles` | `commonplace/main.py` |
| Reciprocal rank fusion and hybrid search | `commonplace/retrieval/hybrid.py` |
| `POST /ask`: retrieve, answer, cite, snapshot | `commonplace/functions/ask.py` |
| `snapshots` table | `commonplace/db/models.py`, `alembic/versions/0002_snapshots.py` |
| Replay check | `scripts/replay_backfill.py` |

### Flow control

All three knobs are configuration on `ingest-article`. Inngest enforces them in its
queue, before any of my code runs.

- **Concurrency, keyed by publication** (`CONCURRENCY_PER_PUBLICATION=3`). Never more
  than three requests in flight against one site, and one big feed cannot starve
  the others.
- **Throttle, global** (`INGEST_RUNS_PER_MINUTE=50`). No key, because the embedding
  quota is shared by every publication. It limits run starts; each run embeds its
  article in about one request, so this is close to "embedding requests per minute".
- **Priority.** `event.data.via == 'manual' ? 600 : 0`. Priority is a time shift in
  seconds, not a rank: a manual run is queued as if it had arrived ten minutes
  earlier, so it overtakes whatever was queued in the last ten minutes.

There is still no idempotency key. It would refuse a re-send of the same article
for 24 hours, and re-sending is how a failed article gets retried.

### Hybrid search

Full-text search finds exact tokens and misses paraphrases; vector search does the
opposite. Reciprocal rank fusion merges the two lists using positions only:

```
score(passage) = sum over the engines that returned it of 1 / (60 + rank)
```

Scores from the two engines live on different scales and are never added. A passage
both engines found collects points twice, so agreement beats being one engine's
favourite. Ties break on passage id so the same query always returns the same order.
Each engine is asked for 15 candidates; every result says where it came from
(`bm25`, `vector` or `both`) and its rank in each engine.

### Answers

`/ask` puts the top five passages into the prompt as numbered blocks and asks
Claude to answer from them only, citing `[n]` inline. Afterwards:

- Only passages whose number appears in the answer become citations.
- `as_of` is the publication date of the newest cited source, so I can see how
  fresh an answer is.
- A snapshot row stores the question, the answer, the citations and **all 15
  candidates** with their ranks, not just the cited ones. With only the cited
  passages there is no telling "the right passage was never retrieved" from "it was
  retrieved and ignored", and those need different fixes.
- The snapshot is committed first. Then `cp/answer.produced` is sent in the
  background; the caller does not wait for it and a failure to send is only logged.

Without `ANTHROPIC_API_KEY` the endpoint still answers 200: the passages come back
as citations with a note. If the Claude call fails, the failure becomes the answer
text and the snapshot is still stored.

## Problems I hit (and fixed)

1. **A negative priority is a delay, not a low rank.** My first expression gave
   backfill runs `-600`. The three backfills queued 60 runs and nothing happened:
   every run sat `QUEUED` for exactly 600 seconds, and then every *step* of every
   run waited another 600. An hour for a job that takes 26 seconds. Non-manual
   runs now get `0`; manual runs keep `+600`, which is all the queue-jump needs.
2. **The feed step was carrying whole posts.** `fetch-feed` returned the raw feed,
   and feeds embed full post bodies: one of mine was 2.9 MB against Inngest's 4 MB
   step output cap. The step now returns title, link, author and date per entry.
3. **A failed cost write re-ran the paid call.** An exception in the cost-capture
   middleware made Inngest discard the step's output and retry `embed`. Recording a
   cost can no longer fail the step.
4. **Redirects could leave the allowlist.** The article URL was checked, the page it
   redirected to was not. Every hop is now checked before it is requested.
5. **One odd feed entry could sink a backfill.** An entry linking to the homepage has
   no post slug and raised outside any step. Such entries are skipped, and two links
   ending in the same slug count once.
6. **Smaller ones:** a throttled `robots.txt` (429) was cached as "allow all"; the
   page's date-only value was overwriting the feed's full timestamp; a very long
   unbroken token produced a run of redundant chunks; `/jobs` reported `Running`
   and `Cancelled` in Inngest's casing and 404'd in the moment before a run started.

## Verification

| Check | Result |
|---|---|
| `make check` | pass, 45 unit tests |
| Backfill three feeds at once | 60 articles, 1,529 passages in 26 s; all three jobs `completed` 20/20 |
| Embedding cost of the full corpus | 336k tokens, **$0.0067** |
| Replay: `--verify`, `--all`, `--verify` | identical: `publications=3 articles=60 passages=1529 embedding_calls=60` |
| Queue jump (concurrency 1, 10 runs/min) | manual article queued 21st of 21, finished 2nd |
| `POST /articles` outside the allowlist | 400 |
| `GET /search` (hybrid) | results carry `source` = `both`, `vector`, `bm25` with per-engine ranks |
| `POST /ask` | cited answer in about 6 s, `snapshot_id` returned, row has 15 candidates, `cp/answer.produced` received |
| `POST /ask` without `ANTHROPIC_API_KEY` | 200, five passages as citations |

## Next

Keep the corpus fresh without re-reading unchanged feeds (conditional requests on a
schedule), supersede passages when an article is edited instead of overwriting them,
record runs that fail for good, and expose the whole thing to Claude as MCP tools.
