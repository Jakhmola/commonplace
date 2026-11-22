# 01 - Durable ingest and a first look at search

## Goal

Take one newsletter (Latent Space), ingest its five newest posts into Postgres in a
way that survives a crash halfway through, and search the result two ways.

## Plan

```
POST /publications ──> cp/publication.added ──> add-publication ──> cp/article.discovered (x5)
                                                 fetch-feed                      │
                                                 save-publication                ▼
                                                 find-new                  ingest-article
                                                 fan-out                   check-allowlist
                                                                           fetch-html
                                                                           parse
                                                                           chunk
                                                                           embed      (the paid call)
                                                                           persist
                                                                           emit ──> cp/article.ingested
```

The API never does slow work. It sends an event and hands back the event id;
`GET /jobs/{event_id}` reports progress. Every step result is recorded by Inngest,
so a retry replays the function but finished steps return instantly instead of
running again. That is the whole point of splitting ingest into seven steps:
a failure in `embed` must never re-fetch or re-parse the page.

## What I built

| Piece | Where |
|---|---|
| URL canonicalisation (`host/path`, no www, query, fragment, trailing slash) | `commonplace/sources/urls.py` |
| Allowlist of feeds and hosts | `publications.yaml`, `commonplace/sources/allowlist.py` |
| Polite fetch (User-Agent, robots.txt, timeout, 5 MB cap) | `commonplace/sources/fetch.py` |
| Feed parsing (pointer only, never trusts feed content) | `commonplace/sources/rss.py` |
| Article extraction (trafilatura) and content hash | `commonplace/sources/parse.py` |
| Chunking: 1000 chars, 150 overlap, whitespace cuts only | `commonplace/sources/chunk.py` |
| Embeddings (text-embedding-3-small, 1536 d) | `commonplace/sources/embed.py` |
| Schema: publications, articles, passages, jobs, costs | `alembic/versions/0001_core.py` |
| All SQL | `commonplace/db.py` |
| Inngest functions | `commonplace/functions/` |
| API: `/publications`, `/jobs/{id}`, `/search`, `/health` | `commonplace/main.py` |

Decisions worth remembering:

- **The dedupe key is the cleaned URL** (`latent.space/p/formal-verification`), not a
  per-publication slug. It needs no lookup, and the same shape works for local files
  later (`file:<sha256>`).
- **Jobs count, they don't track.** A job row stores the canonical ids it expects;
  progress is `count(articles WHERE canonical_id = ANY(job.ids))` at read time. No
  counter to keep in sync, no read-repair.
- **Cost is logged in the same transaction as the passages.** A memoized `embed` step
  is never re-run on replay, and `persist` commits once, so each real embedding call
  is logged exactly once.
- **No Inngest idempotency key.** Duplicates are already impossible (discovery skips
  stored ids, every write is an upsert), and an idempotency key would stop me
  re-sending an article on purpose.
- **Passages keep the original text.** Search vectors are a generated `tsvector`
  column with a GIN index; embeddings use an HNSW cosine index.

## Problems I hit (and fixed)

1. **Inngest could not reach the app on Linux.** The dev server runs in Docker and
   calls back to `host.docker.internal:8000`, but uvicorn binds `127.0.0.1`, so the
   callback never arrives. Instead of exposing the API on every interface, the
   Inngest container now uses `network_mode: host` and calls `localhost:8000`.
2. **Step output size.** The longest post (97k chars) became 114 passages. As JSON
   floats, the `embed` step's output was 3.8 MB, right under Inngest's 4 MB step
   limit. Vectors now cross step boundaries as base64 float32 (8 KB per vector instead
   of 33 KB, bit-exact for what OpenAI returns). Ceiling is now about 490 passages,
   roughly a 490k-char article.
3. **Messy titles and bylines from page metadata.** Page titles came back with
   the site name appended and bylines with a trailing `·`. The feed's own `title` and
   `dc:creator` are clean, so those win and page metadata is only a fallback.
4. **Feed links carried `?utm_source`.** Caught by the feed parser test; links are
   cleaned before they are fetched.
5. **`/jobs` leaked a stack trace with local paths** on failure. It now returns only
   the error message.
6. **`uvicorn --reload` leaves an orphan.** Killing the reloader with SIGKILL leaves its
   worker alive and still serving. Ctrl+C in the terminal is fine; a hard kill has to
   target the process that owns the port.

## Verification

| Check | Result |
|---|---|
| `make check` (ruff, format, mypy, pytest) | pass, 11 unit tests |
| `GET /health` | `database: ok, inngest: ok` |
| `POST /publications` (Latent Space) then `GET /jobs/{id}` | `completed`, 5/5 articles, 234 passages |
| Same publication again | `queued: 0`, counts unchanged, no new embedding calls |
| Feed not on the allowlist | job `failed` immediately with a clear message, no retries |
| Provider outage (bad OpenAI key) | `embed` failed twice; after fixing the key the third attempt passed. `check-allowlist`, `fetch-html`, `parse`, `chunk` ran once each |
| Real crash (SIGKILL during `embed` of the 114-passage post) | after restart only `embed`, `persist`, `emit` ran; counts back to 5 / 234 |

Cost of the whole phase, including every re-ingest above: 8 embedding calls,
110k tokens, **$0.0022**.

## Eval: sparse vs dense

Ten hand-written questions over the five posts, each with one article that answers it
(`evals/queries_v1.yaml`). Half reuse the article's words (`keyword`), half say the
same thing differently (`paraphrase`). Metric: does any of the top 5 passages come
from the right article (hit@5), and how high (MRR).

```
uv run python scripts/eval_retrieval.py
```

| mode | all hit@5 | all MRR | keyword hit@5 | keyword MRR | paraphrase hit@5 | paraphrase MRR |
|---|---|---|---|---|---|---|
| sparse | 0.40 | 0.35 | 0.80 | 0.70 | 0.00 | 0.00 |
| dense | 0.80 | 0.68 | 0.80 | 0.80 | 0.80 | 0.55 |

| query | sparse | dense |
|---|---|---|
| SMT solver used in production pipelines at Amazon | 1 | - |
| how can engineers keep AI generated code from being low quality | - | 2 |
| Lean proof of the Riemann Hypothesis | 2 | 1 |
| famous unsolved math conjecture cracked with help from an AI model | - | 1 |
| largest IPO in history beating SpaceX | - | 1 |
| founders keep majority voting control after the company goes public | - | 1 |
| why Linear never hired an SDR | 1 | 1 |
| how much of the work in a project tracker is now created by AI agents | - | 4 |
| Gemini 4 Neon price per million tokens | 1 | 1 |
| Google mid-sized model named after a chemical element | - | - |

Latency for one query: sparse 2 ms, dense 770 ms (almost all of it the query
embedding round trip to OpenAI).

What it says:

- **Sparse is all or nothing.** `websearch_to_tsquery` ANDs every term, and a passage
  must contain all of them. "largest IPO in history beating SpaceX" matches 0 passages,
  although each word is in the article; the same terms OR'ed match 24. Every paraphrase
  scores zero for the same reason. This is not BM25 either: `ts_rank_cd` has no IDF.
- **Dense misses exact identifiers.** "SMT solver" is a precise term that only sparse
  finds. Dense wins on meaning, sparse wins on rare exact tokens; neither alone is
  enough, which is the case for fusing them next.
- **Both missed** "chemical element": the article says "Neon (10)" and "atomic weights",
  and no passage is close enough. A fair miss.

## Next

Three feeds with flow control (concurrency per publication, a global throttle,
priority for manual adds), hybrid search with reciprocal rank fusion, an OR-style
sparse query, and `/ask` with citations.
