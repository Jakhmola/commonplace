# 01 - Durable ingest and a first look at search

## Goal

Take one newsletter (Latent Space), ingest its five newest posts into Postgres in a
way that survives a crash halfway through, and search the result two ways.

## Plan

```
POST /publications --> cp/publication.added --> add-publication --> cp/article.discovered (x5)
                                                 match-allowlist                 |
                                                 fetch-feed                      v
                                                 upsert-publication        ingest-article
                                                 discover                  check-allowlist
                                                 emit                      fetch-html
                                                                           parse
                                                                           chunk
                                                                           embed      (the paid call)
                                                                           persist
                                                                           emit --> cp/article.ingested
```

The API never does slow work. It sends an event and hands back the event id;
`GET /jobs/{event_id}` reports progress. Every step result is recorded by Inngest,
so a retry replays the function but finished steps return instantly instead of
running again. That is the whole point of splitting ingest into seven steps:
a failure in `embed` must never re-fetch or re-parse the page.

## What I built

| Piece | Where |
|---|---|
| URL normalisation and canonical ids (`pub:<publication>:<post-slug>`) | `commonplace/schemas/ids.py` |
| Event names and payloads | `commonplace/schemas/events.py` |
| Allowlist of feeds and hosts | `publications.yaml`, `commonplace/sources/allowlist.py` |
| Polite fetch (User-Agent, robots.txt, timeout, 5 MB cap) | `commonplace/sources/fetcher.py` |
| Feed parsing (pointer only, never trusts feed content) | `commonplace/sources/rss.py` |
| Article extraction (trafilatura) and content hash | `commonplace/sources/parser.py` |
| Chunking: 1000 chars, 150 overlap, whitespace cuts only | `commonplace/sources/chunking.py` |
| Embeddings (text-embedding-3-small, 1536 d), packed vectors | `commonplace/sources/embedder.py` |
| Tables (SQLAlchemy models) and migration | `commonplace/db/models.py`, `alembic/versions/0001_core.py` |
| Every query | `commonplace/db/queries.py` |
| Sparse and dense retrievers | `commonplace/retrieval/` |
| Inngest client and cost capture | `commonplace/inngest_client.py` |
| Inngest functions | `commonplace/functions/` |
| API: `/publications`, `/jobs/{id}`, `/search`, `/health` | `commonplace/main.py`, `commonplace/health.py` |

Tables: `publications`, `articles`, `passages`, `cursors`, `jobs`, `costs`.

Decisions worth remembering:

- **The dedupe key is `pub:<publication>:<post-slug>`.** The publication comes from
  the allowlist entry, not the URL's host, so a newsletter that lives on two domains
  (Latent Space serves `www.latent.space` and links to `latent.space`) still gets one
  id per post.
- **The allowlist matches hosts exactly.** No wildcards and no `www.` stripping, so a
  lookalike host or an unlisted subdomain is refused. A second domain has to be
  listed on purpose (`archive_urls`).
- **Jobs count, they don't track.** `ingest-article` never learns which job asked for
  it, so nothing increments a counter. A job row stores the canonical ids it expects
  and `done` is how many of them exist, recounted when the job is read.
- **Cost capture is a middleware, not a call site concern.** Any step that returns a
  dict with a `usage` key gets one `costs` row. A memoized step is never re-run on
  replay, so its output never passes through the middleware twice.
- **Passages carry `valid_from` / `valid_to`.** Nothing sets `valid_to` yet; searches
  already filter on it so that superseding an edited article later needs no query
  changes.
- **No Inngest idempotency key.** Duplicates are already impossible (discovery skips
  stored ids, every write is an upsert), and an idempotency key would stop me
  re-sending an article on purpose.
- **At most 3 articles per publication at a time** (concurrency keyed on
  `publication_id`), to stay polite to one server while other feeds keep moving.

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

7. **A retrying run looked failed.** Between retries the dev server's REST API labels
   a run `Failed` with no end time. `/jobs` now reports a failure only once the run
   has actually ended.
8. **The feed links to a host the allowlist did not have.** Latent Space's feed is on
   `www.latent.space` but its article links use `latent.space`. Exact host matching
   refused them, correctly; the bare domain is now listed under `archive_urls`.

## Verification

| Check | Result |
|---|---|
| `make check` (ruff, format, mypy, pytest) | pass, 21 unit tests |
| `alembic check` | models and migration agree |
| `GET /health` | `database: ok, inngest: ok` |
| `POST /publications` (Latent Space) then `GET /jobs/{id}` | `completed`, 5/5 articles, 234 passages, 5 `costs` rows |
| Same publication again | `emitted: 0`, counts unchanged, no new embedding calls |
| Feed not on the allowlist | job `failed` immediately with a clear message, no retries |
| Real crash (SIGKILL while ingest steps were in flight) | 11 step calls before the kill, 29 after the restart: 40 in total, exactly what an uninterrupted ingest of 5 articles makes. No finished step ran twice; counts back to 5 / 234, 5 `costs` rows |

Embedding the five posts costs 54k tokens, **$0.0011**.

## Sparse vs dense, by hand

Trying both modes of `/search` on the same questions:

- **Sparse is all or nothing.** `websearch_to_tsquery` ANDs every term, so a passage
  must contain all of them. "largest IPO in history beating SpaceX" matches nothing,
  although each word is in the article. Any paraphrase fails the same way.
- **Dense misses exact identifiers.** "SMT solver" is a precise term that sparse finds
  and dense ranks low. Dense wins on meaning, sparse wins on rare exact tokens.
- **Latency:** sparse answers in a few milliseconds; dense takes most of a second,
  almost all of it the query embedding round trip.

Neither alone is enough, which is the case for fusing them next.

## Next

Three feeds with flow control (a global throttle, priority for manual adds), a
backfill of each feed's archive, hybrid search with reciprocal rank fusion, and
`/ask` with citations.
