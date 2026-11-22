# 03 - Freshness, edit history, failures, and an MCP server

## Goal

Until now the corpus only changed when I asked. This phase makes it look after
itself: poll the feeds on a schedule without re-reading ones that have not changed,
keep history when an article is edited instead of overwriting it, remember the
runs that fail for good, and let an agent (Claude Code) use the whole thing as tools.

## What I built

| Piece | Where |
|---|---|
| Conditional feed polling, every five minutes and on demand | `commonplace/functions/freshness.py`, `commonplace/sources/rss.py` |
| Change detection and passage supersession | `commonplace/functions/ingest.py`, `commonplace/db/queries.py` |
| Failure capture | `commonplace/functions/ops.py` |
| One module behind both front ends | `commonplace/operations.py` |
| MCP server, eight tools, stdio | `commonplace/mcp/` |
| `failures` table, partial unique index on passages | `alembic/versions/0003_failures_and_passage_history.py` |
| Integration tests against the real Postgres | `tests/integration/` |

### Polling without re-reading

`poll-feeds` runs on a cron (`*/5 * * * *`) and on the event `cp/feed.poll`. For
each feed it sends one conditional request with the validators kept in `cursors`:

```
GET /feed
If-None-Match: W/"7fac4-plcf..."        <- the ETag from last time
```

- **304**: no body, no parsing, nothing queued. Only the counters move.
- **200**: the feed is parsed and only articles that are not stored yet are queued,
  with `via="poll"`.

The new validators are saved **after** the new articles are queued, in their own
step. If they were saved first and the run died before queueing, the next poll
would get a 304 and those articles would stay hidden until the feed changed again.

Not every server plays along. Of my three feeds, two send an ETag and answer 304;
the third sends none and is a 200 every time. That costs a download but nothing
else, because stored articles are skipped either way.

### Edits supersede, they do not overwrite

After `parse`, a new step compares the content hash with the stored one:

| Stored hash | Result | What happens |
|---|---|---|
| none | `new` | chunk, embed, store, send `cp/article.ingested` |
| same | `unchanged` | stop. No chunking, no embedding call, no rows |
| different | `changed` | chunk, embed, close the current passages, insert new ones, send `cp/article.updated` |

"Close" means `valid_to = now()`. Nothing is deleted. Every search already filters
on `valid_to IS NULL`, so answers only ever use the current version, while the old
passages stay in the table: an answer given last month can still be traced to the
text it was based on. The unique index on `(article_id, chunk_index)` now applies
only to current passages, since a superseded passage keeps its index.

`persist` had to stay safe to run twice. Inngest re-runs a step if its response is
lost after the transaction committed; a naive second run would supersede the
passages the first run had just written. It now checks whether this exact version
is already stored and, if so, writes nothing.

### Failures are data

Inngest emits `inngest/function.failed` when a run fails for good. A small function
turns each one into a `failures` row: function, run, the error message (never the
stack trace) and the event that started the run. It ignores its own failures, or a
failure to record a failure would record itself forever.

This also fixed something I had left open since phase 1: a job whose article failed
permanently stayed `running`. A job now counts its expected articles as done
(stored), failed (a failure row exists and it is still not stored) or pending, and
settles as `failed` with the reasons listed. If the article is retried and arrives,
the old failure stops counting.

### An MCP server

`cp-mcp` exposes eight tools over stdio:

| Tool | Kind | Does |
|---|---|---|
| `cp_health` | read | dependency status and corpus size |
| `cp_search` | read | hybrid search, passages with sources |
| `cp_ask` | read | cited answer |
| `cp_list_publications` | read | publications, article counts, last poll |
| `cp_job_status` | read | progress of an ingestion job |
| `cp_add_publication` | starts work | backfill one feed |
| `cp_backfill` | starts work | backfill one or all publications |
| `cp_sync` | starts work | poll feeds now |

The rule behind the split: reads answer directly and fast; anything slow returns a
handle immediately and the agent polls. An agent should never be stuck inside a
long tool call. Each tool carries annotations (read-only or not, never destructive,
safe to repeat) and a description written for the model that has to choose it.

The tools and the HTTP endpoints call the same functions in `operations.py`; each
front end only translates `OperationError` into its own kind of error. `.mcp.json`
registers the server for Claude Code when it is opened in this repository.

## Problems I hit (and fixed)

1. **Keyword search was silent on real questions.** The full-text query required
   every word, so "What does X say about open models?" matched almost nothing and
   hybrid search was running on vectors alone. Matching any word fixed recall and
   broke ranking: that question matched 463 of 1,529 passages, mostly through
   "agent" and "compare". Each matched term now adds its inverse document frequency
   to the score, so one rare term outweighs any number of common ones. On the same
   question the keyword engine now supplies 15 of 24 candidates instead of 6 of 21.
2. **An embedding outage took search down.** Hybrid search now embeds the query
   before it touches the database and falls back to keywords if that call fails.
3. **Errors that were not API errors escaped.** The Anthropic SDK can raise before
   it sends anything; `/ask` now turns any exception into the answer text, and a
   response with no text (budget spent on thinking) is reported rather than
   returned empty.
4. **Citations like `[1, 2]` and `[1-3]` were not counted.** They are now.
5. **`/health` counted superseded passages** as part of the corpus. It counts
   current ones.

## Verification

| Check | Result |
|---|---|
| `make check` | pass, 74 tests (unit + integration, the latter rolled back) |
| `alembic check` | models and migrations agree |
| Poll twice | first 200 for all; second: two feeds 304 (ETag), one 200 (no validators); 0 articles queued; `cursors` counters updated |
| Simulated edit (stored hash changed, article re-sent) | 11 passages superseded, 11 new current ones, `cp/article.updated` sent, one embedding call |
| Same article re-sent unchanged | `status: unchanged`, no embedding call, no rows |
| Search after the edit | 1,529 current passages searchable out of 1,540 rows |
| Article outside the allowlist sent directly | run fails without retries; a `failures` row holds the message and the original event |
| MCP over stdio | 8 tools listed; `cp_health`, `cp_search`, `cp_list_publications`, `cp_sync` called; unknown publication and unknown job come back as tool errors; `limit=99` is rejected by the schema |

## Limits I know about

- Polling finds **new** articles. An edit to an existing article is only noticed
  when that article is ingested again (`POST /articles`); RSS has no reliable
  "updated" signal to poll on.
- `robots.txt` is fetched without the redirect allowlist check that article pages
  get. Only rules are read from it.
- `POST /articles` accepts any path on an allowlisted host, so `/archive` would be
  ingested as an article.

## Next

Extract claims and entities from each article into a graph, add the graph as a
third retriever, and make every claim traceable to the exact passage it came from.
