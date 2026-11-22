# commonplace

A personal knowledge base built from what I read. It ingests newsletters (and later books
and PDFs), keeps them fresh, and answers questions with citations I can click and check.

Every slow pipeline runs as durable [Inngest](https://www.inngest.com) steps, so a crash
or a provider outage resumes where it stopped instead of starting over (and paying twice).

## Stack

- **FastAPI** for the HTTP API
- **Inngest** for durable, retryable, rate-limited pipelines
- **Postgres + pgvector** for text, embeddings and full-text search in one place
- **Claude** for answers grounded in the retrieved passages
- **uv**, **ruff**, **mypy**, **pytest**

## Getting started

Requirements: Docker, [uv](https://docs.astral.sh/uv/), an OpenAI API key (embeddings)
and an Anthropic API key (answers).

```bash
cp .env.example .env    # fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
make start              # containers, migrations, then the app in the foreground
make check              # lint, types, tests
```

| Tool | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Inngest dev server | http://localhost:8288 |
| Adminer (Postgres) | http://localhost:8081 |

## Using it

```bash
# ingest every post in an allowlisted feed (returns at once)
curl -X POST localhost:8000/publications -H 'content-type: application/json' \
  -d '{"feed_url": "https://www.latent.space/feed"}'
curl localhost:8000/jobs/<event_id>                      # progress

# ingest one article now, ahead of any queued backfill
curl -X POST localhost:8000/articles -H 'content-type: application/json' \
  -d '{"url": "https://www.latent.space/p/<post-slug>"}'

# search: hybrid (default), sparse or dense
curl 'localhost:8000/search?q=formal+verification'
curl 'localhost:8000/search?q=formal+verification&mode=sparse'

# ask a question, get an answer with citations
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What are the main approaches to evaluating LLMs?"}'

# show that replaying every backfill writes nothing new
uv run python scripts/replay_backfill.py --verify
uv run python scripts/replay_backfill.py --all

# check the feeds for new articles now (they are also polled every five minutes)
curl -X POST localhost:8288/e/dev -H 'content-type: application/json' \
  -d '{"name": "cp/feed.poll", "data": {}}'
```

Feeds must be listed in [`publications.yaml`](publications.yaml). `/ask` needs
`ANTHROPIC_API_KEY`; without it the retrieved passages are returned on their own.

## Using it from Claude Code (MCP)

`cp-mcp` serves the knowledge base as MCP tools over stdio: `cp_search`, `cp_ask`,
`cp_list_publications`, `cp_job_status`, `cp_health`, `cp_add_publication`,
`cp_backfill` and `cp_sync`. [`.mcp.json`](.mcp.json) registers it for Claude Code
when Claude Code is started in this directory (it asks once before enabling it).
From anywhere else:

```bash
claude mcp add commonplace -- uv run --directory /path/to/commonplace cp-mcp
```

The stack (`make start`) has to be running.

## Tests

`make test` runs everything. Unit tests need nothing. The tests in
`tests/integration/` use the real Postgres from `make up`, each inside a
transaction that is rolled back, and are skipped when it is not running.

## Progress

Build notes for each phase live in [`docs/notes/`](docs/notes/).
