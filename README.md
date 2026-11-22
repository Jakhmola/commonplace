# commonplace

A personal knowledge base built from what I read. It ingests newsletters (and later books
and PDFs), keeps them fresh, and answers questions with citations I can click and check.

Every slow pipeline runs as durable [Inngest](https://www.inngest.com) steps, so a crash
or a provider outage resumes where it stopped instead of starting over (and paying twice).

## Stack

- **FastAPI** for the HTTP API
- **Inngest** for durable, retryable, rate-limited pipelines
- **Postgres + pgvector** for text, embeddings and BM25 full-text search in one place
- **uv**, **ruff**, **mypy**, **pytest**

## Getting started

Requirements: Docker, [uv](https://docs.astral.sh/uv/), an OpenAI API key.

```bash
cp .env.example .env    # fill in OPENAI_API_KEY
make start              # containers, migrations, then the app in the foreground
make check              # lint, types, tests
```

| Tool | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Inngest dev server | http://localhost:8288 |
| Adminer (Postgres) | http://localhost:8081 |

## Progress

Build notes for each phase live in [`docs/notes/`](docs/notes/).
