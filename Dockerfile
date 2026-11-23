# The app as a container: API, Inngest function endpoint and migrations.
# Postgres, Memgraph and Inngest are separate (see infra/docker-compose.prod.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, from the lockfile only, so this layer is rebuilt when
# the lockfile changes and not on every code change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY alembic.ini publications.yaml ./
COPY alembic ./alembic
COPY evals ./evals
COPY commonplace ./commonplace
RUN uv sync --frozen --no-dev

RUN useradd --system --no-create-home app
USER app

EXPOSE 8000

# Migrations run before the server starts, so a new image never serves an old schema.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn commonplace.main:app --host 0.0.0.0 --port 8000"]
