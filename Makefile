.PHONY: up start stop logs reset db-clear lint test check

-include .env
export

COMPOSE := docker compose -f infra/docker-compose.yml
APP_PORT ?= 8000

up:
	$(COMPOSE) up -d --wait

start: up
	@if [ -f alembic.ini ]; then uv run alembic upgrade head; fi
	@if [ -f commonplace/main.py ]; then uv run uvicorn commonplace.main:app --reload --port $(APP_PORT); fi

stop:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

# Destructive: drops the volumes.
reset:
	$(COMPOSE) down -v

# Destructive: empties every app table (keeps the schema) and the whole graph.
db-clear:
	$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-commonplace} -d $${POSTGRES_DB:-commonplace} -f - < infra/db_clear.sql
	$(COMPOSE) exec -T memgraph mgconsole < infra/graph_clear.cypher

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest -q

check: lint test
