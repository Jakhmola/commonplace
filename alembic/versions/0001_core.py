"""core tables: publications, articles, passages, jobs, costs

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE publications (
            id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            slug       text NOT NULL UNIQUE,
            name       text NOT NULL,
            feed_url   text NOT NULL,
            homepage   text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE articles (
            id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            canonical_id   text NOT NULL UNIQUE,
            publication_id uuid NOT NULL REFERENCES publications(id),
            url            text NOT NULL,
            title          text NOT NULL,
            author         text,
            published_at   timestamptz,
            content_hash   text NOT NULL,
            body           text NOT NULL,
            created_at     timestamptz NOT NULL DEFAULT now(),
            updated_at     timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE passages (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            article_id  uuid NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
            chunk_index int  NOT NULL,
            body        text NOT NULL,
            embedding   vector(1536) NOT NULL,
            tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
            UNIQUE (article_id, chunk_index)
        )
    """)
    op.execute("CREATE INDEX passages_tsv_idx ON passages USING gin (tsv)")
    op.execute(
        "CREATE INDEX passages_embedding_idx ON passages USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("""
        CREATE TABLE jobs (
            event_id       text PRIMARY KEY,
            publication_id uuid NOT NULL REFERENCES publications(id),
            canonical_ids  text[] NOT NULL,
            created_at     timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE costs (
            id         bigserial PRIMARY KEY,
            kind       text NOT NULL,
            model      text NOT NULL,
            tokens     int  NOT NULL,
            usd        numeric(12, 6) NOT NULL,
            article_id uuid REFERENCES articles(id) ON DELETE SET NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE costs, jobs, passages, articles, publications")
