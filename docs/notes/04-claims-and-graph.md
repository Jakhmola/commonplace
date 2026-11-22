# 04 - Claims, a graph, and a third retriever

## Goal

Passages are text. To compare what different authors say about the same thing I
need something smaller and more structured: individual claims, each tied to the
sentence that supports it, connected through the topics they are about.

## Ontology before code

The graph's vocabulary was fixed first and is closed:

```
(Article)-[:PUBLISHED_IN]->(Publication)
(Article)-[:WRITTEN_BY]->(Author)
(Article)-[:ASSERTS]->(Claim)
(Claim)-[:ABOUT]->(Topic)
(Claim)-[:SUPERSEDES]->(Claim)
```

Five node labels, five edge types, nothing else. Topics are what connect one
publication's claims to another's, and they only do that if "RAG",
"retrieval-augmented generation" and "RAGs" end up as one node. So every topic
string is canonicalised in code before it is stored: aliases, a singular head noun,
and a stop list for words that would connect everything to everything ("AI",
"model", "system").

## What I built

| Piece | Where |
|---|---|
| Ontology and canonicalisation | `commonplace/graph/ontology.py` |
| Claim models, deterministic claim ids | `commonplace/schemas/entities.py`, `commonplace/schemas/ids.py` |
| The quote guardrail | `commonplace/sources/quote_validator.py` |
| Agent layer (spec + one call inside a step) | `commonplace/agents/core.py`, `providers.py`, `prompts.py` |
| Claim extractor role | `commonplace/agents/roles/extractor.py` |
| `extract-claims` function | `commonplace/functions/enrich.py` |
| `claims` table, provenance queries | `commonplace/db/`, `alembic/versions/0004_claims.py` |
| Memgraph client and all Cypher | `commonplace/graph/client.py`, `commonplace/graph/cypher.py` |
| Graph retriever, three-way fusion | `commonplace/retrieval/graph.py`, `commonplace/retrieval/hybrid.py` |
| `GET /sources/{id}`, MCP `cp_sources` | `commonplace/operations.py`, `commonplace/mcp/tools.py` |
| Backlog script | `scripts/enrich_backlog.py` |

### The pipeline

```
cp/article.ingested | cp/article.updated --> extract-claims
    load-article
    extract-claims-llm     one model call, schema-constrained output
    validate-quote-spans   plain code: is the quote really in the article?
    persist-claims-pg      close the old version's claims, upsert the new
    sync-claims-memgraph   the same, merged into the graph
    emit                   cp/claims.extracted {claim_count, dropped_spans}
```

Runs are serialized per article and capped at two across all articles. Postgres is
written before the graph: the graph is derived data, so a failed graph step can be
retried without losing anything.

### The quote guardrail

A model asked to quote will sometimes paraphrase, tidy punctuation, or join two
sentences with an ellipsis, and the result still reads like a quotation. Each claim
must carry a `quote_span`, and a claim is kept only if that span is a verbatim
substring of the article. This is string matching: free, deterministic, and it runs
before anything is stored.

Two refinements mattered:

- Line breaks and typographic variants are ignored when matching. Models retype
  curly quotes and long dashes as their keyboard forms, which is a difference of
  typography, not of words.
- What is stored is cut from the **article**, not taken from the model. The
  provenance record is the source's own text.

Each claim also has a type (`measurement`, `opinion` or `prediction`), and the
answer prompt is told not to present an opinion or a prediction as something that
was measured.

### The agent layer

No framework. An `Agent` is a small declarative record (role, prompt, model, output
schema). `run_agent` makes its one model call inside `step.run`, which is what
gives it durability: the result is memoized, a retry of a later step never pays for
the call again, and its `usage` passes through the cost middleware. All provider
access is in one function, `call_llm`, which `/ask` now uses too, so answers are
finally recorded in `costs` as well.

### Idempotence and edits

- Claim ids are `uuid5` of article id, content hash, quote and claim text. A replay
  of the same extraction lands on the same rows and the same nodes.
- Every graph write is a `MERGE` on a key. A test checks all Cypher for three rules:
  parameters only, `MERGE` never `CREATE`, and every read of claims filters on
  `valid_to IS NULL`.
- A new version of an article closes the previous version's claims (`valid_to`) in
  both stores and links each new claim to the closed claim it replaces on the same
  topic (`supersedes_id` in Postgres, `SUPERSEDES` in the graph).
- A duplicate `cp/article.ingested` for an article that already has claims does
  nothing. Only an edit re-extracts.

### The graph as a retriever

Keyword and vector search match text. The graph follows structure: content words in
the question are canonicalised the same way topics are, the matching topics lead to
current claims, and each claim leads to the passage that holds its quote. That gives
a third ranking of passages, fused with the other two by rank (`hybrid_3way_rrf_k60`).
The graph has no similarity score at all, which is exactly why rank fusion fits.

Results now say which engines found them (`bm25`, `vector`, `both`, `graph`,
`bm25+graph`, `vector+graph`, `all`) and carry the claim that led there. The graph
is optional at read time: with Memgraph stopped, search and `/ask` run on two
engines and `/health` reports `degraded`.

### Sources

`GET /sources/{id}` and the MCP tool `cp_sources` take a claim id or a passage id
from any citation and return the exact text, the article, author and publication,
the claim's type and topics, and whether it is still current.

## Problems I hit (and fixed)

1. **Plural stripping mangled names.** "Synopsys" became "synopsy". Words ending in
   `ys` are now left alone and only known acronyms lose a plural `s` ("GPUs").
2. **A second claim from the same passage pushed other results down** in the graph
   ranking. A passage now takes one position, with its best-ranked claim.
3. **Fresh extraction is not deterministic.** Sending "ingested" again for an article
   re-ran the model, produced a slightly different set of claims and replaced the
   old ones. Hence the rule above: only an edit re-extracts.
4. **A replay erased supersession links.** On a second run nothing is stale any more,
   so the computed link was empty and overwrote the stored one. The upsert now keeps
   an existing link.
5. From the review of the previous phase: one failing feed blocked the feeds after
   it, an article that could never be ingested was re-queued on every poll, and an
   old failure could mark a new job as failed. All fixed.

## Verification

| Check | Result |
|---|---|
| `make check` | pass, 121 tests |
| Extraction of the whole corpus | 60 articles, 464 claims (210 opinion, 186 measurement, 68 prediction), about 5.5 minutes, **$3.55** |
| Quote guardrail on real output | 1 claim dropped of about 400 proposed |
| Graph | 60 articles, 7 authors, 3 publications, 471 claims, 672 topics, 1,414 `ABOUT` edges |
| Topics shared by all three publications | open models (53 claims), coding agents (19), reinforcement learning (14), fine-tuning (7), ... |
| Duplicate `cp/article.ingested` | `status: already_extracted`, no model call |
| Simulated edit | 5 old claims closed, 8 current, 8 `SUPERSEDES` links, `cp/article.updated` then `cp/claims.extracted` |
| `/ask` on a cross-publication question | variant `hybrid_3way_rrf_k60`; 13 of 37 candidates came through the graph; claim types kept apart in the answer |
| `/sources/<claim_id>` and `/sources/<passage_id>` | exact quote, article, validity; bad id 400, unknown id 404 |
| Memgraph stopped | `/search` answers in 0.5 s on two engines, `/health` is `degraded`; after restart the graph leg is back |

## Limits I know about

- An article whose extraction yields no valid claim has no claims, so the backlog
  script will offer it again each time.
- If the graph step fails for good, Postgres has the claims and the graph does not.
  There is no resync command yet; clearing the claims of that article and running
  the backlog script rebuilds both.
- Topic names for unknown terms are just capitalised slugs ("Openai").

## Next

Evaluate all of this: define what a bad answer is, check the cheap things with
code, judge the rest with a model that has been calibrated against labelled
examples, and compare the retrieval variants on a fixed set of questions.
