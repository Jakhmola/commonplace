"""Every Cypher statement the project runs.

Three rules, checked by a test:

1. Values arrive as `$parameters`. No statement is ever built by formatting
   a string.
2. Writes use `MERGE` on a key, never `CREATE`, so running a statement twice
   leaves one node and one edge.
3. Every statement that reads claims says which ones are current
   (`valid_to IS NULL`).
"""

# The one place CREATE appears: indexes and uniqueness constraints on each
# label's key. Run once per process; "already exists" is ignored.
SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT ON (p:Publication) ASSERT p.slug IS UNIQUE",
    "CREATE CONSTRAINT ON (a:Author) ASSERT a.slug IS UNIQUE",
    "CREATE CONSTRAINT ON (a:Article) ASSERT a.canonical_id IS UNIQUE",
    "CREATE CONSTRAINT ON (c:Claim) ASSERT c.claim_id IS UNIQUE",
    "CREATE CONSTRAINT ON (t:Topic) ASSERT t.slug IS UNIQUE",
    "CREATE INDEX ON :Publication(slug)",
    "CREATE INDEX ON :Author(slug)",
    "CREATE INDEX ON :Article(canonical_id)",
    "CREATE INDEX ON :Claim(claim_id)",
    "CREATE INDEX ON :Topic(slug)",
]

# The article with its publication and author.
MERGE_ARTICLE_SKELETON = """
MERGE (p:Publication {slug: $pub_slug})
SET p.name = $pub_name, p.homepage = $pub_homepage
MERGE (au:Author {slug: $author_slug})
SET au.name = $author_name
MERGE (art:Article {canonical_id: $canonical_id})
SET art.article_id = $article_id,
    art.url = $url,
    art.title = $title,
    art.published_at = $published_at,
    art.content_hash = $content_hash
MERGE (art)-[:PUBLISHED_IN]->(p)
MERGE (art)-[:WRITTEN_BY]->(au)
"""

# Close the article's claims that are not part of its current version.
SUPERSEDE_ARTICLE_CLAIMS = """
MATCH (art:Article {canonical_id: $canonical_id})-[:ASSERTS]->(c:Claim)
WHERE c.valid_to IS NULL AND NOT c.claim_id IN $active_claim_ids
SET c.valid_to = $now
RETURN count(c) AS superseded
"""

# The current claims and the topics they are about.
MERGE_CLAIMS_BATCH = """
MATCH (art:Article {canonical_id: $canonical_id})
UNWIND $claims AS claim
MERGE (c:Claim {claim_id: claim.claim_id})
ON CREATE SET c.valid_from = claim.valid_from
SET c.article_id = claim.article_id,
    c.text = claim.text,
    c.quote_span = claim.quote_span,
    c.claim_type = claim.claim_type,
    c.confidence = claim.confidence,
    c.valid_to = null
MERGE (art)-[:ASSERTS]->(c)
FOREACH (topic IN claim.topics |
    MERGE (t:Topic {slug: topic.slug})
    SET t.name = topic.name
    MERGE (c)-[:ABOUT]->(t)
)
"""

MERGE_SUPERSEDES_EDGES = """
UNWIND $pairs AS pair
MATCH (new:Claim {claim_id: pair.new_claim_id})
MATCH (old:Claim {claim_id: pair.old_claim_id})
MERGE (new)-[r:SUPERSEDES]->(old)
SET r.at = pair.at
"""

# Current claims about any of the given topics. A topic matches by its slug,
# or when one of the plain search terms is a whole word of its slug. Claims
# that match more of the topics come first.
SEARCH_CLAIMS_BY_TOPICS = """
MATCH (t:Topic)<-[:ABOUT]-(c:Claim)<-[:ASSERTS]-(art:Article)-[:PUBLISHED_IN]->(pub:Publication)
WHERE c.valid_to IS NULL
  AND (
    t.slug IN $topic_slugs
    OR any(term IN $search_terms WHERE term IN split(t.slug, '-'))
  )
OPTIONAL MATCH (art)-[:WRITTEN_BY]->(au:Author)
RETURN c.claim_id AS claim_id,
       c.article_id AS article_id,
       c.text AS claim_text,
       c.quote_span AS quote_span,
       c.claim_type AS claim_type,
       c.confidence AS confidence,
       art.title AS title,
       art.url AS url,
       art.published_at AS published_at,
       au.name AS author,
       pub.slug AS publication_slug,
       collect(DISTINCT t.slug) AS matched_topic_slugs,
       count(DISTINCT t) AS matched_topics
ORDER BY matched_topics DESC, confidence DESC, published_at DESC, claim_id ASC
LIMIT $limit
"""

COUNT_ACTIVE_CLAIMS_AND_TOPICS = """
OPTIONAL MATCH (c:Claim)
WHERE c.valid_to IS NULL
OPTIONAL MATCH (c)-[:ABOUT]->(t:Topic)
RETURN count(DISTINCT c) AS claims, count(DISTINCT t) AS topics
"""

# Topics that connect the most publications: where authors can be compared.
TOPICS_SHARED_ACROSS_PUBLICATIONS = """
MATCH (pub:Publication)<-[:PUBLISHED_IN]-(:Article)-[:ASSERTS]->(c:Claim)-[:ABOUT]->(t:Topic)
WHERE c.valid_to IS NULL
WITH t, count(DISTINCT pub) AS publications, count(DISTINCT c) AS claims
WHERE publications > 1
RETURN t.slug AS topic, t.name AS name, publications, claims
ORDER BY publications DESC, claims DESC, topic ASC
LIMIT $limit
"""
