-- What each retriever contributed to one answer's prompt.
--
--   docker compose -f infra/docker-compose.yml exec -T postgres \
--     psql -U commonplace -d commonplace -v sid=<snapshot_id> -f - < scripts/explain_answer.sql
--
-- For every passage the model was shown: its rank in keyword search, vector
-- search and the claim graph, and the points each rank earned under reciprocal
-- rank fusion (1000 / (60 + rank)). The last row is each retriever's share of
-- all the points in the prompt. k = 60 is settings.rrf_k.
with shown as (
  select x->>'title' as title,
         x->>'claim_type' as claim_type,
         (x->>'sparse_rank')::int as kw_rank,
         (x->>'dense_rank')::int as vec_rank,
         (x->>'graph_rank')::int as graph_rank
  from snapshots, jsonb_array_elements(retrieved) as x
  where snapshot_id = :'sid' and (x->>'in_prompt')::bool
),
points as (
  select *,
         coalesce(1000.0 / (60 + kw_rank), 0) as kw,
         coalesce(1000.0 / (60 + vec_rank), 0) as vec,
         coalesce(1000.0 / (60 + graph_rank), 0) as graph
  from shown
)
select passage, claim, kw_rank, vec_rank, graph_rank, kw, vec, graph, total
from (
  select 0 as last,
         left(title, 42) as passage,
         coalesce(claim_type, '-') as claim,
         coalesce(kw_rank::text, '-') as kw_rank,
         coalesce(vec_rank::text, '-') as vec_rank,
         coalesce(graph_rank::text, '-') as graph_rank,
         round(kw, 1)::text as kw,
         round(vec, 1)::text as vec,
         round(graph, 1)::text as graph,
         kw + vec + graph as sort_key,
         round(kw + vec + graph, 1)::text as total
  from points
  union all
  select 1, 'share of the prompt', '', '', '', '',
         round(100 * sum(kw) / sum(kw + vec + graph)) || '%',
         round(100 * sum(vec) / sum(kw + vec + graph)) || '%',
         round(100 * sum(graph) / sum(kw + vec + graph)) || '%',
         0, '100%'
  from points
) as rows
order by last, sort_key desc;
