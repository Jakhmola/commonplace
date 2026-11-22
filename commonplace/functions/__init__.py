"""The functions served to Inngest."""

from typing import Any

import inngest

from commonplace.functions.backfill import backfill_publication
from commonplace.functions.enrich import extract_claims
from commonplace.functions.evals import evaluate_answer, run_eval_suite
from commonplace.functions.freshness import poll_feeds
from commonplace.functions.ingest import ingest_article
from commonplace.functions.ops import handle_function_failed
from commonplace.functions.research import run_research_agent

FUNCTIONS: list[inngest.Function[Any]] = [
    backfill_publication,
    ingest_article,
    poll_feeds,
    handle_function_failed,
    extract_claims,
    evaluate_answer,
    run_eval_suite,
    run_research_agent,
]
