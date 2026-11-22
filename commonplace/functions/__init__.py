"""The functions served to Inngest."""

from typing import Any

import inngest

from commonplace.functions.backfill import backfill_publication
from commonplace.functions.freshness import poll_feeds
from commonplace.functions.ingest import ingest_article
from commonplace.functions.ops import handle_function_failed

FUNCTIONS: list[inngest.Function[Any]] = [
    backfill_publication,
    ingest_article,
    poll_feeds,
    handle_function_failed,
]
