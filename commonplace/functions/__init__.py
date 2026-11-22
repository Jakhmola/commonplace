"""The functions served to Inngest."""

from typing import Any

import inngest

from commonplace.functions.backfill import backfill_publication
from commonplace.functions.ingest import ingest_article

FUNCTIONS: list[inngest.Function[Any]] = [backfill_publication, ingest_article]
