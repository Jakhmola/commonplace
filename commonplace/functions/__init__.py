from typing import Any

import inngest

from commonplace.functions.discovery import add_publication
from commonplace.functions.ingest import ingest_article

FUNCTIONS: list[inngest.Function[Any]] = [add_publication, ingest_article]
