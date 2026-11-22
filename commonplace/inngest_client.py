import logging

import inngest

from commonplace.config import settings

client = inngest.Inngest(
    app_id="commonplace",
    is_production=not settings.inngest_dev,
    api_base_url=settings.inngest_base_url if settings.inngest_dev else None,
    event_api_base_url=settings.inngest_base_url if settings.inngest_dev else None,
    logger=logging.getLogger("uvicorn"),
)
