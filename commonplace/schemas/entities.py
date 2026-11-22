"""Claims and topics as they cross three boundaries: out of the LLM, through the
quote guardrail, and into Postgres and the graph.
"""

from typing import Literal

from pydantic import BaseModel, Field

# What kind of statement a claim is. An answer must not present an opinion or a
# prediction as if it had been measured.
ClaimType = Literal["opinion", "measurement", "prediction"]


class ExtractedTopic(BaseModel):
    slug: str
    name: str


class RawExtractedClaim(BaseModel):
    """One claim exactly as the model proposed it. Nothing here is trusted yet."""

    claim_text: str = Field(
        min_length=5, description="A concise, self-contained assertion the author makes"
    )
    quote_span: str = Field(
        description="An exact, character-for-character substring of the article supporting it"
    )
    claim_type: ClaimType
    entities: list[str] = Field(
        default_factory=list, description="1 to 5 specific technical topics the claim is about"
    )
    confidence: float = Field(
        default=0.9, ge=0.0, le=1.0, description="How directly the quote supports the claim"
    )


class ClaimExtractionBatch(BaseModel):
    claims: list[RawExtractedClaim] = Field(default_factory=list)


class ValidatedClaim(BaseModel):
    """A claim whose quote was found in the article and whose topics are canonical."""

    claim_id: str
    article_id: str
    text: str
    quote_span: str  # the article's own text, not the model's retyping of it
    claim_type: ClaimType
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    topics: list[ExtractedTopic] = Field(min_length=1)
