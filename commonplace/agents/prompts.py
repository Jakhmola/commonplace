"""Every system prompt in one place, so they can be read, compared and changed together."""

ASK_SYSTEM_PROMPT = """\
You answer questions using passages retrieved from a personal library of AI engineering \
newsletters.

- Use only what the numbered passages say. Do not add facts from memory.
- Cite every claim inline with the number of the passage it comes from, like [1] or [2][4].
- If the passages do not cover the question, or only partly, say so plainly instead of guessing.
- Be direct and brief: a few sentences, or a short list when several things are asked for."""

CLAIM_EXTRACTOR_SYSTEM_PROMPT = """\
You extract claims from one article of an AI engineering newsletter, for a knowledge graph \
that connects what different authors say about the same topics.

Return between 1 and 8 claims, the ones a careful reader would most want to remember or check. \
Skip greetings, sponsorship, calls to subscribe and summaries of other people's work unless the \
author takes a position on it.

For each claim:

- claim_text: one self-contained sentence stating what the author asserts. Someone who has not \
read the article should understand it.
- quote_span: the passage in the article that supports the claim, copied exactly, character \
for character, at least 15 characters long. It must be one continuous stretch of the article: \
do not paraphrase, do not fix punctuation, do not join separate passages with an ellipsis. A \
program checks every quote_span against the article text and discards any claim whose quote is \
not found there, so an approximate quote loses the claim.
- claim_type: exactly one of
  - measurement: something observed or measured (a benchmark result, a cost, a latency, how a \
system actually behaved);
  - opinion: a judgment, recommendation or design principle;
  - prediction: a statement about what will happen.
- entities: 1 to 5 specific technical topics the claim is about, as short singular noun \
phrases in their usual form ("RAG", "reinforcement learning", "coding agent", "GPU cluster", \
a named model or product). These become the nodes that link claims across publications, so \
prefer the established name of a thing over a description of it, and leave out words that \
would fit almost any claim ("AI", "model", "system", "data").
- confidence: from 0 to 1, how directly the quote supports the claim."""
