# commonplace: a one-page writeup

## The problem
I read several AI engineering newsletters and cannot reliably recall who said what, whether it was measured or only argued, or where two authors disagree. I also don't have time to read every article properly, and what I do read I forget. I wanted a personal knowledge base that reads for me: it ingests what I follow, stays current, and extracts what I care about, answering with citations I can check. Two things make that harder than "RAG over some posts": the pipelines call paid APIs and must survive failures without paying twice, and an answer that sounds right is worthless unless I can tell when it is wrong.

## The approach

Six phases, each ending in something that runs:

1. **Durable ingest**: fetch, parse, chunk, embed and store as Inngest steps with
   idempotent writes.
2. **Flow control, hybrid search, cited answers** from retrieved passages only.
3. **Freshness**: conditional feed polling, edits that supersede instead of
   delete, failure capture, MCP tools.
4. **Claims and a graph**: a model extracts claims, code keeps only those whose
   quote is verbatim in the article, a graph links them through topics.
5. **Evaluation**: a failure taxonomy, checks in code, two binary LLM judges
   measured against labelled traces, a retrieval benchmark.
6. **A bounded research agent, a weekly digest, a deployable image.**

Today: 3 feeds, 60 articles, 1,529 passages, 466 claims, 228 tests, 13 MCP tools.

## Key decisions and trade-offs

- **Every paid call is its own durable step.** A retry never repeats a finished
  call and each call's cost is recorded. The price: code is shaped by replay, and
  anything non-deterministic has to live inside a step.
- **Rank fusion, not score mixing.** Keyword scores, vector distances and graph
  hits share no scale, so only positions are combined. Robust, but it cannot
  express that one engine is far more certain.
- **A claim must quote its source verbatim, and code checks it.** Some good claims
  are lost to an inexact quote; none are invented.
- **Nothing is deleted.** Passages and claims carry validity intervals, so an old
  answer stays traceable after an article is edited.
- **The agent's limits live in code, not in prompts**: an iteration cap, a dollar
  cap and a convergence check end the research loop.
- **Measure, then say what the numbers show.** Hybrid search answers questions across publications, pulling together what different authors said and attaching claim types so I can see what was measured and what was only argued. On my 25-question retrieval benchmark, the search that answers use reaches recall@5 of 0.82.
- 
## How I used AI tools

The repository was built with Claude Code, working largely autonomously over long sessions, but I directed it at every phase. The architecture is not my original design: it follows the reference design of a course I was working through, with the code written independently to the same patterns and results. My part was the goal, the constraints, the decisions, and the quality gates around each phase.

My workflow for each phase:

1. **Brainstorm and plan.** I used a brainstorming skill to work through the problem and shape the overall plan with Claude.
2. **Phase-wise plan.** The agent broke the plan into a plan per phase and divided the work.
3. **I critique the plan.** I reviewed each phase plan myself and pushed back, and the agent fixed it before any code was written.
4. **Implementation.** Only after the revised plan was agreed did the agent implement it, writing tests before code and running the feature against the live system.
5. **Agent review.** A fresh read-only agent reviewed the diff, followed by one fix pass. These reviews earned their cost: they found that the benchmark was not measuring the search that answers use, and that a bracketed date such as `[2024-01-15]` crashed the citation parser after the model had been paid.
6. **I smoke test and QA.** When a phase was done, I smoke tested and QA'd it myself before moving to the next.

Claude is also inside the product (answers, claim extraction, judges), with real limits: the calibration traces were labelled by the assistant and still need my review, the judge prompts were tuned on the traces they are measured on, and one verdict flipped between two identical runs.

## What I would do next

- Cap passages per article in a prompt; cross-publication questions (recall 0.50
  to 0.60) point straight at it.
- Score long reports per statement instead of pass or fail as a whole.
- Label a held-out set of traces myself and call each judge several times.
- Add books and PDFs: only a parser is missing.
- Test the Inngest Cloud deployment end to end; so far it is verified only up to
  a container that boots and rejects unsigned requests.
