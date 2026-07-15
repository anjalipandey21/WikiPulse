# AGENTS.md

## Project

**WikiPulse** is an eight-hour AI Builder assessment project.

The app analyzes the most-viewed English Wikipedia articles from the latest seven complete days, groups related articles into topic clusters, and converts commercially useful clusters into an Emerging Audience Portfolio.

Use the term **pageviews**, not searches. Wikipedia Pageviews data shows what people viewed, not what they searched for.

## How to work in this repository

- Inspect the relevant files before editing and make the smallest change needed for the current task.
- Only modify files that are directly required. Do not refactor, rename, or reformat unrelated code.
- Do not add a dependency, service, abstraction, or folder unless the current task clearly requires it.
- For a broad or ambiguous task, explain the minimal plan first and wait for approval before writing code.
- After editing, summarize changed files, commands run, test results, and any remaining risk.

## Product scope

The MVP should provide two views:

1. **Weekly Topic Landscape** — meaningful topic clusters such as sports, science, technology, entertainment, or history.
2. **Emerging Audience Portfolio** — only coherent, safe, and commercially meaningful audience segments derived from those clusters.

Keep the MVP limited to:

- English Wikipedia
- Latest seven complete days
- Top 50–100 aggregated articles
- Approximately 5–10 topic clusters
- Approximately 3–6 commercial audience segments
- At most one critique/revision cycle
- A simple FastAPI backend and React/TypeScript dashboard

Do not add authentication, user accounts, a production database, queues, Kubernetes resources, or unrelated product features unless explicitly requested.

### Human-review persistence and deployment limitation

- Phase 3 stores Human Review checkpoints and minimal run/receipt indexes in SQLite. The default ignored runtime file is `backend/data/wikipulse_review.db`; override it with `WIKIPULSE_REVIEW_DB_PATH`.
- Restart hydration is read-only and preserves pending/terminal runs, absolute expiry, and completed start/command idempotency.
- The backend must still run with one application worker; multi-worker coordination is deferred.
- An analyst edit interrupted by process failure is never repeated automatically. The hydrated run fails safely and the analyst must start a new analysis.

## Required architecture

Keep these concerns separate:

- `services/`: Wikimedia API access, response normalization, caching, and article-summary enrichment
- `filtering/`: deterministic cleanup, regex rules, and text normalization
- `clustering/`: embeddings, similarity, clustering, keywords, and confidence calculations
- `agent/`: LLM prompts, structured generation, routing, critique, and bounded workflow state
- `models/`: Pydantic request, internal, and response schemas
- `api/`: FastAPI routes and request handling
- `frontend/`: presentation and API integration only; no business logic
- `tests/`: focused tests for deterministic logic and agent boundaries

Do not create every module in advance. Add a file only when the current phase needs it.

## Data pipeline

Preserve this order unless a task explicitly changes it:

1. Fetch seven complete days from the Wikimedia Pageviews API.
2. Normalize titles and aggregate duplicate articles in Python.
3. Remove administrative and obvious noise pages with deterministic rules.
4. Enrich retained articles with public Wikipedia summaries.
5. Create local embeddings from article title plus summary.
6. Build candidate topic clusters with deterministic clustering.
7. Calculate topic confidence and size metrics in Python.
8. Route only cluster-level context to the LLM.
9. Generate structured commercial audience interpretations.
10. Validate results programmatically.
11. Run critique only for failed or ambiguous results, with one revision maximum.
12. Return traceable topic and audience results to the UI.

## LLM and cost rules

- Use normal Python for fetching, aggregation, sorting, arithmetic, thresholds, validation, and obvious filtering.
- Do not send every article individually to a generative LLM. Send compact cluster-level context only.
- Use local embeddings for semantic similarity when practical.
- Require structured LLM output validated by Pydantic; never trust free-form output directly.
- Keep LLM provider and model names configurable through backend environment variables.
- Set `MAX_REVISIONS = 1`; never implement an unbounded autonomous loop.
- Cache repeatable API or LLM results when the current task requires it, using inputs and prompt/model version in the cache key.
- Track useful efficiency metrics, but do not invent cost savings or unsupported percentages.

## Confidence and calculations

Keep these concepts separate:

- **Topic confidence:** deterministic score describing how strongly articles belong together.
- **Size index:** `cluster_views / total_analyzed_views * 100`, calculated in Python.
- **Commercial confidence:** LLM-supported qualitative judgment with a clear reason.

Do not ask the LLM to calculate pageview totals, percentages, or clustering confidence.

## Reliability and safety

- Every topic and audience must remain traceable to real supporting Wikipedia articles and view counts.
- Never allow the model to cite an article that is not present in its source cluster.
- Do not turn tragedies, deaths, disasters, violent events, or other sensitive breaking news into advertising audiences.
- Isolated or weakly related articles may remain unclustered; do not force every article into a topic.
- Handle Wikimedia timeouts, missing days, malformed responses, and unavailable summaries without crashing the full analysis.
- Use a clear Wikimedia `User-Agent`.
- Keep API keys server-side and load them from environment variables only.
- Never commit `.env`, secrets, tokens, generated caches, or local logs.

## Coding conventions

### Python

- Use type hints and small, single-purpose functions.
- Use Pydantic models at external boundaries.
- Prefer async HTTP calls where they improve the Wikimedia service.
- Keep deterministic logic pure where possible so it can be unit tested.
- Raise or translate clear domain-specific errors instead of silently swallowing failures.

### TypeScript

- Use typed API contracts.
- Keep components focused and avoid duplicating backend calculations.
- Show loading, empty, partial-failure, and error states.
- Do not add a UI framework unless explicitly approved.

## Testing and verification

For every behavior change:

- Add or update the smallest relevant test.
- Run `python -m pytest` for backend changes when tests are available.
- Run the configured frontend lint/build commands for frontend changes.
- Review the final diff for unrelated edits and remove them.
- Do not change tests only to make incorrect behavior pass.

Important deterministic cases include:

- Seven-day duplicate aggregation
- Administrative-page filtering
- Size-index calculations
- Cluster-confidence calculations
- Unsupported article references
- Sensitive-content rejection
- Critique-loop termination after one revision
- Malformed structured LLM output

## AI development log

`AI_DEVELOPMENT_LOG.md` is evidence for the assessment.

- Do not invent or rewrite the user's prompts.
- Only add a prompt when the user explicitly asks, and preserve its wording.
- Record what Codex changed, what was manually reviewed, and why a decision was accepted or rejected.
- Keep entries concise and factual.

## Definition of done

A task is complete only when:

- The requested behavior is implemented without unrelated changes.
- Relevant tests or verification commands have been run.
- Secrets and environment files remain protected.
- The response summarizes changed files and validation results.
- Any limitation, skipped test, assumption, or unresolved issue is stated clearly.
