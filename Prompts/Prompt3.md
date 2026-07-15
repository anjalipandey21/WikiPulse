Prompt 3: Truthful live progress streaming
Act as a staff-level AI engineer experienced in LangGraph event streaming, FastAPI streaming responses, React, TypeScript, and cancellation-safe async systems.

Implement truthful live progress for WikiPulse analysis.

Add:

POST /api/audience-analysis/stream

Use application/x-ndjson.

Requirements:

1. Preserve the existing non-streaming endpoint and final AudienceAnalysisResponse.

2. Emit only strict public events:

Progress:
{
  "type": "progress",
  "sequence": 1,
  "stage": "fetching_pageviews"
}

Result:
{
  "type": "result",
  "sequence": 12,
  "result": { existing AudienceAnalysisResponse }
}

Error:
{
  "type": "error",
  "sequence": 8,
  "status_code": 502,
  "error": {
    "code": "audience_provider_unavailable",
    "message": "Audience generation is temporarily unavailable."
  }
}

3. Emit progress only from actual workflow boundaries:

- waiting_for_slot
- fetching_pageviews
- selecting_articles
- enriching_summaries
- modeling_topics
- routing_commercial_clusters
- preparing_audience_evidence
- generating_audience_decisions
- validating_audience_decisions
- revising_audience_decisions
- validating_revised_decisions
- finalizing_audience_results
- assembling_response

4. Never serialize:

- raw LangGraph events
- graph state
- prompts
- raw model output
- model or response IDs
- token-level events
- arbitrary backend messages
- exception details
- percentages or invented timing estimates

5. Use allowlisted LangGraph node starts only.

6. Use one bounded request-local queue and a monotonic sequence counter.

7. Handle cancellation correctly:

- abort waiting requests safely
- cancel and await the producer
- re-raise CancelledError
- do not emit an error event for a disconnected browser
- ensure asyncio.to_thread work finishes safely before releasing shared encoder resources
- prevent provider calls after cancellation during topic modeling

8. Frontend:

- add a strict incremental NDJSON parser
- preserve incomplete lines across chunks
- require increasing sequence numbers
- require exactly one terminal event
- reject malformed JSON and unknown stages
- use the existing AbortController flow
- preserve the previous successful dashboard during refresh
- preserve previous results after refresh failure
- use fixed frontend copy for progress stages

9. Add backend and frontend tests for:

- progress ordering
- terminal result
- safe terminal errors
- malformed stream
- unknown stage
- decreasing sequence
- EOF without terminal event
- trailing events
- cancellation
- previous-result preservation

Do not add WebSockets, polling, job storage, new LLM calls, or new dependencies.

Run backend tests, frontend tests, lint, production build, and git diff --check.

Do not commit or push.
