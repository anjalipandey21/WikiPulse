Prompt 4: Agent Journey / explainability

Act as a staff-level AI observability and full-stack engineer.

Design and implement a public-safe Agent Journey for completed WikiPulse analyses.

The journey must be a deterministic post-run projection of observable workflow outcomes.

It must not expose chain-of-thought, prompts, raw provider output, model metadata, exception details, or inferred reasoning.

For each cluster, allow only events such as:

- generation_requested
- decision_received
- validation_passed
- validation_failed
- revision_requested
- revision_failed
- audience_published
- provider_skipped
- decision_dropped

Requirements:

1. Build immutable backend trace snapshots from:

- preparation order
- initial validation report
- revision validation report
- final segments
- provider skips
- explicit drops

2. Use deterministic response-local trace IDs.

3. Associate every final segment, provider skip, and validation drop with exactly one trace.

4. Commercial-routing skips must not receive an agent trace because they were filtered before LLM generation.

5. Expose only safe issue codes and reference IDs.

6. Add strict public API DTOs.

7. Add frontend runtime guards that reject:

- duplicate trace IDs
- orphan trace IDs
- cluster mismatch
- outcome mismatch
- invalid event sequences

8. Render the journey as a collapsed native disclosure:

View agent journey

Each trace must also be collapsed by default.

9. Use fixed user-friendly copy.

Include this visible notice:

“This records workflow outcomes, not private model reasoning.”

10. Links from audience cards and diagnostics should:

- open the disclosure
- reveal the selected trace
- scroll it into view
- focus its heading accessibly

11. Add tests for:

- initial publish
- provider skip
- corrected revision
- failed revision
- still-invalid revision
- unknown initial output
- unknown revision output
- empty preparation
- safe serialization
- no private provider/model data
- collapsed UI state
- hidden internal identifiers

Do not change LangGraph execution, providers, prompts, or calculations.

Run focused tests, full backend tests, frontend tests, lint, build, and git diff --check.

Do not commit or push.

