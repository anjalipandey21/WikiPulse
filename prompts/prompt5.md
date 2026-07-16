Prompt 5: Read-only adversarial review
Act as a staff-level AI systems reviewer.

Perform a read-only adversarial review of the current WikiPulse implementation.

Do not modify files, install dependencies, commit, or push.

Review the actual implementation and tests, not only the implementation summary.

Focus on:

1. Correctness
- output ordering
- object identity
- metrics
- token accounting
- one-revision bound
- error handling

2. LangGraph behavior
- graph topology
- conditional routing
- no accidental second revision
- no shared mutable state between invocations
- runtime-context isolation
- concurrent invocation safety

3. Validation
- malformed structured output
- missing decisions
- unknown clusters
- duplicate references
- cross-cluster references
- unsupported evidence
- wrong confidence values
- provider skip behavior

4. Streaming
- strict NDJSON sequences
- exactly one terminal event
- malformed stream handling
- cancellation and producer cleanup
- no raw graph-event exposure
- previous-result preservation

5. Privacy and security
- no API keys
- no prompts
- no raw model output
- no response or model IDs
- no exception details
- no chain-of-thought
- no provider clients or locks in serialized state

6. Frontend
- accessible controls
- keyboard focus
- loading and error states
- stable list keys
- no duplicate IDs
- responsive behavior

7. Tests
- identify missing edge cases
- distinguish tested guarantees from assumptions
- inspect whether mocks bypass production behavior

Run read-only verification commands where appropriate.

Required output:

A. Verdict:
- APPROVE
- APPROVE WITH NON-BLOCKING NOTES
- BLOCK

B. Findings ordered by severity

For every finding include:

- exact file and symbol
- reproduced or inferred
- concrete failure scenario
- minimal correction

C. Correctness assessment
D. Concurrency and cancellation assessment
E. Privacy assessment
F. Test-quality assessment
G. Exact verification results

Do not edit anything.
Stop after the review.
