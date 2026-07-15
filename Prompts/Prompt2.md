Prompt 2: LangGraph bounded orchestration
Act as a staff-level LangGraph and Python engineer.

Implement the approved behavior-preserving LangGraph orchestration for the WikiPulse audience workflow.

Requirements:

1. Keep the existing public function unchanged:

async def run_audience_workflow(
    preparation: AudiencePreparation,
    provider: AudienceGenerationProvider,
) -> AudienceWorkflowResult

2. Reuse the existing helpers for:

- validation
- indexing
- revision-request construction
- drop creation
- result merging
- ordering
- token and timing metrics

Do not duplicate business logic inside graph routing functions.

3. Build one private compiled StateGraph with these stages:

START
→ empty preparation check
→ generate_initial
→ validate_initial
→ optional revise_once
→ validate_revision
→ merge_and_build_result
→ END

4. Enforce MAX_REVISIONS = 1 structurally.

There must be no graph edge from revision validation back to the revision node.

5. Pass the provider through invocation-scoped LangGraph runtime context.

Do not serialize provider clients, locks, callbacks, or network objects in graph state.

6. Preserve existing behavior:

- empty preparation performs zero provider calls
- initial provider failure propagates unchanged
- revision provider failure becomes explicit drops
- valid initial results preserve identity
- only known invalid clusters are revised
- unknown outputs are dropped but never revised
- ordering and metrics remain identical
- edited or revised output cannot replace unrelated valid output

7. Add focused parity tests for:

- empty preparation
- all-valid output
- targeted revision
- unknown output
- revision failure
- still-invalid revision
- fatal initial provider error
- exactly one revision call
- validation against the original preparation and then the revision subset

Run:

- focused workflow tests
- full backend unittest discovery
- Python compilation
- git diff --check

Do not modify the API, frontend, provider prompts, or unrelated files.
Do not commit or push.

At completion, report exact files changed and verification results.
