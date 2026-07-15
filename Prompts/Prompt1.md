Prompt 1: Repository audit and architecture
Act as a staff-level Python and AI systems architect.

First read AGENTS.md and inspect the complete WikiPulse repository.

Trace the current pipeline end to end:

Wikimedia Pageviews API
→ article normalization
→ noise filtering
→ Wikipedia summary enrichment
→ TF-IDF keyword extraction
→ MiniLM embeddings
→ semantic clustering
→ topic finalization
→ commercial-safety routing
→ audience evidence preparation
→ LLM audience generation
→ deterministic validation
→ API response

Do not modify any files yet.

Identify:

1. The responsibility of every major module.
2. The input and output contracts between layers.
3. Which calculations are deterministic Python logic.
4. Which decisions are delegated to the LLM.
5. Current error, retry, ordering, and validation behavior.
6. The smallest place where LangGraph can orchestrate the workflow without moving business logic into graph nodes.
7. Existing tests that must continue passing.

Propose a minimal implementation plan that preserves:

- public APIs
- object identity
- ordering
- metrics
- token accounting
- provider error behavior
- MAX_REVISIONS = 1

Do not edit, install dependencies, commit, or push.

Return:

A. Current architecture
B. Exact workflow stages
C. Proposed LangGraph topology
D. Files that would change
E. Tests required for behavioral parity
F. Risks and non-goals
