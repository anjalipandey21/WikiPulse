import { useEffect, useRef } from 'react'

import type {
  AudienceDecisionTraceResponse,
  AudienceTraceEventCode,
  AudienceTraceOutcome,
} from '../api/types'

const EVENT_TEXT: Record<AudienceTraceEventCode, string> = {
  generation_requested:
    'This cluster was included in the initial structured-generation batch.',
  decision_received: 'A structured decision was returned for validation.',
  validation_passed: 'Deterministic Python validation accepted the decision.',
  validation_failed: 'Deterministic Python validation found recorded issues.',
  revision_requested:
    'The cluster and its exact issues were included in the one allowed revision batch.',
  revision_failed: 'The revision request failed safely without replacing valid results.',
  audience_published: 'The validated audience was added to the portfolio.',
  provider_skipped: 'The validated decision was to skip audience creation.',
  decision_dropped: 'The unresolved decision was excluded from the portfolio.',
}

const OUTCOME_TEXT: Record<AudienceTraceOutcome, string> = {
  published: 'Published audience',
  provider_skipped: 'Provider skip',
  validation_dropped: 'Validation drop',
}

interface AgentJourneyPanelProps {
  traces: readonly AudienceDecisionTraceResponse[]
  traceRequest: AgentJourneyRequest | null
}

export interface AgentJourneyRequest {
  traceId: string
  sequence: number
}

export function AgentJourneyPanel({
  traces,
  traceRequest,
}: AgentJourneyPanelProps) {
  const disclosureRef = useRef<HTMLDetailsElement>(null)

  useEffect(() => {
    if (!traceRequest) return
    const disclosure = disclosureRef.current
    const target = document.getElementById(traceRequest.traceId)
    if (!disclosure || !target || !disclosure.contains(target)) return

    disclosure.open = true
    const frame = requestAnimationFrame(() => {
      target.scrollIntoView({ block: 'start' })
      target
        .querySelector<HTMLElement>('[data-trace-heading]')
        ?.focus({ preventScroll: true })
    })
    return () => cancelAnimationFrame(frame)
  }, [traceRequest])

  return (
    <section className="journey-section" aria-labelledby="journey-title">
      <h2 className="visually-hidden" id="journey-title">
        Audience agent journey
      </h2>
      <details className="agent-journey" ref={disclosureRef}>
        <summary>
          <span className="journey-summary-copy">
            <span className="eyebrow">Bounded workflow</span>
            <strong className="journey-summary-title">View agent journey</strong>
          </span>
          <span className="journey-summary-count">
            {traces.length} {traces.length === 1 ? 'outcome' : 'outcomes'}
          </span>
        </summary>

        <div className="journey-content">
          <p className="journey-note">
            Recorded generation and validation outcomes only. This is not private
            model reasoning.
          </p>

          {traces.length === 0 ? (
            <p className="journey-empty">
              No commercially eligible clusters were sent for audience generation.
            </p>
          ) : (
            <ol className="journey-list">
              {traces.map((trace) => (
                <li key={trace.trace_id}>
                  <article className="journey-card" id={trace.trace_id}>
                    <header className="journey-card-header">
                      <div>
                        <p className="journey-cluster-id">
                          {trace.source_known ? trace.cluster_id : 'Unmatched output'}
                        </p>
                        <h3 tabIndex={-1} data-trace-heading>
                          {trace.cluster_name ?? 'Unknown source cluster'}
                        </h3>
                      </div>
                      <span className={`journey-outcome outcome-${trace.final_outcome}`}>
                        {OUTCOME_TEXT[trace.final_outcome]}
                      </span>
                    </header>

                    <ol className="journey-events">
                      {trace.events.map((event) => (
                        <li key={event.sequence}>
                          <span className="journey-marker" aria-hidden="true" />
                          <div>
                            <p className="journey-event-meta">
                              <span>{event.phase}</span>
                              <code>{event.code}</code>
                            </p>
                            <p>{EVENT_TEXT[event.code]}</p>
                            {event.outcome_code ? (
                              <p className="journey-code">
                                Outcome code: <code>{event.outcome_code}</code>
                              </p>
                            ) : null}
                            {event.issues.length > 0 ? (
                              <ul className="journey-issues" aria-label="Validation issues">
                                {event.issues.map((issue, index) => (
                                  <li key={`${issue.code}-${issue.reference_id ?? 'none'}-${index}`}>
                                    <code>{issue.code}</code>
                                    {issue.reference_id ? (
                                      <span>
                                        {' '}· reference <code>{issue.reference_id}</code>
                                      </span>
                                    ) : null}
                                  </li>
                                ))}
                              </ul>
                            ) : null}
                          </div>
                        </li>
                      ))}
                    </ol>
                  </article>
                </li>
              ))}
            </ol>
          )}
        </div>
      </details>
    </section>
  )
}
