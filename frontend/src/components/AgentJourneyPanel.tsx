import { useEffect, useRef } from 'react'

import type {
  AudienceDecisionTraceResponse,
  AudienceTraceEventResponse,
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

const EVENT_LABEL: Record<AudienceTraceEventCode, string> = {
  generation_requested: 'Generation started',
  decision_received: 'Decision received',
  validation_passed: 'Validation passed',
  validation_failed: 'Validation failed',
  revision_requested: 'Revision requested',
  revision_failed: 'Revision failed',
  audience_published: 'Audience published',
  provider_skipped: 'Audience skipped',
  decision_dropped: 'Decision dropped',
}

const SUMMARY_STEP: Partial<Record<AudienceTraceEventCode, string>> = {
  validation_passed: 'Validated',
  validation_failed: 'Validation failed',
  revision_failed: 'Revision failed',
  audience_published: 'Published',
  provider_skipped: 'Skipped',
  decision_dropped: 'Dropped',
}

function summarizeTraceEvents(
  events: readonly AudienceTraceEventResponse[],
): string {
  const steps: string[] = []
  let pendingRequest: 'initial' | 'revision' | null = null

  for (const event of events) {
    if (event.code === 'generation_requested') {
      steps.push('Generation requested')
      pendingRequest = 'initial'
      continue
    }

    if (event.code === 'revision_requested') {
      steps.push('Revision requested')
      pendingRequest = 'revision'
      continue
    }

    if (event.code === 'decision_received') {
      if (pendingRequest) {
        steps[steps.length - 1] = pendingRequest === 'initial' ? 'Generated' : 'Revised'
        pendingRequest = null
      } else {
        steps.push('Received')
      }
      continue
    }

    const step = SUMMARY_STEP[event.code]
    if (step) steps.push(step)
  }

  return steps.join(' → ')
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
    if (target.tagName === 'DETAILS') {
      (target as HTMLDetailsElement).open = true
    }
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
                  <details className="journey-card" id={trace.trace_id}>
                    <summary className="journey-card-summary" data-trace-heading>
                      <span className="journey-card-summary-copy">
                        <span className="journey-card-header">
                          <span
                            className="journey-card-title"
                            role="heading"
                            aria-level={3}
                          >
                            {trace.cluster_name ?? 'Unmatched audience decision'}
                          </span>
                          <span
                            className={`journey-outcome outcome-${trace.final_outcome}`}
                          >
                            {OUTCOME_TEXT[trace.final_outcome]}
                          </span>
                        </span>
                        <span className="journey-path">
                          {summarizeTraceEvents(trace.events)}
                        </span>
                      </span>
                      <span className="journey-disclosure-icon" aria-hidden="true">
                        +
                      </span>
                    </summary>

                    <div className="journey-card-content">
                      <ol className="journey-events">
                        {trace.events.map((event) => (
                          <li key={event.sequence}>
                            <span className="journey-marker" aria-hidden="true" />
                            <div>
                              <p className="journey-event-label">
                                {EVENT_LABEL[event.code]}
                              </p>
                              <p>{EVENT_TEXT[event.code]}</p>
                              {event.issues.length > 0 ? (
                                <p className="journey-issue-count">
                                  {event.issues.length}{' '}
                                  {event.issues.length === 1 ? 'issue was' : 'issues were'}{' '}
                                  recorded for this validation step.
                                </p>
                              ) : null}
                            </div>
                          </li>
                        ))}
                      </ol>
                    </div>
                  </details>
                </li>
              ))}
            </ol>
          )}
        </div>
      </details>
    </section>
  )
}
