import type {
  AudienceReviewRun,
  ReviewJourneyEventCode,
} from '../api/audienceReviewTypes'
import { formatPercent, humanizeCode } from '../formatters.js'
import { EDIT_DROP_TEXT } from '../reviewUi.js'

const JOURNEY_EVENT_TEXT: Partial<Record<ReviewJourneyEventCode, string>> = {
  review_requested: 'Review requested',
  analyst_approved: 'Analyst approved',
  analyst_rejected: 'Analyst rejected',
  analyst_edit_requested: 'Edit requested',
  edited_decision_received: 'Edited decision received',
  edited_decision_validated: 'Edited decision validated',
  edited_audience_published: 'Edited audience published',
  analyst_edit_dropped: 'Analyst edit dropped',
  review_expired: 'Review expired',
  audience_published: 'Audience published',
}

interface ReviewOutcomeSummaryProps {
  run: AudienceReviewRun
}

export function ReviewOutcomeSummary({ run }: ReviewOutcomeSummaryProps) {
  const metrics = run.automatic_workflow_metrics
  const hasOutcomes =
    run.published_audiences.length > 0 ||
    run.rejected_reviews.length > 0 ||
    run.edit_validation_drops.length > 0 ||
    run.expired_reviews.length > 0 ||
    run.provider_skips.length > 0 ||
    run.validation_drops.length > 0
  const journeyItems = run.journey.flatMap((trace) => {
    const seen = new Set<string>()
    const editedPublicationRecorded = trace.events.some(
      (event) => event.code === 'edited_audience_published',
    )
    return trace.events.flatMap((event) => {
      if (editedPublicationRecorded && event.code === 'audience_published') return []
      const label = JOURNEY_EVENT_TEXT[event.code]
      if (!label || seen.has(label)) return []
      seen.add(label)
      return [{ key: `${trace.trace_id}:${event.sequence}`, label, name: trace.cluster_name }]
    })
  })

  if (!hasOutcomes && run.journey.length === 0) return null

  return (
    <div className="review-outcomes">
      <section aria-labelledby="review-outcomes-title">
        <div className="section-heading compact-heading">
          <div>
            <p className="eyebrow">Committed outcomes</p>
            <h2 id="review-outcomes-title">Review results</h2>
          </div>
          <p>Only terminal, authoritative outcomes are shown.</p>
        </div>

        <div className="outcome-grid">
          {run.published_audiences.map((item) => (
            <article className="outcome-card outcome-published" key={item.review_id}>
              <span>{item.publication_source === 'analyst_edit' ? 'Analyst-edited audience' : 'Published original'}</span>
              <h3>{item.audience.name}</h3>
              <p>{item.audience.description}</p>
            </article>
          ))}
          {run.rejected_reviews.map((item) => (
            <article className="outcome-card outcome-rejected" key={item.review_id}>
              <span>Analyst rejected</span>
              <h3>{item.cluster_name}</h3>
              <p>{humanizeCode(item.reason_code)}</p>
            </article>
          ))}
          {run.edit_validation_drops.map((item) => (
            <article className="outcome-card outcome-dropped" key={item.review_id}>
              <span>Analyst edit dropped</span>
              <h3>{item.cluster_name}</h3>
              <p>{EDIT_DROP_TEXT[item.drop_code]}</p>
            </article>
          ))}
          {run.expired_reviews.map((item) => (
            <article className="outcome-card outcome-expired" key={item.review_id}>
              <span>Review expired</span>
              <h3>{item.cluster_name}</h3>
              <p>No analyst decision was committed before the run expired.</p>
            </article>
          ))}
          {run.provider_skips.map((item) => (
            <article className="outcome-card outcome-skipped" key={item.trace_id}>
              <span>Provider skip</span>
              <h3>{item.cluster_name}</h3>
              <p>{item.reason}</p>
            </article>
          ))}
          {run.validation_drops.map((item) => (
            <article className="outcome-card outcome-dropped" key={item.trace_id}>
              <span>Validation drop</span>
              <h3>{item.cluster_name ?? 'Unmatched decision'}</h3>
              <p>{humanizeCode(item.drop_code)}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="review-metric-panel" aria-labelledby="review-metrics-title">
        <h2 id="review-metrics-title">Workflow metrics</h2>
        <dl>
          <div><dt>Review candidates</dt><dd>{run.progress.total_reviews}</dd></div>
          <div><dt>Completed reviews</dt><dd>{run.progress.completed_reviews}</dd></div>
          <div><dt>Provider calls</dt><dd>{metrics.provider_call_count}</dd></div>
          <div><dt>Validated audiences</dt><dd>{metrics.final_valid_decision_count}</dd></div>
          <div><dt>Validation issues</dt><dd>{metrics.validation_issue_count}</dd></div>
          <div><dt>Provider time</dt><dd>{metrics.provider_elapsed_seconds.toFixed(1)}s</dd></div>
          <div><dt>Validation yield</dt><dd>{formatPercent(metrics.initial_decision_count === 0 ? 0 : metrics.final_valid_decision_count / metrics.initial_decision_count)}</dd></div>
        </dl>
      </section>

      <section className="review-journey" aria-labelledby="review-journey-title">
        <h2 id="review-journey-title">Review journey</h2>
        <p>Public workflow events only. Private guidance and model reasoning are not included.</p>
        <ol>
          {journeyItems.map((event) => (
            <li key={event.key}>
              <span>{event.label}</span>
              <strong>{event.name ?? 'Audience decision'}</strong>
            </li>
          ))}
        </ol>
      </section>
    </div>
  )
}
