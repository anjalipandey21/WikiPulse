import { useState } from 'react'

import type {
  AnalystEditableField,
  PendingReview,
  RejectReasonCode,
} from '../api/audienceReviewTypes'
import { formatPageviews, formatPercent, formatSizeIndex } from '../formatters.js'
import { normalizeAnalystFeedback } from '../reviewUi.js'

const REJECT_REASONS: ReadonlyArray<{
  value: RejectReasonCode
  label: string
}> = [
  { value: 'safety_concern', label: 'Safety concern' },
  { value: 'insufficient_evidence', label: 'Insufficient evidence' },
  { value: 'misleading_recommendation', label: 'Misleading recommendation' },
  { value: 'not_commercially_useful', label: 'Not commercially useful' },
  { value: 'duplicate_audience', label: 'Duplicate audience' },
  { value: 'other', label: 'Other' },
]

const EDIT_GROUPS: ReadonlyArray<{
  value: AnalystEditableField
  label: string
}> = [
  { value: 'audience_positioning', label: 'Audience positioning' },
  { value: 'supporting_evidence', label: 'Supporting evidence' },
  { value: 'buying_power', label: 'Buying power' },
  { value: 'brand_categories', label: 'Brand categories' },
  { value: 'commercial_confidence', label: 'Commercial confidence' },
]

interface PendingReviewCardProps {
  review: PendingReview
  busyAction: 'approve' | 'reject' | 'edit' | null
  onApprove: () => void
  onReject: (reason: RejectReasonCode, privateNote?: string) => void
  onEdit: (feedback: string, fields: AnalystEditableField[]) => void
}

export function PendingReviewCard({
  review,
  busyAction,
  onApprove,
  onReject,
  onEdit,
}: PendingReviewCardProps) {
  const [panel, setPanel] = useState<'reject' | 'edit' | null>(null)
  const [reason, setReason] = useState<RejectReasonCode | ''>('')
  const [privateNote, setPrivateNote] = useState('')
  const [feedback, setFeedback] = useState('')
  const [fields, setFields] = useState<AnalystEditableField[]>([])
  const recommendation = review.original_recommendation
  const normalizedFeedback = normalizeAnalystFeedback(feedback)
  const feedbackLength = Array.from(normalizedFeedback).length
  const feedbackValid = feedbackLength >= 10 && feedbackLength <= 600
  const rejectValid = reason !== '' && (reason !== 'other' || privateNote.trim() !== '')
  const busy = busyAction !== null

  function submitReject() {
    if (reason === '' || (reason === 'other' && privateNote.trim() === '')) return
    onReject(reason, privateNote.trim() || undefined)
    setPrivateNote('')
    setReason('')
    setPanel(null)
  }

  function submitEdit() {
    if (!feedbackValid || fields.length === 0) return
    onEdit(normalizedFeedback, fields)
    setFeedback('')
    setFields([])
    setPanel(null)
  }

  function toggleField(field: AnalystEditableField) {
    setFields((current) =>
      current.includes(field)
        ? current.filter((item) => item !== field)
        : [...current, field],
    )
  }

  return (
    <article className="review-card" aria-labelledby="review-audience-name">
      <header className="review-card-heading">
        <div>
          <p className="eyebrow">
            Audience {review.position} of {review.total_reviews}
          </p>
          <h2 id="review-audience-name">{recommendation.name}</h2>
          <p className="review-cluster-name">From {review.cluster_name}</p>
        </div>
        <span className={`power-badge power-${recommendation.buying_power}`}>
          {recommendation.buying_power} buying power
        </span>
      </header>

      <p className="audience-description">{recommendation.description}</p>

      <dl className="review-metrics">
        <div><dt>Weekly reach</dt><dd>{formatPageviews(review.cluster_pageviews)}</dd></div>
        <div><dt>Articles</dt><dd>{review.article_count}</dd></div>
        <div><dt>Size index</dt><dd>{formatSizeIndex(review.size_index)}</dd></div>
        <div><dt>Topic confidence</dt><dd>{formatPercent(review.topic_confidence)}</dd></div>
        <div><dt>Commercial confidence</dt><dd>{formatPercent(recommendation.commercial_confidence)}</dd></div>
      </dl>

      <dl className="audience-reasons review-reasons">
        <div>
          <dt>Buying-power assessment</dt>
          <dd>{recommendation.buying_power_reason}</dd>
        </div>
        <div>
          <dt>Commercial-confidence rationale</dt>
          <dd>{recommendation.commercial_confidence_reason}</dd>
        </div>
      </dl>

      {recommendation.brand_categories.length > 0 ? (
        <ul className="tag-list" aria-label="Recommended brand categories">
          {recommendation.brand_categories.map((category) => (
            <li key={category}>{category}</li>
          ))}
        </ul>
      ) : null}

      <div className="evidence-list review-evidence">
        <h3>Supporting evidence</h3>
        <ul>
          {review.evidence.map(({ reference_id, article }) => (
            <li key={reference_id}>
              <a href={article.url} target="_blank" rel="noreferrer">
                <span>{article.title}</span>
                <span>{formatPageviews(article.weekly_views)} ↗</span>
              </a>
              {article.summary ? <p>{article.summary}</p> : null}
            </li>
          ))}
        </ul>
      </div>

      <div className="review-actions" aria-label="Analyst actions">
        <button
          className="button button-primary"
          type="button"
          disabled={busy}
          onClick={onApprove}
        >
          {busyAction === 'approve' ? 'Approving…' : 'Approve'}
        </button>
        <button
          className="button button-secondary"
          type="button"
          disabled={busy}
          aria-expanded={panel === 'reject'}
          onClick={() => setPanel(panel === 'reject' ? null : 'reject')}
        >
          Reject
        </button>
        <button
          className="button button-secondary"
          type="button"
          disabled={busy || !review.edit_available}
          aria-expanded={panel === 'edit'}
          onClick={() => setPanel(panel === 'edit' ? null : 'edit')}
          title={review.edit_available ? undefined : 'The one analyst edit has been used.'}
        >
          Request edit
        </button>
      </div>

      {panel === 'reject' ? (
        <section className="review-action-panel" aria-labelledby="reject-title">
          <h3 id="reject-title">Reject this audience</h3>
          <p>Rejection is terminal for this candidate and does not affect later reviews.</p>
          <label>
            Reason
            <select
              value={reason}
              onChange={(event) => setReason(event.target.value as RejectReasonCode | '')}
              disabled={busy}
            >
              <option value="">Select a reason</option>
              {REJECT_REASONS.map((item) => (
                <option value={item.value} key={item.value}>{item.label}</option>
              ))}
            </select>
          </label>
          <label>
            Private note <span>(optional unless reason is Other)</span>
            <textarea
              value={privateNote}
              maxLength={240}
              rows={3}
              onChange={(event) => setPrivateNote(event.target.value)}
              disabled={busy}
            />
          </label>
          <p className="private-input-note">Private notes are not shown in results or the journey.</p>
          <button
            className="button button-danger"
            type="button"
            disabled={busy || !rejectValid}
            onClick={submitReject}
          >
            Confirm rejection
          </button>
        </section>
      ) : null}

      {panel === 'edit' ? (
        <section className="review-action-panel" aria-labelledby="edit-title">
          <h3 id="edit-title">Request one bounded edit</h3>
          <p>This consumes the candidate’s only analyst-edit allowance.</p>
          <fieldset className="edit-groups">
            <legend>What may change?</legend>
            {EDIT_GROUPS.map((group) => (
              <label key={group.value}>
                <input
                  type="checkbox"
                  checked={fields.includes(group.value)}
                  onChange={() => toggleField(group.value)}
                  disabled={busy}
                />
                {group.label}
              </label>
            ))}
          </fieldset>
          <label>
            Analyst guidance
            <textarea
              value={feedback}
              rows={5}
              maxLength={600}
              aria-describedby="feedback-help feedback-count"
              onChange={(event) => setFeedback(event.target.value)}
              disabled={busy}
            />
          </label>
          <div className="field-help-row">
            <p id="feedback-help" className="private-input-note">
              Private guidance is used only for this edit and is not shown publicly.
            </p>
            <span id="feedback-count" aria-live="polite">{feedbackLength} / 600</span>
          </div>
          <button
            className="button button-primary"
            type="button"
            disabled={busy || fields.length === 0 || !feedbackValid}
            onClick={submitEdit}
          >
            Apply analyst edit
          </button>
        </section>
      ) : null}
    </article>
  )
}
