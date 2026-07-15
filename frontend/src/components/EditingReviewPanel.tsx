import type { EditingReview } from '../api/audienceReviewTypes'

interface EditingReviewPanelProps {
  review: EditingReview
  onRefresh: () => void
}

export function EditingReviewPanel({
  review,
  onRefresh,
}: EditingReviewPanelProps) {
  return (
    <section className="status-panel review-editing" aria-live="polite" aria-busy="true">
      <div className="status-icon" aria-hidden="true"><span className="loading-ring" /></div>
      <p className="eyebrow">Audience {review.position} of {review.total_reviews}</p>
      <h2>Applying analyst edit…</h2>
      <p>
        WikiPulse is regenerating one bounded recommendation for {review.cluster_name}.
        The original recommendation remains unchanged until validation commits an outcome.
      </p>
      <button className="button button-secondary" type="button" onClick={onRefresh}>
        Check status
      </button>
    </section>
  )
}
