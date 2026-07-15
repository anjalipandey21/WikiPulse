import type { AudienceAnalysisResponse } from '../api/types'
import { buildTickerInsights, type TickerInsight } from '../stitchEnhancements.js'

interface LiveAttentionTickerProps {
  result: AudienceAnalysisResponse | null
}

export function LiveAttentionTicker({
  result,
}: LiveAttentionTickerProps) {
  if (result === null) return null

  const insights = buildTickerInsights(result)
  if (insights.length === 0) return null

  const accessibleSummary = insights
    .map((insight) => `${insight.label}: ${insight.value}`)
    .join('. ')

  return (
    <section
      className="live-attention-ticker"
      aria-label="Current weekly attention brief"
      tabIndex={0}
    >
      <p className="visually-hidden">{accessibleSummary}.</p>
      <div className="live-attention-track" aria-hidden="true">
        <TickerInsightList insights={insights} />
        <TickerInsightList insights={insights} duplicate />
      </div>
    </section>
  )
}

function TickerInsightList({
  insights,
  duplicate = false,
}: {
  insights: readonly TickerInsight[]
  duplicate?: boolean
}) {
  return (
    <ul className="live-attention-list" aria-hidden={duplicate || undefined}>
      <li className="live-attention-label">
        <span className="live-attention-dot" />
        <strong>Live attention</strong>
        <span>Weekly brief</span>
      </li>
      {insights.map((insight) => (
        <li key={`${insight.label}-${insight.value}`}>
          <span>{insight.label}</span>
          <strong>{insight.value}</strong>
        </li>
      ))}
    </ul>
  )
}
