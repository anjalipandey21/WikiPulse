import type { AudienceAnalysisMetricsResponse } from '../api/types'
import { formatCompactNumber, formatExactNumber } from '../formatters'

interface MetricSummaryProps {
  metrics: AudienceAnalysisMetricsResponse
}

export function MetricSummary({ metrics }: MetricSummaryProps) {
  const topic = metrics.topic_analysis
  const funnel = metrics.audience_funnel
  const items = [
    {
      label: 'Selected pageviews',
      value: formatCompactNumber(topic.selected_pageviews),
      exact: formatExactNumber(topic.selected_pageviews),
    },
    {
      label: 'Topic clusters',
      value: formatExactNumber(topic.topic_cluster_count),
    },
    {
      label: 'Clustered articles',
      value: formatExactNumber(topic.clustered_article_count),
    },
    {
      label: 'Audience segments',
      value: formatExactNumber(funnel.final_segment_count),
    },
  ]

  return (
    <dl className="metric-summary" aria-label="Analysis summary">
      {items.map((item) => (
        <div className="metric-summary-item" key={item.label}>
          <dt>{item.label}</dt>
          <dd title={item.exact}>{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}
