import type { AudienceAnalysisMetricsResponse } from '../api/types'
import { formatCompactNumber, formatExactNumber } from '../formatters.js'
import { CountUpMetric } from './CountUpMetric.js'

interface MetricSummaryProps {
  metrics: AudienceAnalysisMetricsResponse
}

export function MetricSummary({ metrics }: MetricSummaryProps) {
  const topic = metrics.topic_analysis
  const funnel = metrics.audience_funnel
  const items = [
    {
      label: 'Selected pageviews',
      value: topic.selected_pageviews,
      format: formatCompactNumber,
      exact: formatExactNumber(topic.selected_pageviews),
    },
    {
      label: 'Topic clusters',
      value: topic.topic_cluster_count,
      format: formatExactNumber,
    },
    {
      label: 'Clustered articles',
      value: topic.clustered_article_count,
      format: formatExactNumber,
    },
    {
      label: 'Audience segments',
      value: funnel.final_segment_count,
      format: formatExactNumber,
    },
  ]

  return (
    <dl className="metric-summary" aria-label="Analysis summary">
      {items.map((item) => (
        <div className="metric-summary-item" key={item.label}>
          <dt>{item.label}</dt>
          <dd title={item.exact}>
            <CountUpMetric value={item.value} format={item.format} />
          </dd>
        </div>
      ))}
    </dl>
  )
}
