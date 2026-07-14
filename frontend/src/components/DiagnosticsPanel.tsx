import type { AudienceAnalysisResponse } from '../api/types'
import {
  formatExactNumber,
  formatPageviews,
  humanizeCode,
} from '../formatters'

interface DiagnosticsPanelProps {
  result: AudienceAnalysisResponse
  onViewTrace: (traceId: string) => void
}

export function DiagnosticsPanel({
  result,
  onViewTrace,
}: DiagnosticsPanelProps) {
  const topic = result.metrics.topic_analysis
  const funnel = result.metrics.audience_funnel
  const workflow = result.metrics.workflow
  const hasOutcomes =
    result.commercial_skips.length > 0 ||
    result.provider_skips.length > 0 ||
    result.validation_drops.length > 0 ||
    result.rejected_articles.length > 0 ||
    result.unclustered_articles.length > 0

  return (
    <section className="diagnostics-section" aria-labelledby="diagnostics-title">
      <details className="diagnostics">
        <summary>
          <span>
            <span className="eyebrow">Transparency</span>
            <strong id="diagnostics-title">Analysis details</strong>
          </span>
          <span>{hasOutcomes ? 'Review outcomes' : 'View run metrics'}</span>
        </summary>

        <div className="diagnostics-content">
          <div className="diagnostic-metrics">
            <div>
              <span>Fetched articles</span>
              <strong>{formatExactNumber(topic.fetched_article_count)}</strong>
            </div>
            <div>
              <span>Summary coverage</span>
              <strong>
                {formatExactNumber(topic.summary_available_article_count)} /{' '}
                {formatExactNumber(topic.selected_article_count)}
              </strong>
            </div>
            <div>
              <span>Eligible audience pageviews</span>
              <strong>{formatPageviews(funnel.commercial_eligible_pageviews)}</strong>
            </div>
            <div>
              <span>Represented pageviews</span>
              <strong>{formatPageviews(funnel.represented_audience_pageviews)}</strong>
            </div>
            <div>
              <span>Provider calls</span>
              <strong>{formatExactNumber(workflow.provider_call_count)}</strong>
            </div>
            <div>
              <span>Provider tokens</span>
              <strong>{formatExactNumber(workflow.provider_total_tokens)}</strong>
            </div>
          </div>

          <div className="diagnostic-groups">
            <div>
              <h3>Deterministic filtering</h3>
              {result.commercial_skips.length === 0 &&
              result.rejected_articles.length === 0 ? (
                <p className="muted-copy">No deterministic exclusions.</p>
              ) : (
                <ul className="diagnostic-list">
                  {result.commercial_skips.map((item) => (
                    <li key={`commercial-${item.cluster_id}`}>
                      <strong>{item.cluster_name}</strong>
                      <span>{humanizeCode(item.reason)}</span>
                    </li>
                  ))}
                  {result.rejected_articles.map((item) => (
                    <li key={`rejected-${item.article.url}`}>
                      <strong>{item.article.title}</strong>
                      <span>{humanizeCode(item.reason)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div>
              <h3>Generation outcomes</h3>
              {result.provider_skips.length === 0 &&
              result.validation_drops.length === 0 ? (
                <p className="muted-copy">No provider skips or validation drops.</p>
              ) : (
                <ul className="diagnostic-list">
                  {result.provider_skips.map((item) => (
                    <li key={`provider-${item.cluster_id}`}>
                      <strong>{item.cluster_name}</strong>
                      <span>{item.reason}</span>
                      <a
                        className="journey-link"
                        href={`#${item.trace_id}`}
                        onClick={() => onViewTrace(item.trace_id)}
                      >
                        View agent journey
                      </a>
                    </li>
                  ))}
                  {result.validation_drops.map((item, index) => (
                    <li key={`drop-${item.cluster_id}-${item.phase}-${index}`}>
                      <strong>
                        {item.source_known ? item.cluster_id : 'Unmatched output'}
                      </strong>
                      <span>
                        {humanizeCode(item.drop_code)} · {item.phase}
                        {item.issues.length > 0
                          ? ` · ${item.issues.map((issue) => humanizeCode(issue.code)).join(', ')}`
                          : ''}
                      </span>
                      <a
                        className="journey-link"
                        href={`#${item.trace_id}`}
                        onClick={() => onViewTrace(item.trace_id)}
                      >
                        View agent journey
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div>
              <h3>Unclustered articles</h3>
              {result.unclustered_articles.length === 0 ? (
                <p className="muted-copy">Every selected article was clustered.</p>
              ) : (
                <ul className="diagnostic-list diagnostic-links">
                  {result.unclustered_articles.map((article) => (
                    <li key={article.url}>
                      <a href={article.url} target="_blank" rel="noreferrer">
                        <strong>{article.title}</strong>
                        <span>{formatPageviews(article.weekly_views)} ↗</span>
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </details>
    </section>
  )
}
