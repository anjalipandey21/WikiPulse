import type { AudienceSegmentResponse } from '../api/types'
import {
  formatPageviews,
  formatPercent,
  formatSizeIndex,
} from '../formatters'

interface AudiencePortfolioProps {
  segments: AudienceSegmentResponse[]
  topicNames: ReadonlyMap<string, string>
}

export function AudiencePortfolio({
  segments,
  topicNames,
}: AudiencePortfolioProps) {
  return (
    <section className="dashboard-section" aria-labelledby="portfolio-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">View 02</p>
          <h2 id="portfolio-title">Emerging Audience Portfolio</h2>
        </div>
        <p>
          Commercial interpretations grounded in safe, traceable topic evidence.
        </p>
      </div>

      <div className="audience-grid">
        {segments.map((segment) => {
          const topicLabels = segment.topic_cluster_ids.map(
            (topicId) => topicNames.get(topicId) ?? topicId,
          )
          const boundedSize = Math.min(Math.max(segment.size_index, 0), 100)

          return (
            <article className="audience-card" key={segment.id}>
              <header className="audience-card-header">
                <div>
                  <p className="audience-source">
                    {topicLabels.join(' · ')}
                  </p>
                  <h3>{segment.name}</h3>
                </div>
                <span className={`power-badge power-${segment.buying_power}`}>
                  {segment.buying_power} buying power
                </span>
              </header>

              <p className="audience-description">{segment.description}</p>

              <div className="audience-score-grid">
                <div>
                  <span>Size index</span>
                  <strong>{formatSizeIndex(segment.size_index)}</strong>
                  <progress
                    value={boundedSize}
                    max="100"
                    aria-label={`${segment.name} size index ${formatSizeIndex(segment.size_index)}`}
                  />
                </div>
                <div>
                  <span>Commercial confidence</span>
                  <strong>{formatPercent(segment.commercial_confidence)}</strong>
                </div>
              </div>

              <dl className="audience-reasons">
                <div>
                  <dt>Why they matter</dt>
                  <dd>{segment.commercial_confidence_reason}</dd>
                </div>
                <div>
                  <dt>Buying power signal</dt>
                  <dd>{segment.buying_power_reason}</dd>
                </div>
              </dl>

              {segment.brand_categories.length > 0 ? (
                <ul className="tag-list" aria-label="Relevant brand categories">
                  {segment.brand_categories.map((category) => (
                    <li key={category}>{category}</li>
                  ))}
                </ul>
              ) : null}

              <div className="evidence-list">
                <h4>Supporting evidence</h4>
                <ul>
                  {segment.supporting_articles.map((article) => (
                    <li key={article.url}>
                      <a href={article.url} target="_blank" rel="noreferrer">
                        <span>{article.title}</span>
                        <span>{formatPageviews(article.weekly_views)} ↗</span>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
