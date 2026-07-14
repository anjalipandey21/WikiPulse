import type { CSSProperties } from 'react'

import type { TopicClusterResponse } from '../api/types'
import {
  formatExactNumber,
  formatPageviews,
  formatPercent,
} from '../formatters'
import { PageviewSparkline } from './PageviewSparkline'

interface TopicLandscapeProps {
  topics: TopicClusterResponse[]
  selectedTopicId: string
  onSelectTopic: (topicId: string) => void
}

export function TopicLandscape({
  topics,
  selectedTopicId,
  onSelectTopic,
}: TopicLandscapeProps) {
  const selectedTopic =
    topics.find((topic) => topic.id === selectedTopicId) ?? topics[0]
  const maximumViews = Math.max(...topics.map((topic) => topic.total_views), 1)

  return (
    <section className="dashboard-section" aria-labelledby="landscape-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">View 01</p>
          <h2 id="landscape-title">Weekly Topic Landscape</h2>
        </div>
        <p>
          Coherent article groups ranked by pageviews across the latest seven
          complete days.
        </p>
      </div>

      <div className="topic-layout">
        <ol className="topic-ranking" aria-label="Topics ranked by pageviews">
          {topics.map((topic, index) => {
            const relativeWidth = Math.max(
              (topic.total_views / maximumViews) * 100,
              4,
            )
            const style = {
              '--topic-share': `${relativeWidth}%`,
            } as CSSProperties

            return (
              <li key={topic.id}>
                <button
                  type="button"
                  className="topic-rank-button"
                  aria-pressed={topic.id === selectedTopic.id}
                  onClick={() => onSelectTopic(topic.id)}
                  style={style}
                >
                  <span className="topic-rank-number">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <span className="topic-rank-copy">
                    <strong>{topic.name}</strong>
                    <span>{formatPageviews(topic.total_views)}</span>
                  </span>
                  <span className="topic-rank-bar" aria-hidden="true" />
                </button>
              </li>
            )
          })}
        </ol>

        <article className="topic-detail" aria-labelledby="selected-topic-title">
          <header className="topic-detail-header">
            <div>
              <p className="eyebrow">Selected topic</p>
              <h3 id="selected-topic-title">{selectedTopic.name}</h3>
            </div>
            <div className="confidence-chip">
              <span>Topic confidence</span>
              <strong>
                {selectedTopic.confidence_score === null
                  ? 'Not available'
                  : formatPercent(selectedTopic.confidence_score)}
              </strong>
            </div>
          </header>

          <p className="topic-description">
            {selectedTopic.description ??
              'A deterministic semantic cluster of related Wikipedia articles.'}
          </p>

          <div className="topic-facts" aria-label="Selected topic metrics">
            <span>
              <strong>{formatExactNumber(selectedTopic.article_count)}</strong>{' '}
              articles
            </span>
            <span>
              <strong>{formatPageviews(selectedTopic.total_views)}</strong>
            </span>
          </div>

          {selectedTopic.keywords.length > 0 ? (
            <ul className="tag-list" aria-label="Topic keywords">
              {selectedTopic.keywords.map((keyword) => (
                <li key={keyword}>{keyword}</li>
              ))}
            </ul>
          ) : null}

          <div className="article-list">
            <h4>Article signals</h4>
            <ul>
              {selectedTopic.articles.map((article) => (
                <li className="article-signal" key={article.url}>
                  <div className="article-signal-main">
                    <a
                      href={article.url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {article.title}
                      <span className="external-mark" aria-hidden="true">↗</span>
                    </a>
                    <span>{formatPageviews(article.weekly_views)}</span>
                  </div>
                  <PageviewSparkline
                    dailyViews={article.daily_views}
                    articleTitle={article.title}
                  />
                  {article.summary ? (
                    <details className="article-summary">
                      <summary>Read summary</summary>
                      <p>{article.summary}</p>
                    </details>
                  ) : null}
                </li>
              ))}
            </ul>
          </div>
        </article>
      </div>
    </section>
  )
}
