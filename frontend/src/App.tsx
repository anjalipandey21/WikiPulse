import { useEffect, useRef, useState } from 'react'

import {
  AudienceAnalysisApiError,
  runAudienceAnalysis,
} from './api/audienceAnalysis'
import type { AudienceAnalysisResponse } from './api/types'
import {
  AgentJourneyPanel,
  type AgentJourneyRequest,
} from './components/AgentJourneyPanel'
import { AudiencePortfolio } from './components/AudiencePortfolio'
import { DiagnosticsPanel } from './components/DiagnosticsPanel'
import { MetricSummary } from './components/MetricSummary'
import { StatusPanel } from './components/StatusPanel'
import { TopicLandscape } from './components/TopicLandscape'
import { formatDate } from './formatters'

interface DisplayError {
  code: string
  message: string
}

export function App() {
  const [result, setResult] = useState<AudienceAnalysisResponse | null>(null)
  const [selectedTopicId, setSelectedTopicId] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<DisplayError | null>(null)
  const [traceRequest, setTraceRequest] = useState<AgentJourneyRequest | null>(null)
  const activeRequest = useRef<AbortController | null>(null)
  const isMounted = useRef(true)

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
      activeRequest.current?.abort()
    }
  }, [])

  async function handleRunAnalysis() {
    if (activeRequest.current) return

    const controller = new AbortController()
    activeRequest.current = controller
    setIsLoading(true)
    setError(null)

    try {
      const nextResult = await runAudienceAnalysis(controller.signal)
      if (!isMounted.current) return
      setResult(nextResult)
      setSelectedTopicId(nextResult.topics[0]?.id ?? '')
    } catch (caught) {
      if (!isMounted.current || isAbortError(caught)) return
      setError(toDisplayError(caught))
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null
        if (isMounted.current) setIsLoading(false)
      }
    }
  }

  function handleViewTrace(traceId: string) {
    setTraceRequest((current) => ({
      traceId,
      sequence: (current?.sequence ?? 0) + 1,
    }))
  }

  const topicNames = new Map(
    result?.topics.map((topic) => [topic.id, topic.name]) ?? [],
  )
  const period = result ? getAnalysisPeriod(result) : null
  const hasPartialResult = Boolean(
    result && (!result.is_publishable || result.validation_drops.length > 0),
  )

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header className="site-header">
        <a className="brand" href="/" aria-label="WikiPulse home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 36 36">
              <path d="M3 20h6l3-9 5 16 4-12 3 5h9" />
            </svg>
          </span>
          <span>
            <strong>WikiPulse</strong>
            <small>Wikipedia attention intelligence</small>
          </span>
        </a>

        {result ? (
          <button
            className="button button-secondary"
            type="button"
            onClick={handleRunAnalysis}
            disabled={isLoading}
          >
            {isLoading ? 'Refreshing…' : 'Refresh analysis'}
          </button>
        ) : null}
      </header>

      <main id="main-content" aria-busy={isLoading}>
        <h1 className="visually-hidden">WikiPulse audience analysis dashboard</h1>

        {!result && !isLoading && !error ? (
          <div className="welcome-state">
            <div className="welcome-art" aria-hidden="true">
              <span className="orbit orbit-one" />
              <span className="orbit orbit-two" />
              <span className="pulse-core">W</span>
            </div>
            <StatusPanel
              eyebrow="Seven days of public attention"
              title="See what the world is reading—and who it reveals."
              message="WikiPulse turns the latest seven complete days of English Wikipedia pageviews into coherent topics and commercially safe, evidence-backed audiences."
              actionLabel="Run weekly analysis"
              onAction={handleRunAnalysis}
            />
            <ul className="welcome-facts" aria-label="Analysis approach">
              <li><strong>Deterministic</strong><span>Pageviews and local clustering</span></li>
              <li><strong>Traceable</strong><span>Every insight links to evidence</span></li>
              <li><strong>Bounded AI</strong><span>One revision maximum</span></li>
            </ul>
          </div>
        ) : null}

        {!result && isLoading ? (
          <StatusPanel
            eyebrow="Analysis in progress"
            title="Reading the shape of the week"
            message="WikiPulse is fetching public pageviews, finding coherent topics, and validating safe commercial audiences. This is one serialized analysis request."
            icon={<span className="loading-ring" />}
            busy
          />
        ) : null}

        {!result && error ? (
          <StatusPanel
            eyebrow={`Error · ${error.code}`}
            title="The analysis could not be completed"
            message={error.message}
            actionLabel="Try again"
            onAction={handleRunAnalysis}
          />
        ) : null}

        {result ? (
          <div className="dashboard-results">
            <section className="result-intro" aria-labelledby="result-title">
              <div>
                <p className="eyebrow">Weekly intelligence brief</p>
                <h2 id="result-title">The week in attention</h2>
                <p>
                  English Wikipedia pageviews organized into semantic topics and
                  traceable audience opportunities.
                </p>
              </div>
              <div className="period-card">
                <span>Analysis period</span>
                <strong>
                  {period
                    ? `${formatDate(period.start)} – ${formatDate(period.end)}`
                    : 'Latest seven complete days'}
                </strong>
              </div>
            </section>

            {isLoading ? (
              <div className="notice notice-info" role="status">
                <span className="notice-dot" aria-hidden="true" />
                Refreshing the analysis. The previous result remains available.
              </div>
            ) : null}

            {error ? (
              <div className="notice notice-error" role="alert">
                <div>
                  <strong>Refresh failed</strong>
                  <span>{error.message} The previous result is still shown.</span>
                </div>
                <button
                  className="text-button"
                  type="button"
                  onClick={handleRunAnalysis}
                  disabled={isLoading}
                >
                  Try again
                </button>
              </div>
            ) : null}

            {hasPartialResult ? (
              <div className="notice notice-warning" role="status">
                <span className="notice-dot" aria-hidden="true" />
                Some audience decisions were excluded by deterministic validation.
                Valid topics and audiences remain available below.
              </div>
            ) : null}

            <MetricSummary metrics={result.metrics} />

            {result.topics.length > 0 ? (
              <TopicLandscape
                topics={result.topics}
                selectedTopicId={selectedTopicId}
                onSelectTopic={setSelectedTopicId}
              />
            ) : (
              <StatusPanel
                eyebrow="Weekly Topic Landscape"
                title="No coherent topics this week"
                message="The selected articles did not form clusters above the deterministic similarity threshold. Isolated articles remain available in Analysis details."
              />
            )}

            {result.audience_segments.length > 0 ? (
              <AudiencePortfolio
                segments={result.audience_segments}
                topicNames={topicNames}
                onViewTrace={handleViewTrace}
              />
            ) : (
              <StatusPanel
                eyebrow="Emerging Audience Portfolio"
                title="No publishable audiences this week"
                message="No topic produced a safe, commercially meaningful audience. Topic results and reason-coded exclusions are still available."
              />
            )}

            <AgentJourneyPanel
              traces={result.audience_traces}
              traceRequest={traceRequest}
            />

            <DiagnosticsPanel
              result={result}
              onViewTrace={handleViewTrace}
            />
          </div>
        ) : null}
      </main>

      <footer className="site-footer">
        <span>WikiPulse</span>
        <span>Built from public English Wikipedia pageviews</span>
      </footer>
    </div>
  )
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function toDisplayError(error: unknown): DisplayError {
  if (error instanceof AudienceAnalysisApiError) {
    return { code: error.code, message: error.message }
  }
  return {
    code: 'unexpected_error',
    message: 'An unexpected error prevented the analysis from completing.',
  }
}

function getAnalysisPeriod(
  result: AudienceAnalysisResponse,
): { start: string; end: string } | null {
  const article =
    result.topics[0]?.articles[0] ??
    result.unclustered_articles[0] ??
    result.rejected_articles[0]?.article ??
    result.audience_segments[0]?.supporting_articles[0]

  return article
    ? { start: article.analysis_start_date, end: article.analysis_end_date }
    : null
}
