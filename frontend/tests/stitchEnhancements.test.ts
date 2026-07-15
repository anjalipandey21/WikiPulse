import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type {
  AnalysisProgressStage,
  AudienceAnalysisResponse,
} from '../src/api/types.js'
import { AnalysisPipelineSummary } from '../src/components/AnalysisPipelineSummary.js'
import { CountUpMetric } from '../src/components/CountUpMetric.js'
import { LiveAttentionTicker } from '../src/components/LiveAttentionTicker.js'
import { MetricSummary } from '../src/components/MetricSummary.js'
import { StatusPanel } from '../src/components/StatusPanel.js'
import {
  buildTickerInsights,
  derivePipelineState,
  startCountUpAnimation,
} from '../src/stitchEnhancements.js'

const EMPTY_RESULT: AudienceAnalysisResponse = {
  topics: [],
  audience_segments: [],
  unclustered_articles: [],
  rejected_articles: [],
  commercial_skips: [],
  provider_skips: [],
  validation_drops: [],
  audience_traces: [],
  is_publishable: true,
  metrics: {
    topic_analysis: {
      fetched_article_count: 0,
      rejected_article_count: 0,
      eligible_article_count: 0,
      top_n_omitted_article_count: 0,
      selected_article_count: 0,
      summary_available_article_count: 0,
      summary_missing_article_count: 0,
      topic_cluster_count: 0,
      clustered_article_count: 0,
      unclustered_article_count: 0,
      selected_pageviews: 0,
    },
    audience_funnel: {
      topic_cluster_count: 0,
      commercial_eligible_cluster_count: 0,
      commercial_skipped_cluster_count: 0,
      prepared_cluster_count: 0,
      final_segment_count: 0,
      provider_skipped_cluster_count: 0,
      validation_dropped_source_cluster_count: 0,
      unmatched_provider_output_count: 0,
      commercial_eligible_pageviews: 0,
      represented_audience_pageviews: 0,
      commercial_skip_counts_by_reason: {},
    },
    workflow: {
      initial_decision_count: 0,
      initial_valid_decision_count: 0,
      initial_invalid_report_count: 0,
      revision_count: 0,
      revision_requested_cluster_count: 0,
      revision_decision_count: 0,
      revision_valid_decision_count: 0,
      final_valid_decision_count: 0,
      final_segment_count: 0,
      final_provider_skip_count: 0,
      dropped_source_cluster_count: 0,
      dropped_unmatched_decision_count: 0,
      provider_call_count: 0,
      provider_input_tokens: 0,
      provider_output_tokens: 0,
      provider_total_tokens: 0,
      provider_elapsed_seconds: 0,
      validation_issue_count: 0,
      validation_issue_counts_by_code: {},
      drop_counts_by_code: {},
    },
  },
}

const COMPLETED_RESULT: AudienceAnalysisResponse = {
  ...EMPTY_RESULT,
  topics: [{
    id: 'cluster-forest',
    name: 'Forest Technology',
    description: 'Sustainable materials and technical innovation.',
    articles: [{
      title: 'Synthetic Evidence Article',
      normalized_title: 'Synthetic_Evidence_Article',
      url: 'https://en.wikipedia.org/wiki/Synthetic_Evidence_Article',
      weekly_views: 765_432,
      daily_views: { '2026-07-01': 765_432 },
      summary: 'A public summary.',
      analysis_start_date: '2026-07-01',
      analysis_end_date: '2026-07-07',
    }],
    keywords: ['materials', 'technology'],
    total_views: 1_234_567,
    article_count: 9,
    confidence_score: 0.87,
  }],
  audience_segments: [{
    trace_id: 'trace-premium',
    id: 'audience-premium',
    name: 'Premium Material Innovators',
    description: 'A traceable synthetic audience.',
    topic_cluster_ids: ['cluster-forest'],
    size_index: 45.6,
    buying_power: 'high',
    buying_power_reason: 'Public evidence supports premium purchasing interest.',
    brand_categories: ['Premium consumer goods'],
    supporting_articles: [],
    commercial_confidence: 0.91,
    commercial_confidence_reason: 'Strong public commercial fit.',
  }],
  metrics: {
    ...EMPTY_RESULT.metrics,
    topic_analysis: {
      ...EMPTY_RESULT.metrics.topic_analysis,
      selected_article_count: 12,
      summary_available_article_count: 10,
      topic_cluster_count: 4,
      clustered_article_count: 9,
      selected_pageviews: 1_234_567,
    },
    audience_funnel: {
      ...EMPTY_RESULT.metrics.audience_funnel,
      final_segment_count: 1,
    },
  },
}

test('welcome status structure remains free of completed-analysis enhancements', () => {
  const html = renderToStaticMarkup(createElement(StatusPanel, {
    eyebrow: 'Seven days of public attention',
    title: 'See what the world is reading and who it reveals.',
    message: 'WikiPulse turns public pageviews into evidence-backed audiences.',
    actionLabel: 'Run weekly analysis',
    onAction: () => undefined,
  }))

  assert.match(html, /See what the world is reading and who it reveals\./)
  assert.match(html, /Run weekly analysis/)
  assert.doesNotMatch(html, /live-attention-ticker/)
  assert.doesNotMatch(html, /analysis-pipeline/)
})

test('live attention ticker is absent without a completed result', () => {
  const html = renderToStaticMarkup(
    createElement(LiveAttentionTicker, { result: null }),
  )
  assert.equal(html, '')
})

test('live attention ticker uses only result-backed values and one accessible summary', () => {
  const html = renderToStaticMarkup(
    createElement(LiveAttentionTicker, { result: COMPLETED_RESULT }),
  )

  assert.match(html, /Forest Technology/)
  assert.match(html, /1\.2M pageviews/)
  assert.match(html, /87%/)
  assert.match(html, /Top commercial confidence/)
  assert.match(html, /91% · Premium Material Innovators/)
  assert.match(html, /High · Premium Material Innovators/)
  assert.match(html, /Jul 1 – Jul 7/)
  assert.match(html, /10 of 12 articles/)
  assert.match(html, /aria-hidden="true"/)
  assert.equal(countMatches(html, /class="visually-hidden"/g), 1)
  assert.equal(countMatches(html, /class="live-attention-list"/g), 2)
  assert.doesNotMatch(html, /trending|rising|live change/i)
  assert.doesNotMatch(html, /2\.4M|Global Tech Enthusiasts/)
})

test('live attention ticker safely omits unavailable optional metrics', () => {
  const html = renderToStaticMarkup(
    createElement(LiveAttentionTicker, { result: EMPTY_RESULT }),
  )

  assert.match(html, /Published audiences/)
  assert.doesNotMatch(html, /Top cluster|Topic confidence|Buying power|Summary coverage/)
})

test('live attention ticker selects the real maximum-pageview cluster', () => {
  const lowerCluster = COMPLETED_RESULT.topics[0]
  assert.ok(lowerCluster)
  const higherCluster = {
    ...lowerCluster,
    id: 'cluster-higher',
    name: 'Higher Attention Cluster',
    total_views: lowerCluster.total_views + 500_000,
    confidence_score: 0.93,
  }
  const insights = buildTickerInsights({
    ...COMPLETED_RESULT,
    topics: [lowerCluster, higherCluster],
  })

  assert.deepEqual(insights.slice(0, 3), [
    { label: 'Top cluster', value: 'Higher Attention Cluster' },
    { label: 'Top-cluster attention', value: '1.7M pageviews' },
    { label: 'Topic confidence', value: '93%' },
  ])
})

test('every real stream stage maps to its truthful public pipeline state', () => {
  const expected: Record<AnalysisProgressStage, string> = {
    waiting_for_slot: 'pending,pending,pending,pending,pending,pending',
    fetching_pageviews: 'active,pending,pending,pending,pending,pending',
    selecting_articles: 'complete,pending,pending,pending,pending,pending',
    enriching_summaries: 'complete,active,pending,pending,pending,pending',
    modeling_topics: 'complete,complete,active,pending,pending,pending',
    routing_commercial_clusters: 'complete,complete,complete,pending,pending,pending',
    preparing_audience_evidence: 'complete,complete,complete,active,pending,pending',
    generating_audience_decisions: 'complete,complete,complete,active,pending,pending',
    validating_audience_decisions: 'complete,complete,complete,complete,active,pending',
    revising_audience_decisions: 'complete,complete,complete,complete,active,pending',
    validating_revised_decisions: 'complete,complete,complete,complete,active,pending',
    finalizing_audience_results: 'complete,complete,complete,complete,active,pending',
    assembling_response: 'complete,complete,complete,complete,complete,pending',
  }

  for (const [stage, statuses] of Object.entries(expected)) {
    const state = derivePipelineState(stage as AnalysisProgressStage, null)
    assert.equal(state.steps.map((step) => step.status).join(','), statuses, stage)
  }
})

test('pipeline never publishes during loading and handles terminal audience outcomes', () => {
  const loadingHtml = renderToStaticMarkup(
    createElement(AnalysisPipelineSummary, {
      stage: 'finalizing_audience_results',
    }),
  )
  assert.match(loadingHtml, /pipeline-step-pending[^>]*data-status="pending"[^>]*><span[^>]*>06/)
  assert.doesNotMatch(loadingHtml, /pipeline-step-complete[^>]*data-status="complete"[^>]*><span[^>]*>✓<\/span><span>Published/)

  const completed = derivePipelineState(null, COMPLETED_RESULT)
  assert.equal(completed.steps.at(-1)?.status, 'complete')
  assert.match(completed.description, /1 publishable audience/)

  const empty = derivePipelineState(null, EMPTY_RESULT)
  assert.equal(empty.steps.at(-1)?.status, 'empty')
  assert.equal(empty.description, 'No publishable audience')
})

test('unknown and failed stages fail safely without claiming progress', () => {
  const unknown = derivePipelineState(
    'future_stage' as AnalysisProgressStage,
    null,
  )
  assert.ok(unknown.steps.every((step) => step.status === 'pending'))

  const failed = derivePipelineState('modeling_topics', null, true)
  assert.equal(failed.steps[2]?.status, 'failed')
  assert.equal(failed.steps[3]?.status, 'pending')
  assert.match(failed.description, /Analysis stopped/)
})

test('loading status panel keeps its copy and includes the compact pipeline', () => {
  const html = renderToStaticMarkup(
    createElement(
      StatusPanel,
      {
        eyebrow: 'Analysis in progress',
        title: 'Reading the shape of the week',
        message: 'Existing loading copy.',
        busy: true,
      },
      createElement(AnalysisPipelineSummary, { stage: 'enriching_summaries' }),
    ),
  )

  assert.match(html, /Reading the shape of the week/)
  assert.match(html, /Existing loading copy\./)
  assert.match(html, /analysis-pipeline/)
  assert.match(html, /aria-current="step"/)
})

test('completed summary composes ticker, pipeline, and unchanged KPI values', () => {
  const html = renderToStaticMarkup(createElement(
    'div',
    null,
    createElement(LiveAttentionTicker, { result: COMPLETED_RESULT }),
    createElement(MetricSummary, { metrics: COMPLETED_RESULT.metrics }),
    createElement(AnalysisPipelineSummary, {
      stage: null,
      result: COMPLETED_RESULT,
    }),
  ))

  assert.match(html, /live-attention-ticker/)
  assert.match(html, /Selected pageviews/)
  assert.match(html, /1\.2M/)
  assert.match(html, /Topic clusters/)
  assert.match(html, />4</)
  assert.match(html, /1 publishable audience/)
})

test('count-up exposes the final value immediately to assistive technology', () => {
  const html = renderToStaticMarkup(createElement(CountUpMetric, {
    value: 1234,
    format: (value: number) => `${value} units`,
  }))

  assert.match(html, /aria-hidden="true">1234 units/)
  assert.match(html, /class="visually-hidden">1234 units/)
})

test('count-up animation reaches its integer target without timing sleeps', () => {
  const frames = createFrameHarness()
  const values: number[] = []
  startCountUpAnimation({
    from: 0,
    to: 100,
    duration: 600,
    integer: true,
    reducedMotion: false,
    onValue: (value) => values.push(value),
    requestFrame: frames.request,
    cancelFrame: frames.cancel,
  })

  frames.runNext(0)
  frames.runNext(300)
  frames.runNext(600)
  assert.deepEqual(values, [0, 88, 100])
  assert.equal(frames.pendingCount, 0)
})

test('count-up reduced motion and same-value updates resolve immediately', () => {
  for (const scenario of [
    { from: 0, to: 45, reducedMotion: true },
    { from: 45, to: 45, reducedMotion: false },
  ]) {
    const frames = createFrameHarness()
    const values: number[] = []
    startCountUpAnimation({
      ...scenario,
      duration: 600,
      integer: true,
      onValue: (value) => values.push(value),
      requestFrame: frames.request,
      cancelFrame: frames.cancel,
    })
    assert.deepEqual(values, [45])
    assert.equal(frames.pendingCount, 0)
  }
})

test('count-up value changes animate from the previous settled value', () => {
  const frames = createFrameHarness()
  const values: number[] = []
  startCountUpAnimation({
    from: 100,
    to: 200,
    duration: 600,
    integer: true,
    reducedMotion: false,
    onValue: (value) => values.push(value),
    requestFrame: frames.request,
    cancelFrame: frames.cancel,
  })

  frames.runNext(0)
  frames.runNext(600)
  assert.deepEqual(values, [100, 200])
})

test('count-up cleanup cancels the active animation frame', () => {
  const frames = createFrameHarness()
  const values: number[] = []
  const stop = startCountUpAnimation({
    from: 10,
    to: 20,
    duration: 600,
    integer: true,
    reducedMotion: false,
    onValue: (value) => values.push(value),
    requestFrame: frames.request,
    cancelFrame: frames.cancel,
  })

  assert.equal(frames.pendingCount, 1)
  stop()
  assert.equal(frames.pendingCount, 0)
  assert.equal(frames.cancelCount, 1)
  assert.deepEqual(values, [])
})

function countMatches(value: string, pattern: RegExp): number {
  return [...value.matchAll(pattern)].length
}

function createFrameHarness() {
  let nextHandle = 1
  let cancelCount = 0
  const callbacks = new Map<number, FrameRequestCallback>()

  return {
    request(callback: FrameRequestCallback): number {
      const handle = nextHandle
      nextHandle += 1
      callbacks.set(handle, callback)
      return handle
    },
    cancel(handle: number): void {
      cancelCount += 1
      callbacks.delete(handle)
    },
    runNext(time: number): void {
      const entry = callbacks.entries().next().value as
        | [number, FrameRequestCallback]
        | undefined
      assert.ok(entry, 'expected one scheduled animation frame')
      callbacks.delete(entry[0])
      entry[1](time)
    },
    get pendingCount(): number {
      return callbacks.size
    },
    get cancelCount(): number {
      return cancelCount
    },
  }
}
