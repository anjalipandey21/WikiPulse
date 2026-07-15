import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  AudienceReviewApiError,
  getAudienceReview,
  startAudienceReview,
  submitAudienceReviewCommand,
} from '../src/api/audienceReview.js'
import type {
  AudienceReviewRun,
  PendingReview,
  ReviewCommandRequest,
} from '../src/api/audienceReviewTypes.js'
import { AnalysisModeSelector } from '../src/components/AnalysisModeSelector.js'
import { EditingReviewPanel } from '../src/components/EditingReviewPanel.js'
import { PendingReviewCard } from '../src/components/PendingReviewCard.js'
import { ReviewOutcomeSummary } from '../src/components/ReviewOutcomeSummary.js'
import {
  deriveReviewUiStatus,
  EDIT_DROP_TEXT,
  normalizeAnalystFeedback,
} from '../src/reviewUi.js'

const RUN_ID = '70bbfdcb-a4d9-4583-aa40-827f0813dd21'
const COMMAND_ID = '1adbdc3f-dd67-4358-89cf-1763901c6ee0'

function recommendation() {
  return {
    audience_id: 'audience-1',
    name: 'Space technology planners',
    description: 'Readers following launch systems and satellite infrastructure.',
    topic_cluster_ids: ['cluster-1'],
    size_index: 73,
    buying_power: 'high' as const,
    buying_power_reason: 'Professional equipment and travel needs signal buying power.',
    brand_categories: ['Technology', 'Travel'],
    supporting_article_reference_ids: ['cluster-1:a1'],
    supporting_articles: [article()],
    commercial_confidence: 0.84,
    commercial_confidence_reason: 'The evidence is coherent and commercially relevant.',
  }
}

function article() {
  return {
    title: 'Reusable launch system',
    normalized_title: 'Reusable_launch_system',
    url: 'https://en.wikipedia.org/wiki/Reusable_launch_system',
    weekly_views: 125000,
    daily_views: [{ day: '2026-07-01', pageviews: 17000 }],
    summary: 'A launch system designed for repeated use.',
    analysis_start_date: '2026-07-01',
    analysis_end_date: '2026-07-07',
  }
}

function pendingReview(): PendingReview {
  return {
    status: 'pending_review',
    review_id: 'review-1',
    cluster_id: 'cluster-1',
    expected_version: 1,
    position: 2,
    total_reviews: 5,
    cluster_name: 'Reusable space systems',
    cluster_pageviews: 310000,
    article_count: 3,
    size_index: 73,
    topic_confidence: 0.91,
    original_recommendation: recommendation(),
    evidence: [{ reference_id: 'cluster-1:a1', article: article() }],
    edit_available: true,
  }
}

function metrics() {
  return {
    initial_decision_count: 1,
    initial_valid_decision_count: 1,
    initial_invalid_report_count: 0,
    revision_count: 0,
    revision_requested_cluster_count: 0,
    revision_decision_count: 0,
    revision_valid_decision_count: 0,
    final_valid_decision_count: 1,
    final_segment_count: 1,
    final_provider_skip_count: 0,
    dropped_source_cluster_count: 0,
    dropped_unmatched_decision_count: 0,
    provider_call_count: 1,
    provider_input_tokens: 100,
    provider_output_tokens: 80,
    provider_total_tokens: 180,
    provider_elapsed_seconds: 0.4,
    validation_issue_count: 0,
    validation_issue_counts_by_code: [],
    drop_counts_by_code: [],
  }
}

function makeRun(overrides: Partial<AudienceReviewRun> = {}): AudienceReviewRun {
  return {
    run_id: RUN_ID,
    status: 'pending_review',
    is_complete: false,
    created_at: '2026-07-15T10:00:00Z',
    expires_at: '2026-07-15T11:00:00Z',
    progress: { total_reviews: 5, completed_reviews: 1, queued_reviews: 3, current_position: 2 },
    current_review: pendingReview(),
    published_audiences: [],
    rejected_reviews: [],
    edit_validation_drops: [],
    expired_reviews: [],
    provider_skips: [],
    validation_drops: [],
    journey: [],
    automatic_workflow_metrics: metrics(),
    failure_code: null,
    ...overrides,
  }
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

test('Standard Analysis remains the default visible mode', () => {
  const source = readFileSync('src/App.tsx', 'utf8')
  assert.match(source, /useState<AnalysisMode>\('standard'\)/)
  assert.match(source, /runAudienceAnalysisStream/)
  assert.match(source, /mode === 'review' \? <AudienceReviewWorkspace \/>/)
})

test('mode selector exposes two keyboard-operable radio controls', () => {
  const html = renderToStaticMarkup(
    createElement(AnalysisModeSelector, { mode: 'review', onChange: () => undefined }),
  )
  assert.equal((html.match(/type="radio"/g) ?? []).length, 2)
  assert.match(html, /checked="" value="review"/)
  assert.match(html, /Analyst Review/)
})

test('review API start, GET, and command use only public endpoint contracts', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })
  const requests: Array<{ url: string; init?: RequestInit }> = []
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    requests.push({ url: String(url), init })
    if (String(url).endsWith('/commands')) {
      return jsonResponse({
        receipt: {
          command_id: COMMAND_ID,
          type: 'approve',
          review_id: 'review-1',
          cluster_id: 'cluster-1',
          accepted: true,
          idempotent_replay: false,
          resulting_status: 'published',
          run_status: 'completed',
        },
        run: makeRun({ status: 'completed', is_complete: true, current_review: null }),
      })
    }
    return jsonResponse(makeRun())
  }) as typeof fetch

  await startAudienceReview(RUN_ID)
  await getAudienceReview(RUN_ID)
  await submitAudienceReviewCommand(RUN_ID, {
    type: 'approve', command_id: COMMAND_ID, review_id: 'review-1', cluster_id: 'cluster-1', expected_version: 1,
  })

  assert.deepEqual(requests.map((item) => [item.url, item.init?.method]), [
    ['/api/audience-reviews', 'POST'],
    [`/api/audience-reviews/${RUN_ID}`, 'GET'],
    [`/api/audience-reviews/${RUN_ID}/commands`, 'POST'],
  ])
  assert.deepEqual(JSON.parse(String(requests[0]?.init?.body)), { run_id: RUN_ID })
})

test('ambiguous command retry can reuse the exact command identifier and body', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })
  const bodies: string[] = []
  let calls = 0
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    bodies.push(String(init?.body))
    calls += 1
    if (calls === 1) throw new TypeError('lost response')
    return jsonResponse({
      receipt: { command_id: COMMAND_ID, type: 'edit_recommendation', review_id: 'review-1', cluster_id: 'cluster-1', accepted: true, idempotent_replay: true, resulting_status: 'published', run_status: 'completed' },
      run: makeRun({ status: 'completed', is_complete: true, current_review: null }),
    })
  }) as typeof fetch
  const command: ReviewCommandRequest = {
    type: 'edit_recommendation', command_id: COMMAND_ID, review_id: 'review-1', cluster_id: 'cluster-1', expected_version: 1,
    feedback: 'PRIVATE-FEEDBACK-SENTINEL improve positioning', fields_to_change: ['audience_positioning'],
  }

  await assert.rejects(() => submitAudienceReviewCommand(RUN_ID, command), AudienceReviewApiError)
  await submitAudienceReviewCommand(RUN_ID, command)
  assert.equal(bodies[0], bodies[1])
  assert.equal(JSON.parse(bodies[1] ?? '{}').command_id, COMMAND_ID)
})

test('pending card renders review evidence, metrics, actions, and no private values', () => {
  const html = renderToStaticMarkup(createElement(PendingReviewCard, {
    review: pendingReview(), busyAction: null,
    onApprove: () => undefined, onReject: () => undefined, onEdit: () => undefined,
  }))
  assert.match(html, /Audience 2 of 5/)
  assert.match(html, /Space technology planners/)
  assert.match(html, /310K pageviews/)
  assert.match(html, /91%/)
  assert.match(html, /Request edit/)
  assert.match(html, /Supporting evidence/)
  assert.doesNotMatch(html, /PRIVATE-|feedback.*sentinel/i)
})

test('editing recovery state is truthful and exposes manual refresh', () => {
  const current = pendingReview()
  const html = renderToStaticMarkup(createElement(EditingReviewPanel, {
    review: { ...current, status: 'editing', edit_available: false, expected_version: undefined } as never,
    onRefresh: () => undefined,
  }))
  assert.match(html, /Applying analyst edit/)
  assert.match(html, /Check status/)
  assert.match(html, /aria-busy="true"/)
})

test('completed outcomes distinguish edited publication, drop, rejection, and expiry', () => {
  const run = makeRun({
    status: 'completed', is_complete: true, current_review: null,
    published_audiences: [{ review_id: 'r1', cluster_id: 'c1', trace_id: 't1', publication_source: 'analyst_edit', audience: recommendation() }],
    rejected_reviews: [{ review_id: 'r2', cluster_id: 'c2', cluster_name: 'Rejected topic', reason_code: 'safety_concern' }],
    edit_validation_drops: [{ review_id: 'r3', cluster_id: 'c3', cluster_name: 'Dropped topic', drop_code: 'edit_unsupported_references' }],
    expired_reviews: [{ review_id: 'r4', cluster_id: 'c4', cluster_name: 'Expired topic' }],
    journey: [{ trace_id: 't1', cluster_id: 'c1', cluster_name: 'Edited topic', source_known: true, final_outcome: 'published', review_id: 'r1', events: [
      { sequence: 1, phase: 'review', code: 'review_requested', outcome_code: null, issues: [] },
      { sequence: 2, phase: 'edit', code: 'analyst_edit_requested', outcome_code: null, issues: [] },
      { sequence: 3, phase: 'edit', code: 'edited_audience_published', outcome_code: 'published', issues: [] },
    ] }],
  })
  const html = renderToStaticMarkup(createElement(ReviewOutcomeSummary, { run }))
  assert.match(html, /Analyst-edited audience/)
  assert.match(html, /Analyst rejected/)
  assert.match(html, new RegExp(EDIT_DROP_TEXT.edit_unsupported_references))
  assert.match(html, /Review expired/)
  assert.match(html, /Edit requested/)
  assert.doesNotMatch(html, /PRIVATE-|PRIVATE-NOTE-SENTINEL/)
})

test('feedback normalization and review UI states enforce bounded semantics', () => {
  assert.equal(normalizeAnalystFeedback('  Café\n\ttravel   focus  '), 'Café travel focus')
  assert.equal(Array.from(normalizeAnalystFeedback('1234567890')).length, 10)
  assert.equal(deriveReviewUiStatus(null, 'idle'), 'idle')
  assert.equal(deriveReviewUiStatus(makeRun(), 'idle'), 'pending_review')
  assert.equal(deriveReviewUiStatus(makeRun({ status: 'editing' }), 'idle'), 'editing')
  assert.equal(deriveReviewUiStatus(makeRun({ status: 'expired', is_complete: true }), 'idle'), 'expired')
  assert.equal(deriveReviewUiStatus(makeRun(), 'conflict'), 'conflict')
})

test('safe API errors expose codes without reflecting request-private values', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })
  globalThis.fetch = (async () => jsonResponse({ error: { code: 'invalid_review_command', message: 'The review command is invalid.' } }, 422)) as typeof fetch
  const privateValue = 'PRIVATE-NOTE-SENTINEL'
  const command: ReviewCommandRequest = {
    type: 'reject', command_id: COMMAND_ID, review_id: 'review-1', cluster_id: 'cluster-1', expected_version: 1,
    reason_code: 'other', private_note: privateValue,
  }
  await assert.rejects(
    () => submitAudienceReviewCommand(RUN_ID, command),
    (error: unknown) => {
      assert.ok(error instanceof AudienceReviewApiError)
      assert.equal(error.code, 'invalid_review_command')
      assert.doesNotMatch(String(error), new RegExp(privateValue))
      return true
    },
  )
})
