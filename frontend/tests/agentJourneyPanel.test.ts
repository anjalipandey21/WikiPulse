import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { AgentJourneyPanel } from '../src/components/AgentJourneyPanel.js'
import type {
  AudienceDecisionTraceResponse,
  AudienceTraceEventCode,
  AudienceTraceEventResponse,
  AudienceTracePhase,
} from '../src/api/types.js'


function makeEvents(
  codes: readonly AudienceTraceEventCode[],
): AudienceTraceEventResponse[] {
  return codes.map((code, index) => ({
    sequence: index + 1,
    phase: phaseFor(code),
    code,
    outcome_code: code === 'decision_dropped' ? 'unresolved_decision' : null,
    issues:
      code === 'validation_failed'
        ? [{ code: 'invalid_article_reference', reference_id: 'cluster-1:a9' }]
        : [],
  }))
}


function phaseFor(code: AudienceTraceEventCode): AudienceTracePhase {
  if (
    code === 'revision_requested' ||
    code === 'revision_failed'
  ) {
    return 'revision'
  }
  if (
    code === 'audience_published' ||
    code === 'provider_skipped' ||
    code === 'decision_dropped'
  ) {
    return 'final'
  }
  return 'initial'
}


test('summarizes ordinary, revised, and dropped journeys truthfully', () => {
  const traces: AudienceDecisionTraceResponse[] = [
    {
      trace_id: 'trace-published',
      cluster_id: 'cluster-published',
      cluster_name: 'Published audience',
      source_known: true,
      final_outcome: 'published',
      events: makeEvents([
        'generation_requested',
        'decision_received',
        'validation_passed',
        'audience_published',
      ]),
    },
    {
      trace_id: 'trace-revised',
      cluster_id: 'cluster-revised',
      cluster_name: 'Revised audience',
      source_known: true,
      final_outcome: 'published',
      events: makeEvents([
        'generation_requested',
        'decision_received',
        'validation_failed',
        'revision_requested',
        'decision_received',
        'validation_passed',
        'audience_published',
      ]),
    },
    {
      trace_id: 'trace-dropped',
      cluster_id: 'cluster-dropped',
      cluster_name: 'Dropped audience',
      source_known: true,
      final_outcome: 'validation_dropped',
      events: makeEvents([
        'generation_requested',
        'validation_failed',
        'revision_requested',
        'revision_failed',
        'decision_dropped',
      ]),
    },
    {
      trace_id: 'trace-missing-then-revised',
      cluster_id: 'cluster-missing-then-revised',
      cluster_name: 'Recovered missing decision',
      source_known: true,
      final_outcome: 'published',
      events: makeEvents([
        'generation_requested',
        'validation_failed',
        'revision_requested',
        'decision_received',
        'validation_passed',
        'audience_published',
      ]),
    },
    {
      trace_id: 'trace-unknown',
      cluster_id: 'cluster-unknown',
      cluster_name: null,
      source_known: false,
      final_outcome: 'validation_dropped',
      events: makeEvents([
        'decision_received',
        'validation_failed',
        'decision_dropped',
      ]),
    },
  ]

  const html = renderToStaticMarkup(
    createElement(AgentJourneyPanel, { traces, traceRequest: null }),
  )
  const visibleText = html.replace(/<[^>]+>/g, ' ')

  assert.match(visibleText, /Generated → Validated → Published/)
  assert.match(
    visibleText,
    /Generated → Validation failed → Revised → Validated → Published/,
  )
  assert.match(
    visibleText,
    /Generation requested → Validation failed → Revision requested → Revision failed → Dropped/,
  )
  assert.match(
    visibleText,
    /Generation requested → Validation failed → Revised → Validated → Published/,
  )
  assert.match(visibleText, /Received → Validation failed → Dropped/)
})


test('renders collapsed nested disclosures without visible internal codes or IDs', () => {
  const trace: AudienceDecisionTraceResponse = {
    trace_id: 'trace-internal-1',
    cluster_id: 'candidate-internal-1',
    cluster_name: 'Space technology enthusiasts',
    source_known: true,
    final_outcome: 'validation_dropped',
    events: makeEvents([
      'generation_requested',
      'decision_received',
      'validation_failed',
      'decision_dropped',
    ]),
  }

  const html = renderToStaticMarkup(
    createElement(AgentJourneyPanel, { traces: [trace], traceRequest: null }),
  )
  const visibleText = html.replace(/<[^>]+>/g, ' ')

  assert.match(html, /<details class="agent-journey">/)
  assert.match(
    html,
    /<details class="journey-card" id="trace-internal-1">/,
  )
  assert.doesNotMatch(html, /<details[^>]*\sopen(?:="")?[^>]*>/)
  assert.match(visibleText, /Generated → Validation failed → Dropped/)
  assert.match(
    visibleText,
    /This cluster was included in the initial structured-generation batch\./,
  )
  assert.doesNotMatch(visibleText, /trace-internal-1/)
  assert.doesNotMatch(visibleText, /candidate-internal-1/)
  assert.doesNotMatch(visibleText, /generation_requested/)
  assert.doesNotMatch(visibleText, /invalid_article_reference/)
  assert.doesNotMatch(visibleText, /cluster-1:a9/)
  assert.doesNotMatch(visibleText, /unresolved_decision/)
})
