import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  askWikiPulse,
  AudienceReviewApiError,
} from '../src/api/audienceReview.js'
import { AskWikiPulsePanel } from '../src/components/AskWikiPulsePanel.js'
import { normalizeAssistantQuestion } from '../src/reviewUi.js'

const RUN_ID = '70bbfdcb-a4d9-4583-aa40-827f0813dd21'

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function groundedResponse() {
  return {
    answer: 'The launch-systems audience is supported by the cited public article.',
    evidence_status: 'grounded' as const,
    citations: [{
      article_title: 'Reusable launch system',
      article_url: 'https://en.wikipedia.org/wiki/Reusable_launch_system',
      audience_label: 'Space technology planners',
      relevance: 'Supporting evidence for Space technology planners.',
    }],
    suggested_follow_up_questions: ['What evidence supports this audience?'],
  }
}

test('assistant client submits one public question and parses grounded citations', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })
  const requests: Array<{ url: string; init?: RequestInit }> = []
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    requests.push({ url: String(url), init })
    return jsonResponse(groundedResponse())
  }) as typeof fetch

  const controller = new AbortController()
  const result = await askWikiPulse(RUN_ID, 'What evidence supports this audience?', controller.signal)

  assert.equal(result.evidence_status, 'grounded')
  assert.equal(result.citations[0]?.article_title, 'Reusable launch system')
  assert.equal(requests.length, 1)
  assert.equal(requests[0]?.url, `/api/audience-reviews/${RUN_ID}/questions`)
  assert.equal(requests[0]?.init?.method, 'POST')
  assert.deepEqual(JSON.parse(String(requests[0]?.init?.body)), {
    question: 'What evidence supports this audience?',
  })
  assert.equal(requests[0]?.init?.signal, controller.signal)
})

test('assistant client exposes only stable safe API errors', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })
  globalThis.fetch = (async () => jsonResponse({
    error: { code: 'assistant_provider_failed', message: 'Ask WikiPulse could not answer safely.' },
  }, 502)) as typeof fetch

  await assert.rejects(
    () => askWikiPulse(RUN_ID, 'What evidence supports this audience?'),
    (error: unknown) => error instanceof AudienceReviewApiError && (
      error.code === 'assistant_provider_failed' &&
      !String(error).includes('PRIVATE-')
    ),
  )
})

test('assistant panel is keyboard-accessible, stateless, and grounded in one run', () => {
  const html = renderToStaticMarkup(createElement(AskWikiPulsePanel, {
    runId: RUN_ID,
    audienceCount: 2,
  }))
  assert.match(html, /Ask WikiPulse/)
  assert.match(html, /<form/)
  assert.match(html, /id="ask-wikipulse-question"/)
  assert.match(html, /aria-live="polite"/)
  assert.match(html, /No chat history is saved/)
  assert.match(html, /How do the published audiences differ/)
  assert.doesNotMatch(html, /feedback|private.note|checkpoint|prompt|reasoning/i)
})

test('question normalization matches bounded backend whitespace behavior', () => {
  assert.equal(
    normalizeAssistantQuestion('  What   evidence\n supports this?  '),
    'What evidence supports this?',
  )
})

test('workspace displays assistant only for completed runs with published evidence', () => {
  const source = readFileSync('src/components/AudienceReviewWorkspace.tsx', 'utf8')
  assert.match(source, /run\?\.is_complete && run\.published_audiences\.some/)
  assert.match(source, /supporting_articles\.length > 0/)
  assert.match(source, /<AskWikiPulseDrawer/)
})

test('assistant aborts on run change or unmount and never uses browser storage', () => {
  const source = readFileSync('src/components/AskWikiPulsePanel.tsx', 'utf8')
  assert.match(source, /activeController\.current\?\.abort\(\)/)
  assert.match(source, /\}, \[runId\]\)/)
  assert.doesNotMatch(source, /localStorage|sessionStorage/)
  assert.match(source, /if \(!questionIsValid \|\| loading\) return/)
  assert.match(source, /disabled=\{!questionIsValid \|\| loading\}/)
})

test('standard analysis remains on its original streaming endpoint', () => {
  const appSource = readFileSync('src/App.tsx', 'utf8')
  const analysisSource = readFileSync('src/api/audienceAnalysis.ts', 'utf8')
  assert.match(appSource, /useState<AnalysisMode>\('standard'\)/)
  assert.match(appSource, /runAudienceAnalysisStream/)
  assert.match(analysisSource, /\/api\/audience-analysis\/stream/)
  assert.doesNotMatch(analysisSource, /questions/)
})
