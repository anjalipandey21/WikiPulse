import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { AskWikiPulseDrawer } from '../src/components/AskWikiPulseDrawer.js'
import { AskWikiPulsePanel } from '../src/components/AskWikiPulsePanel.js'

const RUN_ID = '70bbfdcb-a4d9-4583-aa40-827f0813dd21'

test('eligible drawer renders a dormant accessible trigger without requesting an answer', () => {
  let requestCount = 0
  const originalFetch = globalThis.fetch
  globalThis.fetch = (async () => {
    requestCount += 1
    throw new Error('Unexpected request')
  }) as typeof fetch

  try {
    const html = renderToStaticMarkup(createElement(AskWikiPulseDrawer, {
      runId: RUN_ID,
      audienceCount: 2,
    }))

    assert.match(html, />Ask WikiPulse</)
    assert.match(html, /aria-haspopup="dialog"/)
    assert.match(html, /aria-expanded="false"/)
    assert.doesNotMatch(html, /role="dialog"/)
    assert.equal(requestCount, 0)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('drawer owns dialog focus, close, scroll-lock, and run-change cleanup', () => {
  const source = readFileSync('src/components/AskWikiPulseDrawer.tsx', 'utf8')

  assert.match(source, /role="dialog"/)
  assert.match(source, /aria-modal="true"/)
  assert.match(source, /event\.key === 'Escape'/)
  assert.match(source, /event\.key !== 'Tab'/)
  assert.match(source, /getFocusableElements\(drawer\)/)
  assert.match(source, /closeButtonRef\.current\?\.focus\(\)/)
  assert.match(source, /trigger\?\.focus\(\)/)
  assert.match(source, /document\.body\.style\.overflow = 'hidden'/)
  assert.match(source, /document\.body\.style\.overflow = previousOverflow/)
  assert.match(source, /className="ask-drawer-backdrop"[\s\S]*onClick=\{closeDrawer\}/)
  assert.match(source, /\}, \[closeDrawer, runId\]\)/)
})

test('drawer delegates question state and API behavior to the existing panel', () => {
  const drawerSource = readFileSync('src/components/AskWikiPulseDrawer.tsx', 'utf8')
  const panelSource = readFileSync('src/components/AskWikiPulsePanel.tsx', 'utf8')

  assert.match(drawerSource, /<AskWikiPulsePanel/)
  assert.match(drawerSource, /embedded/)
  assert.doesNotMatch(drawerSource, /askWikiPulse\(|\/questions|fetch\(/)
  assert.match(panelSource, /if \(!questionIsValid \|\| loading\) return/)
  assert.match(panelSource, /activeController\.current\?\.abort\(\)/)
})

test('embedded panel retains suggestions, grounded status, and private-safe markup', () => {
  const html = renderToStaticMarkup(createElement(AskWikiPulsePanel, {
    runId: RUN_ID,
    audienceCount: 2,
    embedded: true,
  }))

  assert.match(html, /Ask WikiPulse question and evidence/)
  assert.match(html, /How do the published audiences differ/)
  assert.match(html, /aria-live="polite"/)
  assert.match(html, /No chat history is saved/)
  assert.doesNotMatch(html, /private.note|checkpoint|reasoning|api.key/i)
})

test('workspace retains the existing evidence eligibility rule and App does not own the drawer', () => {
  const workspaceSource = readFileSync('src/components/AudienceReviewWorkspace.tsx', 'utf8')
  const appSource = readFileSync('src/App.tsx', 'utf8')

  assert.match(workspaceSource, /run\?\.is_complete && run\.published_audiences\.some/)
  assert.match(workspaceSource, /supporting_articles\.length > 0/)
  assert.match(workspaceSource, /<AskWikiPulseDrawer/)
  assert.doesNotMatch(appSource, /AskWikiPulseDrawer/)
})

test('drawer styling includes fixed trigger, mobile sheet, focus, and reduced motion behavior', () => {
  const css = readFileSync('src/index.css', 'utf8')

  assert.match(css, /\.ask-drawer-trigger\s*\{[\s\S]*position: fixed/)
  assert.match(css, /\.ask-drawer\s*\{[\s\S]*width: min\(28rem, 100vw\)/)
  assert.match(css, /@media \(max-width: 540px\)[\s\S]*\.ask-drawer\s*\{[\s\S]*width: 100vw/)
  assert.match(css, /:focus-visible\s*\{[\s\S]*outline:/)
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)[\s\S]*\.ask-drawer[\s\S]*animation: none/)
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)[\s\S]*\.ask-drawer-trigger-icon[\s\S]*animation: none/)
})
