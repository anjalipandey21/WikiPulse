import assert from 'node:assert/strict'
import test from 'node:test'

import {
  AudienceAnalysisApiError,
  parseAudienceAnalysisStream,
} from '../src/api/audienceAnalysis.js'
import type {
  AnalysisProgressStage,
  AudienceAnalysisResponse,
} from '../src/api/types.js'


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


function makeStreamResponse(lines: unknown[], splitAt?: number): Response {
  const serialized = `${lines.map((line) => JSON.stringify(line)).join('\n')}\n`
  const chunks =
    splitAt === undefined
      ? [serialized]
      : [serialized.slice(0, splitAt), serialized.slice(splitAt)]
  const encoder = new TextEncoder()
  let index = 0
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      const chunk = chunks[index]
      index += 1
      if (chunk === undefined) {
        controller.close()
      } else {
        controller.enqueue(encoder.encode(chunk))
      }
    },
  })
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'application/x-ndjson' },
  })
}


function makeTrackedReaderResponse(chunks: readonly (string | Error)[]) {
  const encoder = new TextEncoder()
  let index = 0
  let cancelCount = 0
  let releaseCount = 0
  const reader = {
    async read(): Promise<ReadableStreamReadResult<Uint8Array>> {
      const chunk = chunks[index]
      index += 1
      if (chunk === undefined) return { done: true, value: undefined }
      if (typeof chunk !== 'string') throw chunk
      return { done: false, value: encoder.encode(chunk) }
    },
    async cancel(): Promise<void> {
      cancelCount += 1
    },
    releaseLock(): void {
      releaseCount += 1
    },
  } as ReadableStreamDefaultReader<Uint8Array>
  const response = {
    body: { getReader: () => reader },
    headers: new Headers({ 'Content-Type': 'application/x-ndjson' }),
    status: 200,
  } as unknown as Response

  return {
    response,
    get cancelCount() {
      return cancelCount
    },
    get releaseCount() {
      return releaseCount
    },
  }
}


async function assertApiError(
  action: () => Promise<unknown>,
  expectedCode: string,
  expectedStatus: number,
): Promise<void> {
  await assert.rejects(action, (error: unknown) => {
    assert.ok(error instanceof AudienceAnalysisApiError)
    assert.equal(error.code, expectedCode)
    assert.equal(error.status, expectedStatus)
    return true
  })
}


test('parses split progress lines and the exact terminal response', async () => {
  const stages: AnalysisProgressStage[] = []
  const serialized = `${[
    { type: 'progress', sequence: 1, stage: 'waiting_for_slot' },
    { type: 'progress', sequence: 2, stage: 'fetching_pageviews' },
    { type: 'result', sequence: 3, result: EMPTY_RESULT },
  ].map((line) => JSON.stringify(line)).join('\n')}\n`
  const fixture = makeTrackedReaderResponse([
    serialized.slice(0, 31),
    serialized.slice(31),
  ])

  const result = await parseAudienceAnalysisStream(fixture.response, (event) => {
    stages.push(event.stage)
  })

  assert.deepEqual(stages, ['waiting_for_slot', 'fetching_pageviews'])
  assert.deepEqual(result, EMPTY_RESULT)
  assert.equal(fixture.cancelCount, 0)
  assert.equal(fixture.releaseCount, 1)
})


test('returns safe terminal provider errors', async () => {
  const serialized = `${[
    { type: 'progress', sequence: 1, stage: 'waiting_for_slot' },
    {
      type: 'error',
      sequence: 2,
      status_code: 502,
      error: {
        code: 'audience_provider_unavailable',
        message: 'Audience generation is temporarily unavailable.',
      },
    },
  ].map((line) => JSON.stringify(line)).join('\n')}\n`
  const fixture = makeTrackedReaderResponse([serialized])

  await assertApiError(
    () => parseAudienceAnalysisStream(fixture.response, () => undefined),
    'audience_provider_unavailable',
    502,
  )
  assert.equal(fixture.cancelCount, 0)
  assert.equal(fixture.releaseCount, 1)
})


test('cancels the reader for decreasing sequences', async () => {
  const fixture = makeTrackedReaderResponse([`${[
    { type: 'progress', sequence: 2, stage: 'waiting_for_slot' },
    { type: 'progress', sequence: 1, stage: 'fetching_pageviews' },
  ].map((line) => JSON.stringify(line)).join('\n')}\n`])

  await assertApiError(
    () => parseAudienceAnalysisStream(fixture.response, () => undefined),
    'invalid_stream',
    200,
  )
  assert.equal(fixture.cancelCount, 1)
  assert.equal(fixture.releaseCount, 1)
})


test('cancels the reader for events after a terminal result', async () => {
  const fixture = makeTrackedReaderResponse([`${[
    { type: 'result', sequence: 1, result: EMPTY_RESULT },
    { type: 'progress', sequence: 2, stage: 'assembling_response' },
  ].map((line) => JSON.stringify(line)).join('\n')}\n`])

  await assertApiError(
    () => parseAudienceAnalysisStream(fixture.response, () => undefined),
    'invalid_stream',
    200,
  )
  assert.equal(fixture.cancelCount, 1)
  assert.equal(fixture.releaseCount, 1)
})


test('cancels the reader for unknown stages and malformed JSON', async (t) => {
  await t.test('unknown progress stage', async () => {
    const fixture = makeTrackedReaderResponse([
      `${JSON.stringify({ type: 'progress', sequence: 1, stage: 'invented_stage' })}\n`,
    ])

    await assertApiError(
      () => parseAudienceAnalysisStream(fixture.response, () => undefined),
      'invalid_stream',
      200,
    )
    assert.equal(fixture.cancelCount, 1)
    assert.equal(fixture.releaseCount, 1)
  })

  await t.test('malformed JSON', async () => {
    const fixture = makeTrackedReaderResponse(['{"type":"progress"\n'])

    await assertApiError(
      () => parseAudienceAnalysisStream(fixture.response, () => undefined),
      'invalid_stream',
      200,
    )
    assert.equal(fixture.cancelCount, 1)
    assert.equal(fixture.releaseCount, 1)
  })
})


test('propagates AbortError from reader.read without replacing it', async () => {
  const abortError = new DOMException('The operation was aborted.', 'AbortError')
  const fixture = makeTrackedReaderResponse([abortError])

  await assert.rejects(
    () => parseAudienceAnalysisStream(fixture.response, () => undefined),
    (error: unknown) => error === abortError,
  )
  assert.equal(fixture.cancelCount, 0)
  assert.equal(fixture.releaseCount, 1)
})


test('rejects extra envelope fields and missing terminal events', async (t) => {
  await t.test('extra fields', async () => {
    const response = makeStreamResponse([
      {
        type: 'progress',
        sequence: 1,
        stage: 'waiting_for_slot',
        metadata: 'must not be accepted',
      },
    ])
    await assertApiError(
      () => parseAudienceAnalysisStream(response, () => undefined),
      'invalid_stream',
      200,
    )
  })

  await t.test('missing terminal', async () => {
    const response = makeStreamResponse([
      { type: 'progress', sequence: 1, stage: 'waiting_for_slot' },
    ])
    await assertApiError(
      () => parseAudienceAnalysisStream(response, () => undefined),
      'stream_interrupted',
      200,
    )
  })
})
