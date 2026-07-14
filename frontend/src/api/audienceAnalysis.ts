import type {
  ApiErrorResponse,
  AnalysisProgressStage,
  ArticleResponse,
  AudienceAnalysisErrorEvent,
  AudienceAnalysisProgressEvent,
  AudienceAnalysisResponse,
  AudienceAnalysisStreamEvent,
  AudienceDecisionTraceResponse,
  AudienceFunnelMetricsResponse,
  AudienceSegmentResponse,
  AudienceTraceEventCode,
  AudienceTraceEventResponse,
  AudienceWorkflowMetricsResponse,
  CommercialSkippedClusterResponse,
  DroppedAudienceDecisionResponse,
  ProviderSkippedClusterResponse,
  RejectedArticleResponse,
  TopicAnalysisMetricsResponse,
  TopicClusterResponse,
} from './types'

const ENDPOINT = '/api/audience-analysis'
const STREAM_ENDPOINT = '/api/audience-analysis/stream'
const ISO_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/
const NDJSON_MEDIA_TYPE = 'application/x-ndjson'
const MAX_STREAM_LINE_CHARACTERS = 5_000_000

const ANALYSIS_PROGRESS_STAGES = new Set<AnalysisProgressStage>([
  'waiting_for_slot',
  'fetching_pageviews',
  'selecting_articles',
  'enriching_summaries',
  'modeling_topics',
  'routing_commercial_clusters',
  'preparing_audience_evidence',
  'generating_audience_decisions',
  'validating_audience_decisions',
  'revising_audience_decisions',
  'validating_revised_decisions',
  'finalizing_audience_results',
  'assembling_response',
])

type JsonObject = Record<string, unknown>

export class AudienceAnalysisApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(code: string, message: string, status: number) {
    super(message)
    this.name = 'AudienceAnalysisApiError'
    this.code = code
    this.status = status
  }
}

export async function runAudienceAnalysis(
  signal?: AbortSignal,
): Promise<AudienceAnalysisResponse> {
  let response: Response

  try {
    response = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error
    }
    throw new AudienceAnalysisApiError(
      'network_error',
      'WikiPulse could not reach the analysis service.',
      0,
    )
  }

  const payload = await readJson(response)
  if (!response.ok) {
    if (isApiErrorResponse(payload)) {
      throw new AudienceAnalysisApiError(
        payload.error.code,
        payload.error.message,
        response.status,
      )
    }
    throw new AudienceAnalysisApiError(
      'request_failed',
      'The analysis service returned an unexpected error.',
      response.status,
    )
  }

  if (!isAudienceAnalysisResponse(payload)) {
    throw new AudienceAnalysisApiError(
      'invalid_response',
      'The analysis service returned an invalid response.',
      response.status,
    )
  }
  return payload
}

export async function runAudienceAnalysisStream(
  signal: AbortSignal | undefined,
  onProgress: (event: AudienceAnalysisProgressEvent) => void,
): Promise<AudienceAnalysisResponse> {
  let response: Response

  try {
    response = await fetch(STREAM_ENDPOINT, {
      method: 'POST',
      headers: { Accept: NDJSON_MEDIA_TYPE },
      signal,
    })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new AudienceAnalysisApiError(
      'network_error',
      'WikiPulse could not reach the analysis service.',
      0,
    )
  }

  if (!response.ok) {
    const payload = await readJson(response)
    if (isApiErrorResponse(payload)) {
      throw new AudienceAnalysisApiError(
        payload.error.code,
        payload.error.message,
        response.status,
      )
    }
    throw new AudienceAnalysisApiError(
      'request_failed',
      'The analysis service returned an unexpected error.',
      response.status,
    )
  }

  return parseAudienceAnalysisStream(response, onProgress)
}

export async function parseAudienceAnalysisStream(
  response: Response,
  onProgress: (event: AudienceAnalysisProgressEvent) => void,
): Promise<AudienceAnalysisResponse> {
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.toLowerCase().includes(NDJSON_MEDIA_TYPE)) {
    throw invalidStreamError(response.status)
  }
  if (response.body === null) {
    throw invalidStreamError(response.status)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let lastSequence = 0
  let terminalResult: AudienceAnalysisResponse | null = null
  let terminalError: AudienceAnalysisErrorEvent | null = null
  let terminalSeen = false

  function processLine(line: string) {
    if (!line.trim()) throw invalidStreamError(response.status)
    if (line.length > MAX_STREAM_LINE_CHARACTERS) {
      throw invalidStreamError(response.status)
    }

    let payload: unknown
    try {
      payload = JSON.parse(line)
    } catch {
      throw invalidStreamError(response.status)
    }
    if (!isAudienceAnalysisStreamEvent(payload)) {
      throw invalidStreamError(response.status)
    }
    if (payload.sequence <= lastSequence || terminalSeen) {
      throw invalidStreamError(response.status)
    }
    lastSequence = payload.sequence

    if (payload.type === 'progress') {
      onProgress(payload)
      return
    }
    terminalSeen = true
    if (payload.type === 'result') {
      terminalResult = payload.result
    } else {
      terminalError = payload
    }
  }

  function processCompleteLines() {
    let newlineIndex = buffer.indexOf('\n')
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).replace(/\r$/, '')
      buffer = buffer.slice(newlineIndex + 1)
      processLine(line)
      newlineIndex = buffer.indexOf('\n')
    }
    if (buffer.length > MAX_STREAM_LINE_CHARACTERS) {
      throw invalidStreamError(response.status)
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      processCompleteLines()
    }
    buffer += decoder.decode()
    if (buffer) processLine(buffer.replace(/\r$/, ''))
    if (!terminalSeen) {
      throw new AudienceAnalysisApiError(
        'stream_interrupted',
        'The live analysis stream ended before a final result was received.',
        response.status,
      )
    }
  } catch (error) {
    if (isAbortError(error)) throw error
    try {
      await reader.cancel()
    } catch {
      // Preserve the original parsing or stream error if cancellation also fails.
    }
    if (error instanceof AudienceAnalysisApiError) throw error
    throw new AudienceAnalysisApiError(
      'stream_interrupted',
      'The live analysis stream ended unexpectedly.',
      response.status,
    )
  } finally {
    reader.releaseLock()
  }

  const finalError = terminalError as AudienceAnalysisErrorEvent | null
  if (finalError !== null) {
    throw new AudienceAnalysisApiError(
      finalError.error.code,
      finalError.error.message,
      finalError.status_code,
    )
  }
  if (terminalResult === null) {
    throw new AudienceAnalysisApiError(
      'stream_interrupted',
      'The live analysis stream ended before a final result was received.',
      response.status,
    )
  }
  return terminalResult
}

function invalidStreamError(status: number): AudienceAnalysisApiError {
  return new AudienceAnalysisApiError(
    'invalid_stream',
    'The analysis service returned an invalid live response.',
    status,
  )
}

async function readJson(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.toLowerCase().includes('application/json')) {
    return null
  }
  try {
    return await response.json()
  } catch {
    return null
  }
}

function isAudienceAnalysisResponse(
  value: unknown,
): value is AudienceAnalysisResponse {
  if (!isObject(value)) return false
  return (
    isArrayOf(value.topics, isTopicCluster) &&
    isArrayOf(value.audience_segments, isAudienceSegment) &&
    isArrayOf(value.unclustered_articles, isArticle) &&
    isArrayOf(value.rejected_articles, isRejectedArticle) &&
    isArrayOf(value.commercial_skips, isCommercialSkip) &&
    isArrayOf(value.provider_skips, isProviderSkip) &&
    isArrayOf(value.validation_drops, isValidationDrop) &&
    isArrayOf(value.audience_traces, isAudienceTrace) &&
    typeof value.is_publishable === 'boolean' &&
    isMetrics(value.metrics) &&
    hasValidTraceAssociations(value as unknown as AudienceAnalysisResponse)
  )
}

function isAudienceAnalysisStreamEvent(
  value: unknown,
): value is AudienceAnalysisStreamEvent {
  if (!isObject(value) || !isPositiveInteger(value.sequence)) return false
  if (value.type === 'progress') {
    return (
      hasExactKeys(value, ['type', 'sequence', 'stage']) &&
      isAnalysisProgressStage(value.stage)
    )
  }
  if (value.type === 'result') {
    return (
      hasExactKeys(value, ['type', 'sequence', 'result']) &&
      isAudienceAnalysisResponse(value.result)
    )
  }
  if (value.type === 'error') {
    return (
      hasExactKeys(value, ['type', 'sequence', 'status_code', 'error']) &&
      isIntegerInRange(value.status_code, 400, 599) &&
      isApiErrorDetail(value.error)
    )
  }
  return false
}

function isAnalysisProgressStage(
  value: unknown,
): value is AnalysisProgressStage {
  return typeof value === 'string' && ANALYSIS_PROGRESS_STAGES.has(value as AnalysisProgressStage)
}

function isArticle(value: unknown): value is ArticleResponse {
  if (!isObject(value)) return false
  return (
    isString(value.title) &&
    isString(value.normalized_title) &&
    isString(value.url) &&
    isNumber(value.weekly_views) &&
    isNumberRecord(value.daily_views, true) &&
    (value.summary === null || isString(value.summary)) &&
    isIsoDate(value.analysis_start_date) &&
    isIsoDate(value.analysis_end_date)
  )
}

function isTopicCluster(value: unknown): value is TopicClusterResponse {
  if (!isObject(value)) return false
  return (
    isString(value.id) &&
    isString(value.name) &&
    (value.description === null || isString(value.description)) &&
    isArrayOf(value.articles, isArticle) &&
    isArrayOf(value.keywords, isString) &&
    isNumber(value.total_views) &&
    isNumber(value.article_count) &&
    (value.confidence_score === null || isNumber(value.confidence_score))
  )
}

function isAudienceSegment(value: unknown): value is AudienceSegmentResponse {
  if (!isObject(value)) return false
  return (
    isString(value.trace_id) &&
    isString(value.id) &&
    isString(value.name) &&
    isString(value.description) &&
    isArrayOf(value.topic_cluster_ids, isString) &&
    isNumber(value.size_index) &&
    (value.buying_power === 'high' ||
      value.buying_power === 'medium' ||
      value.buying_power === 'low') &&
    isString(value.buying_power_reason) &&
    isArrayOf(value.brand_categories, isString) &&
    isArrayOf(value.supporting_articles, isArticle) &&
    isNumber(value.commercial_confidence) &&
    isString(value.commercial_confidence_reason)
  )
}

function isRejectedArticle(value: unknown): value is RejectedArticleResponse {
  return isObject(value) && isArticle(value.article) && isString(value.reason)
}

function isCommercialSkip(
  value: unknown,
): value is CommercialSkippedClusterResponse {
  return isNamedSkip(value)
}

function isProviderSkip(
  value: unknown,
): value is ProviderSkippedClusterResponse {
  return isNamedSkip(value) && isObject(value) && isString(value.trace_id)
}

function isNamedSkip(value: unknown): boolean {
  return (
    isObject(value) &&
    isString(value.cluster_id) &&
    isString(value.cluster_name) &&
    isString(value.reason)
  )
}

function isValidationDrop(
  value: unknown,
): value is DroppedAudienceDecisionResponse {
  if (!isObject(value)) return false
  return (
    isString(value.trace_id) &&
    isString(value.cluster_id) &&
    typeof value.source_known === 'boolean' &&
    (value.phase === 'initial' || value.phase === 'revision') &&
    isString(value.drop_code) &&
    isArrayOf(value.issues, (issue): boolean => {
      return (
        isObject(issue) &&
        isString(issue.code) &&
        (issue.reference_id === null || isString(issue.reference_id))
      )
    })
  )
}

function isAudienceTrace(
  value: unknown,
): value is AudienceDecisionTraceResponse {
  if (!isObject(value)) return false
  if (
    !isString(value.trace_id) ||
    !isString(value.cluster_id) ||
    !(value.cluster_name === null || isString(value.cluster_name)) ||
    typeof value.source_known !== 'boolean' ||
    !isTraceOutcome(value.final_outcome) ||
    !isArrayOf(value.events, isAudienceTraceEvent)
  ) {
    return false
  }
  return value.events.every((event, index) => event.sequence === index + 1)
}

function isAudienceTraceEvent(
  value: unknown,
): value is AudienceTraceEventResponse {
  if (!isObject(value)) return false
  return (
    isNumber(value.sequence) &&
    Number.isInteger(value.sequence) &&
    value.sequence > 0 &&
    (value.phase === 'initial' ||
      value.phase === 'revision' ||
      value.phase === 'final') &&
    isTraceEventCode(value.code) &&
    (value.outcome_code === null || isString(value.outcome_code)) &&
    isArrayOf(value.issues, isDecisionIssue)
  )
}

function isDecisionIssue(value: unknown): boolean {
  return (
    isObject(value) &&
    isString(value.code) &&
    (value.reference_id === null || isString(value.reference_id))
  )
}

function isTraceEventCode(value: unknown): value is AudienceTraceEventCode {
  return (
    value === 'generation_requested' ||
    value === 'decision_received' ||
    value === 'validation_passed' ||
    value === 'validation_failed' ||
    value === 'revision_requested' ||
    value === 'revision_failed' ||
    value === 'audience_published' ||
    value === 'provider_skipped' ||
    value === 'decision_dropped'
  )
}

function isTraceOutcome(value: unknown): boolean {
  return (
    value === 'published' ||
    value === 'provider_skipped' ||
    value === 'validation_dropped'
  )
}

function hasValidTraceAssociations(
  value: AudienceAnalysisResponse,
): boolean {
  const tracesById = new Map(
    value.audience_traces.map((trace) => [trace.trace_id, trace]),
  )
  if (tracesById.size !== value.audience_traces.length) return false

  const associations = [
    ...value.audience_segments.map((segment) => ({
      traceId: segment.trace_id,
      outcome: 'published',
      clusterId:
        segment.topic_cluster_ids.length === 1
          ? segment.topic_cluster_ids[0]
          : null,
      sourceKnown: true,
    } as const)),
    ...value.provider_skips.map((skipped) => ({
      traceId: skipped.trace_id,
      outcome: 'provider_skipped',
      clusterId: skipped.cluster_id,
      sourceKnown: true,
    } as const)),
    ...value.validation_drops.map((dropped) => ({
      traceId: dropped.trace_id,
      outcome: 'validation_dropped',
      clusterId: dropped.cluster_id,
      sourceKnown: dropped.source_known,
    } as const)),
  ]
  if (
    associations.length !== value.audience_traces.length ||
    new Set(associations.map((item) => item.traceId)).size !==
      associations.length
  ) {
    return false
  }
  return associations.every((association) => {
    const trace = tracesById.get(association.traceId)
    return (
      trace?.final_outcome === association.outcome &&
      trace.cluster_id === association.clusterId &&
      trace.source_known === association.sourceKnown
    )
  })
}

function isMetrics(value: unknown): boolean {
  return (
    isObject(value) &&
    isTopicMetrics(value.topic_analysis) &&
    isFunnelMetrics(value.audience_funnel) &&
    isWorkflowMetrics(value.workflow)
  )
}

function isTopicMetrics(
  value: unknown,
): value is TopicAnalysisMetricsResponse {
  return hasNumberFields(value, [
    'fetched_article_count',
    'rejected_article_count',
    'eligible_article_count',
    'top_n_omitted_article_count',
    'selected_article_count',
    'summary_available_article_count',
    'summary_missing_article_count',
    'topic_cluster_count',
    'clustered_article_count',
    'unclustered_article_count',
    'selected_pageviews',
  ])
}

function isFunnelMetrics(
  value: unknown,
): value is AudienceFunnelMetricsResponse {
  return (
    hasNumberFields(value, [
      'topic_cluster_count',
      'commercial_eligible_cluster_count',
      'commercial_skipped_cluster_count',
      'prepared_cluster_count',
      'final_segment_count',
      'provider_skipped_cluster_count',
      'validation_dropped_source_cluster_count',
      'unmatched_provider_output_count',
      'commercial_eligible_pageviews',
      'represented_audience_pageviews',
    ]) &&
    isObject(value) &&
    isNumberRecord(value.commercial_skip_counts_by_reason)
  )
}

function isWorkflowMetrics(
  value: unknown,
): value is AudienceWorkflowMetricsResponse {
  return (
    hasNumberFields(value, [
      'initial_decision_count',
      'initial_valid_decision_count',
      'initial_invalid_report_count',
      'revision_count',
      'revision_requested_cluster_count',
      'revision_decision_count',
      'revision_valid_decision_count',
      'final_valid_decision_count',
      'final_segment_count',
      'final_provider_skip_count',
      'dropped_source_cluster_count',
      'dropped_unmatched_decision_count',
      'provider_call_count',
      'provider_input_tokens',
      'provider_output_tokens',
      'provider_total_tokens',
      'provider_elapsed_seconds',
      'validation_issue_count',
    ]) &&
    isObject(value) &&
    isNumberRecord(value.validation_issue_counts_by_code) &&
    isNumberRecord(value.drop_counts_by_code)
  )
}

function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  return isObject(value) && isApiErrorDetail(value.error)
}

function isApiErrorDetail(
  value: unknown,
): value is ApiErrorResponse['error'] {
  return (
    isObject(value) &&
    hasExactKeys(value, ['code', 'message']) &&
    isString(value.code) &&
    isString(value.message)
  )
}

function hasNumberFields(value: unknown, fields: readonly string[]): boolean {
  return (
    isObject(value) &&
    fields.every((field) => isNumber(value[field]))
  )
}

function isNumberRecord(value: unknown, dateKeys = false): boolean {
  if (!isObject(value)) return false
  return Object.entries(value).every(([key, item]) => {
    return (!dateKeys || ISO_DATE_PATTERN.test(key)) && isNumber(item)
  })
}

function isArrayOf<T>(
  value: unknown,
  predicate: (item: unknown) => item is T,
): value is T[]
function isArrayOf(
  value: unknown,
  predicate: (item: unknown) => boolean,
): value is unknown[]
function isArrayOf(
  value: unknown,
  predicate: (item: unknown) => boolean,
): value is unknown[] {
  return Array.isArray(value) && value.every(predicate)
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasExactKeys(value: JsonObject, expectedKeys: readonly string[]) {
  const keys = Object.keys(value)
  return (
    keys.length === expectedKeys.length &&
    expectedKeys.every((key) => Object.hasOwn(value, key))
  )
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 1
}

function isIntegerInRange(
  value: unknown,
  minimum: number,
  maximum: number,
): value is number {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= minimum &&
    value <= maximum
  )
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function isString(value: unknown): value is string {
  return typeof value === 'string'
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isIsoDate(value: unknown): value is string {
  return isString(value) && ISO_DATE_PATTERN.test(value)
}
