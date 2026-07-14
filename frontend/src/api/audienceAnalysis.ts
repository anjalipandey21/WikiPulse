import type {
  ApiErrorResponse,
  ArticleResponse,
  AudienceAnalysisResponse,
  AudienceFunnelMetricsResponse,
  AudienceSegmentResponse,
  AudienceWorkflowMetricsResponse,
  CommercialSkippedClusterResponse,
  DroppedAudienceDecisionResponse,
  ProviderSkippedClusterResponse,
  RejectedArticleResponse,
  TopicAnalysisMetricsResponse,
  TopicClusterResponse,
} from './types'

const ENDPOINT = '/api/audience-analysis'
const ISO_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/

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
    typeof value.is_publishable === 'boolean' &&
    isMetrics(value.metrics)
  )
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
  return isNamedSkip(value)
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
  return (
    isObject(value) &&
    isObject(value.error) &&
    isString(value.error.code) &&
    isString(value.error.message)
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

function isString(value: unknown): value is string {
  return typeof value === 'string'
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isIsoDate(value: unknown): value is string {
  return isString(value) && ISO_DATE_PATTERN.test(value)
}
