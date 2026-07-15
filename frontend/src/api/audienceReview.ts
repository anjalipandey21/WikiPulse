import type {
  AudienceReviewCommandResponse,
  AudienceReviewRun,
  AudienceQuestionResponse,
  ReviewApiErrorPayload,
  ReviewCommandRequest,
} from './audienceReviewTypes'

const REVIEW_ENDPOINT = '/api/audience-reviews'

export class AudienceReviewApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(code: string, message: string, status: number) {
    super(message)
    this.name = 'AudienceReviewApiError'
    this.code = code
    this.status = status
  }
}

export function createUuid(): string {
  return crypto.randomUUID()
}

export async function startAudienceReview(
  runId: string,
  signal?: AbortSignal,
): Promise<AudienceReviewRun> {
  return requestReviewRun(REVIEW_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId }),
    signal,
  })
}

export async function getAudienceReview(
  runId: string,
  signal?: AbortSignal,
): Promise<AudienceReviewRun> {
  return requestReviewRun(`${REVIEW_ENDPOINT}/${encodeURIComponent(runId)}`, {
    method: 'GET',
    signal,
  })
}

export async function submitAudienceReviewCommand(
  runId: string,
  command: ReviewCommandRequest,
  signal?: AbortSignal,
): Promise<AudienceReviewCommandResponse> {
  const payload = await requestJson(
    `${REVIEW_ENDPOINT}/${encodeURIComponent(runId)}/commands`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(command),
      signal,
    },
  )
  if (!isCommandResponse(payload)) {
    throw invalidResponseError()
  }
  return payload
}

export async function askWikiPulse(
  runId: string,
  question: string,
  signal?: AbortSignal,
): Promise<AudienceQuestionResponse> {
  const payload = await requestJson(
    `${REVIEW_ENDPOINT}/${encodeURIComponent(runId)}/questions`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal,
    },
  )
  if (!isAudienceQuestionResponse(payload)) throw invalidResponseError()
  return payload
}

async function requestReviewRun(
  url: string,
  init: RequestInit,
): Promise<AudienceReviewRun> {
  const payload = await requestJson(url, init)
  if (!isAudienceReviewRun(payload)) throw invalidResponseError()
  return payload
}

async function requestJson(url: string, init: RequestInit): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(url, {
      ...init,
      headers: { Accept: 'application/json', ...init.headers },
    })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new AudienceReviewApiError(
      'network_error',
      'WikiPulse could not reach the review service.',
      0,
    )
  }

  let payload: unknown = null
  if ((response.headers.get('content-type') ?? '').includes('application/json')) {
    try {
      payload = await response.json()
    } catch {
      payload = null
    }
  }
  if (!response.ok) {
    if (isReviewApiError(payload)) {
      throw new AudienceReviewApiError(
        payload.error.code,
        payload.error.message,
        response.status,
      )
    }
    throw new AudienceReviewApiError(
      'request_failed',
      'The review service returned an unexpected error.',
      response.status,
    )
  }
  return payload
}

function isAudienceReviewRun(value: unknown): value is AudienceReviewRun {
  if (!isObject(value)) return false
  const status = value.status
  const current = value.current_review
  return (
    typeof value.run_id === 'string' &&
    isReviewStatus(status) &&
    typeof value.is_complete === 'boolean' &&
    typeof value.created_at === 'string' &&
    typeof value.expires_at === 'string' &&
    isProgress(value.progress) &&
    (current === null || isCurrentReview(current)) &&
    isArray(value.published_audiences) &&
    isArray(value.rejected_reviews) &&
    isArray(value.edit_validation_drops) &&
    isArray(value.expired_reviews) &&
    isArray(value.provider_skips) &&
    isArray(value.validation_drops) &&
    isArray(value.journey) &&
    isObject(value.automatic_workflow_metrics) &&
    (value.failure_code === null || typeof value.failure_code === 'string')
  )
}

function isCommandResponse(value: unknown): value is AudienceReviewCommandResponse {
  return (
    isObject(value) &&
    isObject(value.receipt) &&
    typeof value.receipt.command_id === 'string' &&
    value.receipt.accepted === true &&
    isAudienceReviewRun(value.run)
  )
}

function isAudienceQuestionResponse(value: unknown): value is AudienceQuestionResponse {
  if (!isObject(value)) return false
  return (
    typeof value.answer === 'string' &&
    ['grounded', 'insufficient_evidence'].includes(String(value.evidence_status)) &&
    isArray(value.citations) &&
    value.citations.every((citation) => (
      isObject(citation) &&
      typeof citation.article_title === 'string' &&
      (citation.article_url === null || typeof citation.article_url === 'string') &&
      typeof citation.audience_label === 'string' &&
      typeof citation.relevance === 'string'
    )) &&
    isArray(value.suggested_follow_up_questions) &&
    value.suggested_follow_up_questions.every((item) => typeof item === 'string')
  )
}

function isCurrentReview(value: unknown): boolean {
  if (!isObject(value) || !isRecommendation(value.original_recommendation)) {
    return false
  }
  const common =
    typeof value.review_id === 'string' &&
    typeof value.cluster_id === 'string' &&
    typeof value.cluster_name === 'string' &&
    isPositiveInteger(value.position) &&
    isPositiveInteger(value.total_reviews) &&
    isNonNegativeNumber(value.cluster_pageviews) &&
    isNonNegativeNumber(value.article_count) &&
    isNonNegativeNumber(value.size_index) &&
    isNonNegativeNumber(value.topic_confidence) &&
    isArray(value.evidence) &&
    typeof value.edit_available === 'boolean'
  if (!common) return false
  if (value.status === 'editing') return value.edit_available === false
  return value.status === 'pending_review' && isPositiveInteger(value.expected_version)
}

function isRecommendation(value: unknown): boolean {
  return (
    isObject(value) &&
    typeof value.audience_id === 'string' &&
    typeof value.name === 'string' &&
    typeof value.description === 'string' &&
    isArray(value.topic_cluster_ids) &&
    isNonNegativeNumber(value.size_index) &&
    ['high', 'medium', 'low'].includes(String(value.buying_power)) &&
    typeof value.buying_power_reason === 'string' &&
    isArray(value.brand_categories) &&
    isArray(value.supporting_articles) &&
    isNonNegativeNumber(value.commercial_confidence) &&
    typeof value.commercial_confidence_reason === 'string'
  )
}

function isProgress(value: unknown): boolean {
  return (
    isObject(value) &&
    isNonNegativeNumber(value.total_reviews) &&
    isNonNegativeNumber(value.completed_reviews) &&
    isNonNegativeNumber(value.queued_reviews) &&
    (value.current_position === null || isPositiveInteger(value.current_position))
  )
}

function isReviewStatus(value: unknown): boolean {
  return [
    'running',
    'pending_review',
    'editing',
    'completed',
    'expired',
    'failed',
  ].includes(String(value))
}

function isReviewApiError(value: unknown): value is ReviewApiErrorPayload {
  return (
    isObject(value) &&
    isObject(value.error) &&
    typeof value.error.code === 'string' &&
    typeof value.error.message === 'string'
  )
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isArray(value: unknown): value is unknown[] {
  return Array.isArray(value)
}

function isNonNegativeNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
}

function invalidResponseError(): AudienceReviewApiError {
  return new AudienceReviewApiError(
    'invalid_response',
    'The review service returned an invalid response.',
    0,
  )
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}
