import type {
  AudienceReviewRun,
  EditDropCode,
} from './api/audienceReviewTypes'

export const REVIEW_RUN_STORAGE_KEY = 'wikipulse.activeReviewRunId'

export function normalizeAssistantQuestion(value: string): string {
  return value.normalize('NFC').trim().replace(/\s+/gu, ' ')
}

export type ReviewUiStatus =
  | 'idle'
  | 'starting'
  | 'pending_review'
  | 'submitting_approve'
  | 'submitting_reject'
  | 'submitting_edit'
  | 'editing'
  | 'completed'
  | 'expired'
  | 'failed'
  | 'conflict'

export const EDIT_DROP_TEXT: Record<EditDropCode, string> = {
  edit_provider_failed: 'The edit service could not complete the request.',
  edit_provider_refused: 'The edit could not be generated safely.',
  edit_provider_missing_output: 'The edit returned no usable recommendation.',
  edit_zero_decisions: 'The edit returned no recommendation.',
  edit_multiple_decisions: 'The edit returned more than one recommendation.',
  edit_wrong_cluster: 'The edit targeted the wrong topic.',
  edit_provider_skip_not_allowed: 'The edit did not return an audience recommendation.',
  edit_unsupported_references: 'The edit used evidence outside the approved set.',
  edit_validation_failed: 'The edited recommendation did not pass validation.',
  edit_intent_conformance_failed: 'The edit did not follow the selected change areas.',
  edit_internal_failure: 'The edit could not be finalized safely.',
}

export function normalizeAnalystFeedback(value: string): string {
  return value.normalize('NFC').trim().replace(/\s+/gu, ' ')
}

export function deriveReviewUiStatus(
  run: AudienceReviewRun | null,
  phase: Exclude<ReviewUiStatus, 'pending_review' | 'editing' | 'completed' | 'expired'>,
): ReviewUiStatus {
  if (phase !== 'idle') return phase
  if (!run) return 'idle'
  if (run.status === 'editing') return 'editing'
  if (run.status === 'expired') return 'expired'
  if (run.status === 'failed') return 'failed'
  if (run.is_complete || run.status === 'completed') return 'completed'
  return 'pending_review'
}
