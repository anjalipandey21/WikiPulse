export type ReviewRunStatus =
  | 'running'
  | 'pending_review'
  | 'editing'
  | 'completed'
  | 'expired'
  | 'failed'

export type RejectReasonCode =
  | 'safety_concern'
  | 'insufficient_evidence'
  | 'misleading_recommendation'
  | 'not_commercially_useful'
  | 'duplicate_audience'
  | 'other'

export type AnalystEditableField =
  | 'audience_positioning'
  | 'supporting_evidence'
  | 'buying_power'
  | 'brand_categories'
  | 'commercial_confidence'

export type EditDropCode =
  | 'edit_provider_failed'
  | 'edit_provider_refused'
  | 'edit_provider_missing_output'
  | 'edit_zero_decisions'
  | 'edit_multiple_decisions'
  | 'edit_wrong_cluster'
  | 'edit_provider_skip_not_allowed'
  | 'edit_unsupported_references'
  | 'edit_validation_failed'
  | 'edit_intent_conformance_failed'
  | 'edit_internal_failure'

export type ReviewJourneyEventCode =
  | 'generation_requested'
  | 'decision_received'
  | 'validation_passed'
  | 'validation_failed'
  | 'revision_requested'
  | 'revision_failed'
  | 'provider_skipped'
  | 'decision_dropped'
  | 'review_requested'
  | 'analyst_approved'
  | 'analyst_rejected'
  | 'analyst_edit_requested'
  | 'edited_decision_received'
  | 'edited_decision_validated'
  | 'edited_audience_published'
  | 'analyst_edit_dropped'
  | 'review_expired'
  | 'audience_published'

export interface ReviewDailyView {
  day: string
  pageviews: number
}

export interface ReviewArticle {
  title: string
  normalized_title: string
  url: string
  weekly_views: number
  daily_views: ReviewDailyView[]
  summary: string | null
  analysis_start_date: string
  analysis_end_date: string
}

export interface ReviewEvidence {
  reference_id: string
  article: ReviewArticle
}

export interface ReviewRecommendation {
  audience_id: string
  name: string
  description: string
  topic_cluster_ids: string[]
  size_index: number
  buying_power: 'high' | 'medium' | 'low'
  buying_power_reason: string
  brand_categories: string[]
  supporting_article_reference_ids: string[]
  supporting_articles: ReviewArticle[]
  commercial_confidence: number
  commercial_confidence_reason: string
}

interface ReviewDisplayContext {
  review_id: string
  cluster_id: string
  position: number
  total_reviews: number
  cluster_name: string
  cluster_pageviews: number
  article_count: number
  size_index: number
  topic_confidence: number
  original_recommendation: ReviewRecommendation
  evidence: ReviewEvidence[]
  edit_available: boolean
}

export interface PendingReview extends ReviewDisplayContext {
  status: 'pending_review'
  expected_version: number
}

export interface EditingReview extends ReviewDisplayContext {
  status: 'editing'
  edit_available: false
}

export type CurrentReview = PendingReview | EditingReview

export interface ReviewProgress {
  total_reviews: number
  completed_reviews: number
  queued_reviews: number
  current_position: number | null
}

export interface PublishedAudience {
  review_id: string
  cluster_id: string
  trace_id: string
  publication_source: 'original' | 'analyst_edit'
  audience: ReviewRecommendation
}

export interface RejectedReview {
  review_id: string
  cluster_id: string
  cluster_name: string
  reason_code: RejectReasonCode
}

export interface EditValidationDrop {
  review_id: string
  cluster_id: string
  cluster_name: string
  drop_code: EditDropCode
}

export interface ExpiredReview {
  review_id: string
  cluster_id: string
  cluster_name: string
}

export interface ProviderSkipReview {
  trace_id: string
  cluster_id: string
  cluster_name: string
  reason: string
}

export interface ValidationDropReview {
  trace_id: string
  cluster_id: string
  cluster_name: string | null
  source_known: boolean
  phase: 'initial' | 'revision'
  drop_code: string
  issue_codes: string[]
}

export interface ReviewJourney {
  trace_id: string
  cluster_id: string
  cluster_name: string | null
  source_known: boolean
  final_outcome: string
  review_id: string | null
  events: Array<{
    sequence: number
    phase: 'initial' | 'revision' | 'review' | 'edit' | 'final'
    code: ReviewJourneyEventCode
    outcome_code: string | null
    issues: Array<{ code: string; reference_id: string | null }>
  }>
}

export interface ReviewWorkflowMetrics {
  initial_decision_count: number
  initial_valid_decision_count: number
  initial_invalid_report_count: number
  revision_count: number
  revision_requested_cluster_count: number
  revision_decision_count: number
  revision_valid_decision_count: number
  final_valid_decision_count: number
  final_segment_count: number
  final_provider_skip_count: number
  dropped_source_cluster_count: number
  dropped_unmatched_decision_count: number
  provider_call_count: number
  provider_input_tokens: number
  provider_output_tokens: number
  provider_total_tokens: number
  provider_elapsed_seconds: number
  validation_issue_count: number
  validation_issue_counts_by_code: Array<{ code: string; count: number }>
  drop_counts_by_code: Array<{ code: string; count: number }>
}

export interface AudienceReviewRun {
  run_id: string
  status: ReviewRunStatus
  is_complete: boolean
  created_at: string
  expires_at: string
  progress: ReviewProgress
  current_review: CurrentReview | null
  published_audiences: PublishedAudience[]
  rejected_reviews: RejectedReview[]
  edit_validation_drops: EditValidationDrop[]
  expired_reviews: ExpiredReview[]
  provider_skips: ProviderSkipReview[]
  validation_drops: ValidationDropReview[]
  journey: ReviewJourney[]
  automatic_workflow_metrics: ReviewWorkflowMetrics
  failure_code: 'automatic_workflow_failed' | 'review_projection_failed' | null
}

export interface ReviewCommandReceipt {
  command_id: string
  type: 'approve' | 'reject' | 'edit_recommendation'
  review_id: string
  cluster_id: string
  accepted: true
  idempotent_replay: boolean
  resulting_status: 'published' | 'rejected' | 'edit_validation_dropped'
  run_status: ReviewRunStatus
}

export interface AudienceReviewCommandResponse {
  receipt: ReviewCommandReceipt
  run: AudienceReviewRun
}

export interface AssistantCitation {
  article_title: string
  article_url: string | null
  audience_label: string
  relevance: string
}

export interface AudienceQuestionResponse {
  answer: string
  citations: AssistantCitation[]
  evidence_status: 'grounded' | 'insufficient_evidence'
  suggested_follow_up_questions: string[]
}

interface CommandIdentity {
  command_id: string
  review_id: string
  cluster_id: string
  expected_version: number
}

export type ReviewCommandRequest =
  | ({ type: 'approve' } & CommandIdentity)
  | ({
      type: 'reject'
      reason_code: RejectReasonCode
      private_note?: string
    } & CommandIdentity)
  | ({
      type: 'edit_recommendation'
      feedback: string
      fields_to_change: AnalystEditableField[]
    } & CommandIdentity)

export interface ReviewApiErrorPayload {
  error: { code: string; message: string }
}
