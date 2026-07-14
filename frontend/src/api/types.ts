export type IsoDateString = string

export type BuyingPower = 'high' | 'medium' | 'low'

export type DecisionPhase = 'initial' | 'revision'

export type AudienceTracePhase = DecisionPhase | 'final'

export type AudienceTraceEventCode =
  | 'generation_requested'
  | 'decision_received'
  | 'validation_passed'
  | 'validation_failed'
  | 'revision_requested'
  | 'revision_failed'
  | 'audience_published'
  | 'provider_skipped'
  | 'decision_dropped'

export type AudienceTraceOutcome =
  | 'published'
  | 'provider_skipped'
  | 'validation_dropped'

export type AnalysisProgressStage =
  | 'waiting_for_slot'
  | 'fetching_pageviews'
  | 'selecting_articles'
  | 'enriching_summaries'
  | 'modeling_topics'
  | 'routing_commercial_clusters'
  | 'preparing_audience_evidence'
  | 'generating_audience_decisions'
  | 'validating_audience_decisions'
  | 'revising_audience_decisions'
  | 'validating_revised_decisions'
  | 'finalizing_audience_results'
  | 'assembling_response'

export interface ArticleResponse {
  title: string
  normalized_title: string
  url: string
  weekly_views: number
  daily_views: Record<IsoDateString, number>
  summary: string | null
  analysis_start_date: IsoDateString
  analysis_end_date: IsoDateString
}

export interface TopicClusterResponse {
  id: string
  name: string
  description: string | null
  articles: ArticleResponse[]
  keywords: string[]
  total_views: number
  article_count: number
  confidence_score: number | null
}

export interface AudienceSegmentResponse {
  trace_id: string
  id: string
  name: string
  description: string
  topic_cluster_ids: string[]
  size_index: number
  buying_power: BuyingPower
  buying_power_reason: string
  brand_categories: string[]
  supporting_articles: ArticleResponse[]
  commercial_confidence: number
  commercial_confidence_reason: string
}

export interface RejectedArticleResponse {
  article: ArticleResponse
  reason: string
}

export interface CommercialSkippedClusterResponse {
  cluster_id: string
  cluster_name: string
  reason: string
}

export interface ProviderSkippedClusterResponse {
  trace_id: string
  cluster_id: string
  cluster_name: string
  reason: string
}

export interface AudienceDecisionIssueResponse {
  code: string
  reference_id: string | null
}

export interface DroppedAudienceDecisionResponse {
  trace_id: string
  cluster_id: string
  source_known: boolean
  phase: DecisionPhase
  drop_code: string
  issues: AudienceDecisionIssueResponse[]
}

export interface AudienceTraceEventResponse {
  sequence: number
  phase: AudienceTracePhase
  code: AudienceTraceEventCode
  outcome_code: string | null
  issues: AudienceDecisionIssueResponse[]
}

export interface AudienceDecisionTraceResponse {
  trace_id: string
  cluster_id: string
  cluster_name: string | null
  source_known: boolean
  final_outcome: AudienceTraceOutcome
  events: AudienceTraceEventResponse[]
}

export interface TopicAnalysisMetricsResponse {
  fetched_article_count: number
  rejected_article_count: number
  eligible_article_count: number
  top_n_omitted_article_count: number
  selected_article_count: number
  summary_available_article_count: number
  summary_missing_article_count: number
  topic_cluster_count: number
  clustered_article_count: number
  unclustered_article_count: number
  selected_pageviews: number
}

export interface AudienceFunnelMetricsResponse {
  topic_cluster_count: number
  commercial_eligible_cluster_count: number
  commercial_skipped_cluster_count: number
  prepared_cluster_count: number
  final_segment_count: number
  provider_skipped_cluster_count: number
  validation_dropped_source_cluster_count: number
  unmatched_provider_output_count: number
  commercial_eligible_pageviews: number
  represented_audience_pageviews: number
  commercial_skip_counts_by_reason: Record<string, number>
}

export interface AudienceWorkflowMetricsResponse {
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
  validation_issue_counts_by_code: Record<string, number>
  drop_counts_by_code: Record<string, number>
}

export interface AudienceAnalysisMetricsResponse {
  topic_analysis: TopicAnalysisMetricsResponse
  audience_funnel: AudienceFunnelMetricsResponse
  workflow: AudienceWorkflowMetricsResponse
}

export interface AudienceAnalysisResponse {
  topics: TopicClusterResponse[]
  audience_segments: AudienceSegmentResponse[]
  unclustered_articles: ArticleResponse[]
  rejected_articles: RejectedArticleResponse[]
  commercial_skips: CommercialSkippedClusterResponse[]
  provider_skips: ProviderSkippedClusterResponse[]
  validation_drops: DroppedAudienceDecisionResponse[]
  audience_traces: AudienceDecisionTraceResponse[]
  is_publishable: boolean
  metrics: AudienceAnalysisMetricsResponse
}

export interface ApiErrorResponse {
  error: {
    code: string
    message: string
  }
}

export interface AudienceAnalysisProgressEvent {
  type: 'progress'
  sequence: number
  stage: AnalysisProgressStage
}

export interface AudienceAnalysisResultEvent {
  type: 'result'
  sequence: number
  result: AudienceAnalysisResponse
}

export interface AudienceAnalysisErrorEvent {
  type: 'error'
  sequence: number
  status_code: number
  error: ApiErrorResponse['error']
}

export type AudienceAnalysisStreamEvent =
  | AudienceAnalysisProgressEvent
  | AudienceAnalysisResultEvent
  | AudienceAnalysisErrorEvent
