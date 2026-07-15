import type {
  AnalysisProgressStage,
  AudienceAnalysisResponse,
} from './api/types'
import {
  formatDate,
  formatExactNumber,
  formatPageviews,
  formatPercent,
  humanizeCode,
} from './formatters.js'

export interface TickerInsight {
  label: string
  value: string
}

export function buildTickerInsights(
  result: AudienceAnalysisResponse,
): TickerInsight[] {
  const insights: TickerInsight[] = []
  const topCluster = result.topics.reduce(
    (current, topic) => (
      current === null || topic.total_views > current.total_views
        ? topic
        : current
    ),
    null as AudienceAnalysisResponse['topics'][number] | null,
  )
  const topCommercialAudience = result.audience_segments.reduce(
    (current, segment) => (
      current === null ||
      segment.commercial_confidence > current.commercial_confidence
        ? segment
        : current
    ),
    null as AudienceAnalysisResponse['audience_segments'][number] | null,
  )
  const periodArticle =
    topCluster?.articles[0] ??
    result.unclustered_articles[0] ??
    result.rejected_articles[0]?.article ??
    result.audience_segments[0]?.supporting_articles[0]
  const topicMetrics = result.metrics.topic_analysis
  const finalAudienceCount = result.metrics.audience_funnel.final_segment_count

  if (topCluster) {
    insights.push({ label: 'Top cluster', value: topCluster.name })
    insights.push({
      label: 'Top-cluster attention',
      value: formatPageviews(topCluster.total_views),
    })
    if (topCluster.confidence_score !== null) {
      insights.push({
        label: 'Topic confidence',
        value: formatPercent(topCluster.confidence_score),
      })
    }
  }

  insights.push({
    label: 'Published audiences',
    value: formatExactNumber(finalAudienceCount),
  })

  if (topCommercialAudience) {
    insights.push({
      label: 'Top commercial confidence',
      value: `${formatPercent(topCommercialAudience.commercial_confidence)} · ${topCommercialAudience.name}`,
    })
    insights.push({
      label: 'Buying power',
      value: `${humanizeCode(topCommercialAudience.buying_power)} · ${topCommercialAudience.name}`,
    })
  }

  if (periodArticle) {
    insights.push({
      label: 'Analysis period',
      value: `${formatDate(periodArticle.analysis_start_date)} – ${formatDate(periodArticle.analysis_end_date)}`,
    })
  }

  if (topicMetrics.selected_article_count > 0) {
    insights.push({
      label: 'Summary coverage',
      value: `${formatExactNumber(topicMetrics.summary_available_article_count)} of ${formatExactNumber(topicMetrics.selected_article_count)} articles`,
    })
  }

  if (topicMetrics.clustered_article_count > 0) {
    insights.push({
      label: 'Clustered articles',
      value: formatExactNumber(topicMetrics.clustered_article_count),
    })
  }

  return insights
}

type PipelineStepId =
  | 'fetched'
  | 'summarized'
  | 'clustered'
  | 'generated'
  | 'validated'
  | 'published'

export type PipelineStepStatus =
  | 'pending'
  | 'active'
  | 'complete'
  | 'failed'
  | 'empty'

export interface PipelineStep {
  id: PipelineStepId
  label: string
  status: PipelineStepStatus
}

const PIPELINE_STEPS = [
  { id: 'fetched', label: 'Fetched' },
  { id: 'summarized', label: 'Summarized' },
  { id: 'clustered', label: 'Clustered' },
  { id: 'generated', label: 'Generated' },
  { id: 'validated', label: 'Validated' },
  { id: 'published', label: 'Published' },
] as const

interface StageProgress {
  completedSteps: number
  activeStep: PipelineStepId | null
  description: string
}

export const ANALYSIS_PIPELINE_STAGE_MAP: Readonly<
  Record<AnalysisProgressStage, StageProgress>
> = {
  waiting_for_slot: {
    completedSteps: 0,
    activeStep: null,
    description: 'Waiting for the shared analysis slot.',
  },
  fetching_pageviews: {
    completedSteps: 0,
    activeStep: 'fetched',
    description: 'Fetching seven complete days of Wikipedia pageviews.',
  },
  selecting_articles: {
    completedSteps: 1,
    activeStep: null,
    description: 'Selecting eligible articles from the weekly pageviews.',
  },
  enriching_summaries: {
    completedSteps: 1,
    activeStep: 'summarized',
    description: 'Adding public Wikipedia summaries.',
  },
  modeling_topics: {
    completedSteps: 2,
    activeStep: 'clustered',
    description: 'Grouping related articles into local topic clusters.',
  },
  routing_commercial_clusters: {
    completedSteps: 3,
    activeStep: null,
    description: 'Routing commercially safe clusters.',
  },
  preparing_audience_evidence: {
    completedSteps: 3,
    activeStep: 'generated',
    description: 'Preparing bounded evidence for audience generation.',
  },
  generating_audience_decisions: {
    completedSteps: 3,
    activeStep: 'generated',
    description: 'Generating structured audience decisions.',
  },
  validating_audience_decisions: {
    completedSteps: 4,
    activeStep: 'validated',
    description: 'Validating audience decisions and evidence references.',
  },
  revising_audience_decisions: {
    completedSteps: 4,
    activeStep: 'validated',
    description: 'Applying the one bounded automatic revision.',
  },
  validating_revised_decisions: {
    completedSteps: 4,
    activeStep: 'validated',
    description: 'Validating the bounded revision results.',
  },
  finalizing_audience_results: {
    completedSteps: 4,
    activeStep: 'validated',
    description: 'Finalizing valid outcomes and explicit drops.',
  },
  assembling_response: {
    completedSteps: 5,
    activeStep: null,
    description: 'Assembling the public dashboard result.',
  },
}

export function derivePipelineState(
  stage: AnalysisProgressStage | null,
  result: AudienceAnalysisResponse | null,
  failed = false,
): { steps: PipelineStep[]; description: string } {
  if (result !== null) {
    const hasPublishedAudience = result.audience_segments.length > 0
    return {
      steps: PIPELINE_STEPS.map((step, index) => ({
        ...step,
        status:
          index < PIPELINE_STEPS.length - 1
            ? 'complete'
            : hasPublishedAudience
              ? 'complete'
              : 'empty',
      })),
      description: hasPublishedAudience
        ? `${result.audience_segments.length} publishable ${result.audience_segments.length === 1 ? 'audience' : 'audiences'} in the completed brief.`
        : 'No publishable audience',
    }
  }

  const progress = stage === null
    ? undefined
    : ANALYSIS_PIPELINE_STAGE_MAP[stage]

  if (progress === undefined) {
    return {
      steps: PIPELINE_STEPS.map((step) => ({ ...step, status: 'pending' })),
      description: failed
        ? 'Analysis stopped before a recognized public stage was available.'
        : 'Waiting for a recognized analysis stage.',
    }
  }

  const steps: PipelineStep[] = PIPELINE_STEPS.map((step, index) => {
    let status: PipelineStepStatus = 'pending'
    if (index < progress.completedSteps) status = 'complete'
    if (step.id === progress.activeStep) status = failed ? 'failed' : 'active'
    return { ...step, status }
  })

  return {
    steps,
    description: failed
      ? `Analysis stopped while ${progress.description.toLowerCase()}`
      : progress.description,
  }
}

export function pipelineStatusLabel(status: PipelineStepStatus): string {
  switch (status) {
    case 'complete': return 'Complete'
    case 'active': return 'Current stage'
    case 'failed': return 'Stopped at this stage'
    case 'empty': return 'No publishable audience'
    default: return 'Pending'
  }
}

export interface CountUpAnimationOptions {
  from: number
  to: number
  duration: number
  integer: boolean
  reducedMotion: boolean
  onValue: (value: number) => void
  requestFrame: (callback: FrameRequestCallback) => number
  cancelFrame: (handle: number) => void
}

export function startCountUpAnimation({
  from,
  to,
  duration,
  integer,
  reducedMotion,
  onValue,
  requestFrame,
  cancelFrame,
}: CountUpAnimationOptions): () => void {
  if (reducedMotion || duration <= 0 || from === to) {
    onValue(to)
    return () => undefined
  }

  let frameHandle = 0
  let startTime: number | null = null
  let stopped = false

  const update = (time: number) => {
    if (stopped) return
    startTime ??= time
    const progress = Math.min((time - startTime) / duration, 1)
    const easedProgress = 1 - (1 - progress) ** 3
    const rawValue = from + (to - from) * easedProgress
    onValue(integer ? Math.round(rawValue) : rawValue)

    if (progress < 1) {
      frameHandle = requestFrame(update)
    }
  }

  frameHandle = requestFrame(update)

  return () => {
    stopped = true
    cancelFrame(frameHandle)
  }
}
