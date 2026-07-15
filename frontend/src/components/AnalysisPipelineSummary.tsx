import type {
  AnalysisProgressStage,
  AudienceAnalysisResponse,
} from '../api/types'
import {
  derivePipelineState,
  pipelineStatusLabel,
} from '../stitchEnhancements.js'

interface AnalysisPipelineSummaryProps {
  stage: AnalysisProgressStage | null
  result?: AudienceAnalysisResponse | null
  failed?: boolean
}

export function AnalysisPipelineSummary({
  stage,
  result = null,
  failed = false,
}: AnalysisPipelineSummaryProps) {
  const { steps, description } = derivePipelineState(stage, result, failed)

  return (
    <section
      className="analysis-pipeline"
      aria-label="Analysis pipeline"
      data-failed={failed || undefined}
    >
      <ol>
        {steps.map((step, index) => (
          <li
            className={`pipeline-step pipeline-step-${step.status}`}
            key={step.id}
            data-status={step.status}
            aria-current={step.status === 'active' ? 'step' : undefined}
          >
            <span className="pipeline-step-mark" aria-hidden="true">
              {step.status === 'complete'
                ? '✓'
                : step.status === 'failed'
                  ? '!'
                  : String(index + 1).padStart(2, '0')}
            </span>
            <span>{step.label}</span>
            <span className="visually-hidden">
              {pipelineStatusLabel(step.status)}
            </span>
          </li>
        ))}
      </ol>
      <p>{description}</p>
    </section>
  )
}
