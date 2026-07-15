import { useCallback, useEffect, useRef, useState } from 'react'

import {
  AudienceReviewApiError,
  createUuid,
  getAudienceReview,
  startAudienceReview,
  submitAudienceReviewCommand,
} from '../api/audienceReview'
import type {
  AnalystEditableField,
  AudienceReviewRun,
  RejectReasonCode,
  ReviewCommandRequest,
} from '../api/audienceReviewTypes'
import { EditingReviewPanel } from './EditingReviewPanel'
import { PendingReviewCard } from './PendingReviewCard'
import { ReviewOutcomeSummary } from './ReviewOutcomeSummary'
import { AskWikiPulseDrawer } from './AskWikiPulseDrawer.js'
import { StatusPanel } from './StatusPanel'
import {
  deriveReviewUiStatus,
  REVIEW_RUN_STORAGE_KEY,
  type ReviewUiStatus,
} from '../reviewUi'

const EDITING_POLL_INTERVAL_MS = 1_750
const MAX_EDITING_POLLS = 20

type RequestPhase =
  | 'idle'
  | 'starting'
  | 'submitting_approve'
  | 'submitting_reject'
  | 'submitting_edit'
  | 'conflict'
  | 'failed'

interface DisplayError {
  code: string
  message: string
}

interface PendingCommand {
  runId: string
  command: ReviewCommandRequest
}

export function AudienceReviewWorkspace() {
  const [run, setRun] = useState<AudienceReviewRun | null>(null)
  const [phase, setPhase] = useState<RequestPhase>('idle')
  const [error, setError] = useState<DisplayError | null>(null)
  const [editingPollAttempt, setEditingPollAttempt] = useState(0)
  const activeControllers = useRef(new Set<AbortController>())
  const pendingCommand = useRef<PendingCommand | null>(null)
  const pendingStartRunId = useRef<string | null>(null)
  const mounted = useRef(true)

  const withController = useCallback(async function withController<T>(work: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const controller = new AbortController()
    activeControllers.current.add(controller)
    try {
      return await work(controller.signal)
    } finally {
      activeControllers.current.delete(controller)
    }
  }, [])

  const acceptRun = useCallback((nextRun: AudienceReviewRun) => {
    setRun(nextRun)
    if (nextRun.status !== 'editing') setEditingPollAttempt(0)
    if (nextRun.is_complete || ['completed', 'expired', 'failed'].includes(nextRun.status)) {
      clearStoredRunId()
    } else {
      writeStoredRunId(nextRun.run_id)
    }
  }, [])

  const refresh = useCallback(async (runId?: string, quiet = false) => {
    const targetRunId = runId ?? readStoredRunId()
    if (!targetRunId) return
    if (!quiet) setError(null)
    try {
      const nextRun = await withController((signal) => getAudienceReview(targetRunId, signal))
      if (!mounted.current) return
      acceptRun(nextRun)
      setPhase('idle')
      return nextRun
    } catch (caught) {
      if (!mounted.current || isAbortError(caught)) return
      const display = toDisplayError(caught)
      if (display.code === 'review_run_not_found') {
        clearStoredRunId()
        setRun(null)
      }
      if (!quiet || display.code === 'review_run_not_found') {
        setError(display)
        setPhase(display.code.includes('conflict') ? 'conflict' : 'failed')
      }
      return null
    }
  }, [acceptRun, withController])

  useEffect(() => {
    mounted.current = true
    const restoreTimer = window.setTimeout(() => {
      const storedRunId = readStoredRunId()
      if (storedRunId) void refresh(storedRunId, true)
    }, 0)
    const controllers = activeControllers.current
    return () => {
      mounted.current = false
      window.clearTimeout(restoreTimer)
      for (const controller of controllers) controller.abort()
      controllers.clear()
    }
  }, [refresh])

  useEffect(() => {
    if (run?.status !== 'editing' || editingPollAttempt >= MAX_EDITING_POLLS) return
    const timer = window.setTimeout(() => {
      void refresh(run.run_id, true).then((nextRun) => {
        if (mounted.current && nextRun?.status === 'editing') {
          setEditingPollAttempt((current) => current + 1)
        }
      })
    }, EDITING_POLL_INTERVAL_MS)
    return () => window.clearTimeout(timer)
  }, [editingPollAttempt, refresh, run?.run_id, run?.status])

  async function start(retryRunId?: string) {
    if (phase === 'starting') return
    const runId = retryRunId ?? createUuid()
    pendingStartRunId.current = runId
    writeStoredRunId(runId)
    setPhase('starting')
    setError(null)
    try {
      const nextRun = await withController((signal) => startAudienceReview(runId, signal))
      if (!mounted.current) return
      pendingStartRunId.current = null
      acceptRun(nextRun)
      setPhase('idle')
    } catch (caught) {
      if (!mounted.current || isAbortError(caught)) return
      const display = toDisplayError(caught)
      setError(display)
      setPhase('failed')
      if (display.code !== 'network_error') {
        pendingStartRunId.current = null
        clearStoredRunId()
      }
    }
  }

  function approve() {
    const current = pendingReview(run)
    if (!current) return
    void submit({
      type: 'approve',
      command_id: createUuid(),
      review_id: current.review_id,
      cluster_id: current.cluster_id,
      expected_version: current.expected_version,
    })
  }

  function reject(reasonCode: RejectReasonCode, privateNote?: string) {
    const current = pendingReview(run)
    if (!current) return
    void submit({
      type: 'reject',
      command_id: createUuid(),
      review_id: current.review_id,
      cluster_id: current.cluster_id,
      expected_version: current.expected_version,
      reason_code: reasonCode,
      ...(privateNote ? { private_note: privateNote } : {}),
    })
  }

  function edit(feedback: string, fieldsToChange: AnalystEditableField[]) {
    const current = pendingReview(run)
    if (!current) return
    void submit({
      type: 'edit_recommendation',
      command_id: createUuid(),
      review_id: current.review_id,
      cluster_id: current.cluster_id,
      expected_version: current.expected_version,
      feedback,
      fields_to_change: fieldsToChange,
    })
  }

  async function submit(command: ReviewCommandRequest, retry = false) {
    if (!run) return
    const request = retry ? pendingCommand.current : { runId: run.run_id, command }
    if (!request) return
    pendingCommand.current = request
    setPhase(commandPhase(request.command.type))
    setError(null)
    try {
      const response = await withController((signal) =>
        submitAudienceReviewCommand(request.runId, request.command, signal),
      )
      if (!mounted.current) return
      pendingCommand.current = null
      acceptRun(response.run)
      setPhase('idle')
    } catch (caught) {
      if (!mounted.current || isAbortError(caught)) return
      const display = toDisplayError(caught)
      setError(display)
      setPhase(isConflictCode(display.code) ? 'conflict' : 'failed')
      if (display.code !== 'network_error') pendingCommand.current = null
      if (display.code === 'review_run_expired') void refresh(request.runId)
    }
  }

  function retryPendingCommand() {
    const request = pendingCommand.current
    if (request) void submit(request.command, true)
  }

  const uiStatus = deriveReviewUiStatus(run, phase)
  const busyAction = phase === 'submitting_approve'
    ? 'approve'
    : phase === 'submitting_reject'
      ? 'reject'
      : phase === 'submitting_edit'
        ? 'edit'
        : pendingCommand.current?.command.type === 'approve'
          ? 'approve'
          : pendingCommand.current?.command.type === 'reject'
            ? 'reject'
            : pendingCommand.current?.command.type === 'edit_recommendation'
              ? 'edit'
              : null

  return (
    <div className="review-workspace" aria-busy={phase === 'starting' || busyAction !== null}>
      <div className="review-intro">
        <div>
          <p className="eyebrow">Bounded human review</p>
          <h2>Review one audience at a time.</h2>
          <p>Approve the validated recommendation, reject it, or request its single bounded edit.</p>
        </div>
        {run && !run.is_complete ? (
          <button className="button button-secondary" type="button" onClick={() => void refresh()}>
            Refresh status
          </button>
        ) : null}
      </div>

      <p className="review-live-status" aria-live="polite">
        {statusMessage(uiStatus, run)}
      </p>

      {error ? (
        <div className="notice notice-error" role="alert">
          <div><strong>{errorTitle(error.code)}</strong><span>{error.message}</span></div>
          {pendingCommand.current ? (
            <button className="text-button" type="button" onClick={retryPendingCommand}>Retry same command</button>
          ) : pendingStartRunId.current ? (
            <button className="text-button" type="button" onClick={() => void start(pendingStartRunId.current ?? undefined)}>Retry same run</button>
          ) : run ? (
            <button className="text-button" type="button" onClick={() => void refresh()}>Reconcile run</button>
          ) : null}
        </div>
      ) : null}

      {!run && phase === 'starting' ? (
        <StatusPanel eyebrow="Starting analyst review" title="Preparing review candidates" message="Running the approved automatic workflow before the first review." icon={<span className="loading-ring" />} busy />
      ) : null}

      {!run && phase !== 'starting' ? (
        <StatusPanel
          eyebrow="Analyst Review"
          title={pendingStartRunId.current ? 'Recover the review run' : 'Start a review run'}
          message="WikiPulse will prepare the weekly analysis, then pause at the first valid audience recommendation."
          actionLabel={pendingStartRunId.current ? 'Retry same run' : 'Start analyst review'}
          onAction={() => void start(pendingStartRunId.current ?? undefined)}
        />
      ) : null}

      {run?.current_review?.status === 'pending_review' ? (
        <PendingReviewCard review={run.current_review} busyAction={busyAction} onApprove={approve} onReject={reject} onEdit={edit} />
      ) : null}

      {run?.current_review?.status === 'editing' ? (
        <EditingReviewPanel review={run.current_review} onRefresh={() => void refresh()} />
      ) : null}

      {run && run.is_complete && run.progress.total_reviews === 0 ? (
        <StatusPanel eyebrow="Review complete" title="No audiences required review" message="Provider skips and deterministic validation drops remain recorded below." />
      ) : null}

      {run?.status === 'expired' ? (
        <div className="notice notice-warning" role="status">This review run expired. Completed outcomes remain unchanged.</div>
      ) : null}

      {run?.status === 'failed' ? (
        <div className="notice notice-error" role="alert"><div><strong>Review run failed safely</strong><span>No raw provider details are available.</span></div></div>
      ) : null}

      {run ? <ReviewOutcomeSummary run={run} /> : null}
      {run?.is_complete && run.published_audiences.some(
        (published) => published.audience.supporting_articles.length > 0,
      ) ? (
        <AskWikiPulseDrawer
          runId={run.run_id}
          audienceCount={run.published_audiences.length}
        />
      ) : null}
    </div>
  )
}

function pendingReview(run: AudienceReviewRun | null) {
  return run?.current_review?.status === 'pending_review' ? run.current_review : null
}

function commandPhase(type: ReviewCommandRequest['type']): RequestPhase {
  if (type === 'approve') return 'submitting_approve'
  if (type === 'reject') return 'submitting_reject'
  return 'submitting_edit'
}

function statusMessage(status: ReviewUiStatus, run: AudienceReviewRun | null): string {
  const messages = {
    idle: 'Ready to start an analyst review.',
    starting: 'Starting analyst review…',
    pending_review: run?.current_review ? `Audience ${run.current_review.position} of ${run.current_review.total_reviews} is ready for review.` : 'A review is ready.',
    submitting_approve: 'Approving audience…',
    submitting_reject: 'Rejecting audience…',
    submitting_edit: 'Applying analyst edit…',
    editing: 'Applying analyst edit…',
    completed: 'Analyst review complete.',
    expired: 'Analyst review expired.',
    failed: 'Analyst review needs attention.',
    conflict: 'The review changed. Reconcile the authoritative run before continuing.',
  }
  return messages[status]
}

function toDisplayError(error: unknown): DisplayError {
  if (error instanceof AudienceReviewApiError) {
    return { code: error.code, message: friendlyErrorMessage(error.code, error.message) }
  }
  return { code: 'internal_error', message: 'The review could not be completed safely.' }
}

function friendlyErrorMessage(code: string, fallback: string): string {
  const messages: Record<string, string> = {
    review_run_not_found: 'This process-local review is no longer available. Start a new run.',
    review_run_expired: 'This review expired before the command was accepted.',
    review_version_conflict: 'Another action changed this review. Refresh to continue.',
    review_command_id_reused: 'This retry no longer matches the original command.',
    review_identity_conflict: 'The active review changed. Refresh to continue.',
    review_not_pending: 'This candidate is no longer pending review.',
    review_currently_editing: 'An analyst edit is already in progress.',
    review_edit_already_attempted: 'The one analyst edit has already been used.',
    network_error: 'The response was not received. Retry the same command or reconcile the run.',
    review_runtime_closed: 'The review service is restarting. Start a new run when it is available.',
  }
  return messages[code] ?? fallback
}

function errorTitle(code: string): string {
  return isConflictCode(code) ? 'Review conflict' : 'Review request failed'
}

function isConflictCode(code: string): boolean {
  return code.includes('conflict') || code === 'review_not_pending' || code === 'review_currently_editing' || code === 'review_edit_already_attempted' || code === 'review_command_id_reused'
}

function readStoredRunId(): string | null {
  try { return sessionStorage.getItem(REVIEW_RUN_STORAGE_KEY) } catch { return null }
}

function writeStoredRunId(runId: string) {
  try { sessionStorage.setItem(REVIEW_RUN_STORAGE_KEY, runId) } catch { /* Storage is optional. */ }
}

function clearStoredRunId() {
  try { sessionStorage.removeItem(REVIEW_RUN_STORAGE_KEY) } catch { /* Storage is optional. */ }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}
