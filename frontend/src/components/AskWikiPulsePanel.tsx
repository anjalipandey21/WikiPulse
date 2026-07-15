import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'

import {
  askWikiPulse,
  AudienceReviewApiError,
} from '../api/audienceReview.js'
import type { AudienceQuestionResponse } from '../api/audienceReviewTypes'
import { normalizeAssistantQuestion } from '../reviewUi.js'

interface AskWikiPulsePanelProps {
  runId: string
  audienceCount: number
  embedded?: boolean
}

const MIN_QUESTION_LENGTH = 3
const MAX_QUESTION_LENGTH = 500

export function AskWikiPulsePanel({
  runId,
  audienceCount,
  embedded = false,
}: AskWikiPulsePanelProps) {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AudienceQuestionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const activeController = useRef<AbortController | null>(null)
  const normalizedQuestion = normalizeAssistantQuestion(question)
  const hasControlCharacter = /\p{C}/u.test(question)
  const questionIsValid = !hasControlCharacter && (
    normalizedQuestion.length >= MIN_QUESTION_LENGTH &&
    normalizedQuestion.length <= MAX_QUESTION_LENGTH
  )
  const fallbackSuggestions = useMemo(() => {
    const suggestions = [
      'Which published audience appears strongest for a premium consumer brand?',
      'What evidence supports the published audience recommendation?',
    ]
    if (audienceCount > 1) {
      suggestions.push('How do the published audiences differ in reach and commercial confidence?')
    }
    return suggestions
  }, [audienceCount])
  const suggestions = result?.suggested_follow_up_questions.length
    ? result.suggested_follow_up_questions
    : fallbackSuggestions

  useEffect(() => {
    setQuestion('')
    setResult(null)
    setError(null)
    setLoading(false)
    activeController.current?.abort()
    activeController.current = null
    return () => {
      activeController.current?.abort()
      activeController.current = null
    }
  }, [runId])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!questionIsValid || loading) return
    const controller = new AbortController()
    activeController.current?.abort()
    activeController.current = controller
    setLoading(true)
    setError(null)
    try {
      const response = await askWikiPulse(runId, normalizedQuestion, controller.signal)
      if (controller.signal.aborted) return
      setResult(response)
    } catch (caught) {
      if (controller.signal.aborted) return
      setError(
        caught instanceof AudienceReviewApiError
          ? assistantErrorMessage(caught.code)
          : 'Ask WikiPulse could not answer safely.',
      )
    } finally {
      if (activeController.current === controller) {
        activeController.current = null
        setLoading(false)
      }
    }
  }

  return (
    <section
      className={`ask-wikipulse${embedded ? ' ask-wikipulse-embedded' : ''}`}
      aria-labelledby={embedded ? undefined : 'ask-wikipulse-title'}
      aria-label={embedded ? 'Ask WikiPulse question and evidence' : undefined}
    >
      {!embedded ? (
        <div className="ask-wikipulse-heading">
          <div>
            <p className="eyebrow">Grounded in published evidence</p>
            <h2 id="ask-wikipulse-title">Ask WikiPulse</h2>
          </div>
          <span className="assistant-badge">One question at a time</span>
        </div>
      ) : null}
      <p className="assistant-intro">
        Ask about the published audiences. Answers use only the evidence shown in this review.
      </p>

      <div className="assistant-suggestions" aria-label="Suggested questions">
        {suggestions.slice(0, 3).map((suggestion) => (
          <button
            className="assistant-suggestion"
            type="button"
            key={suggestion}
            onClick={() => setQuestion(suggestion)}
            disabled={loading}
          >
            {suggestion}
          </button>
        ))}
      </div>

      <form className="assistant-form" onSubmit={submit}>
        <label htmlFor="ask-wikipulse-question">Question</label>
        <div className="assistant-input-row">
          <input
            id="ask-wikipulse-question"
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={MAX_QUESTION_LENGTH}
            placeholder="What evidence supports the strongest audience?"
            autoComplete="off"
            disabled={loading}
            aria-describedby="ask-wikipulse-help"
          />
          <button className="button button-primary" type="submit" disabled={!questionIsValid || loading}>
            {loading ? 'Asking…' : 'Ask'}
          </button>
        </div>
        <p id="ask-wikipulse-help" className="assistant-help">
          {normalizedQuestion.length}/{MAX_QUESTION_LENGTH} characters · No chat history is saved.
        </p>
      </form>

      <div className="assistant-live" aria-live="polite">
        {loading ? 'Finding an answer in the published evidence…' : null}
        {error ? <div className="notice notice-error" role="alert">{error}</div> : null}
        {result ? (
          <div className="assistant-answer">
            <p className="assistant-answer-status">
              {result.evidence_status === 'grounded' ? 'Grounded answer' : 'Insufficient evidence'}
            </p>
            <p>{result.answer}</p>
            {result.citations.length ? (
              <div className="assistant-citations">
                <h3>Sources</h3>
                <ul>
                  {result.citations.map((citation) => (
                    <li key={`${citation.audience_label}:${citation.article_title}`}>
                      {citation.article_url ? (
                        <a href={citation.article_url} target="_blank" rel="noreferrer noopener">
                          {citation.article_title}
                        </a>
                      ) : <strong>{citation.article_title}</strong>}
                      <span>{citation.audience_label} · {citation.relevance}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  )
}

function assistantErrorMessage(code: string): string {
  const messages: Record<string, string> = {
    review_run_not_found: 'This review run is no longer available. Start a new analysis to ask questions.',
    assistant_run_unavailable: 'This review run is not available for questions.',
    assistant_provider_failed: 'Ask WikiPulse could not answer from the evidence right now.',
    assistant_unavailable: 'Ask WikiPulse is temporarily unavailable.',
    invalid_assistant_question: 'Enter a question between 3 and 500 characters.',
    network_error: 'The answer could not be reached. Try the same question again.',
  }
  return messages[code] ?? 'Ask WikiPulse could not answer safely.'
}
