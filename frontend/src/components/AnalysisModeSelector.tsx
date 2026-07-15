export type AnalysisMode = 'standard' | 'review'

interface AnalysisModeSelectorProps {
  mode: AnalysisMode
  onChange: (mode: AnalysisMode) => void
}

export function AnalysisModeSelector({
  mode,
  onChange,
}: AnalysisModeSelectorProps) {
  return (
    <fieldset className="mode-selector">
      <legend>Analysis mode</legend>
      <label>
        <input
          type="radio"
          name="analysis-mode"
          value="standard"
          checked={mode === 'standard'}
          onChange={() => onChange('standard')}
        />
        <span>
          <strong>Standard Analysis</strong>
          <small>Automatic weekly intelligence with live progress</small>
        </span>
      </label>
      <label>
        <input
          type="radio"
          name="analysis-mode"
          value="review"
          checked={mode === 'review'}
          onChange={() => onChange('review')}
        />
        <span>
          <strong>Analyst Review</strong>
          <small>Approve, reject, or request one bounded edit</small>
        </span>
      </label>
    </fieldset>
  )
}
