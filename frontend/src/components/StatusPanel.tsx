import type { ReactNode } from 'react'

interface StatusPanelProps {
  children?: ReactNode
  eyebrow: string
  title: string
  message: string
  icon?: ReactNode
  actionLabel?: string
  onAction?: () => void
  busy?: boolean
}

export function StatusPanel({
  children,
  eyebrow,
  title,
  message,
  icon,
  actionLabel,
  onAction,
  busy = false,
}: StatusPanelProps) {
  return (
    <section
      className="status-panel"
      aria-live="polite"
      aria-busy={busy}
    >
      {icon ? <div className="status-icon" aria-hidden="true">{icon}</div> : null}
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{message}</p>
      {children ? <div className="status-panel-detail">{children}</div> : null}
      {actionLabel && onAction ? (
        <button
          className="button button-primary"
          type="button"
          onClick={onAction}
          disabled={busy}
        >
          {actionLabel}
        </button>
      ) : null}
    </section>
  )
}
