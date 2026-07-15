import { useCallback, useEffect, useId, useRef, useState } from 'react'

import { AskWikiPulsePanel } from './AskWikiPulsePanel.js'

interface AskWikiPulseDrawerProps {
  runId: string
  audienceCount: number
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

export function AskWikiPulseDrawer({
  runId,
  audienceCount,
}: AskWikiPulseDrawerProps) {
  const [isOpen, setIsOpen] = useState(false)
  const titleId = useId()
  const descriptionId = useId()
  const triggerRef = useRef<HTMLButtonElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const drawerRef = useRef<HTMLElement>(null)

  const closeDrawer = useCallback(() => setIsOpen(false), [])

  useEffect(() => {
    closeDrawer()
  }, [closeDrawer, runId])

  useEffect(() => {
    if (!isOpen) return

    const previousOverflow = document.body.style.overflow
    const trigger = triggerRef.current
    document.body.style.overflow = 'hidden'
    const focusFrame = requestAnimationFrame(() => {
      closeButtonRef.current?.focus()
    })

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeDrawer()
        return
      }
      if (event.key !== 'Tab') return

      const drawer = drawerRef.current
      if (!drawer) return
      const focusableElements = getFocusableElements(drawer)
      const firstElement = focusableElements[0]
      const lastElement = focusableElements.at(-1)
      if (!firstElement || !lastElement) {
        event.preventDefault()
        drawer.focus()
        return
      }

      const activeElement = document.activeElement
      if (event.shiftKey && (
        activeElement === firstElement || !drawer.contains(activeElement)
      )) {
        event.preventDefault()
        lastElement.focus()
      } else if (!event.shiftKey && activeElement === lastElement) {
        event.preventDefault()
        firstElement.focus()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => {
      cancelAnimationFrame(focusFrame)
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
      trigger?.focus()
    }
  }, [closeDrawer, isOpen])

  return (
    <>
      <button
        ref={triggerRef}
        className="ask-drawer-trigger"
        type="button"
        onClick={() => setIsOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        hidden={isOpen}
      >
        <span className="ask-drawer-trigger-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" focusable="false">
            <path d="M12 2.8c.5 4.6 2.7 6.8 7.3 7.3-4.6.5-6.8 2.7-7.3 7.3-.5-4.6-2.7-6.8-7.3-7.3C9.3 9.6 11.5 7.4 12 2.8Z" />
            <path d="M18.3 15.8c.2 1.8 1.1 2.7 2.9 2.9-1.8.2-2.7 1.1-2.9 2.9-.2-1.8-1.1-2.7-2.9-2.9 1.8-.2 2.7-1.1 2.9-2.9Z" />
          </svg>
        </span>
        <span>Ask WikiPulse</span>
      </button>

      {isOpen ? (
        <div className="ask-drawer-layer">
          <button
            className="ask-drawer-backdrop"
            type="button"
            tabIndex={-1}
            aria-hidden="true"
            onClick={closeDrawer}
          />
          <section
            ref={drawerRef}
            className="ask-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            aria-describedby={descriptionId}
            tabIndex={-1}
          >
            <header className="ask-drawer-header">
              <div>
                <p>Evidence copilot</p>
                <h2 id={titleId}>Ask WikiPulse</h2>
                <span id={descriptionId}>
                  Grounded in the current Analyst Review evidence
                </span>
              </div>
              <button
                ref={closeButtonRef}
                className="ask-drawer-close"
                type="button"
                onClick={closeDrawer}
                aria-label="Close Ask WikiPulse"
              >
                <span aria-hidden="true">×</span>
              </button>
            </header>
            <div className="ask-drawer-content">
              <AskWikiPulsePanel
                runId={runId}
                audienceCount={audienceCount}
                embedded
              />
            </div>
          </section>
        </div>
      ) : null}
    </>
  )
}

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter((element) => !element.hasAttribute('disabled'))
}
