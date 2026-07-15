import { useEffect, useRef, useState } from 'react'
import { startCountUpAnimation } from '../stitchEnhancements.js'

interface CountUpMetricProps {
  value: number
  format: (value: number) => string
  duration?: number
  integer?: boolean
}

export function CountUpMetric({
  value,
  format,
  duration = 600,
  integer = true,
}: CountUpMetricProps) {
  const reducedMotion = usePrefersReducedMotion()
  const [displayValue, setDisplayValue] = useState(value)
  const lastDisplayValue = useRef(0)

  useEffect(() => {
    const stopAnimation = startCountUpAnimation({
      from: lastDisplayValue.current,
      to: value,
      duration,
      integer,
      reducedMotion,
      onValue: (nextValue) => {
        lastDisplayValue.current = nextValue
        setDisplayValue(nextValue)
      },
      requestFrame: requestAnimationFrame,
      cancelFrame: cancelAnimationFrame,
    })

    return stopAnimation
  }, [duration, integer, reducedMotion, value])

  return (
    <>
      <span aria-hidden="true">{format(displayValue)}</span>
      <span className="visually-hidden">{format(value)}</span>
    </>
  )
}

function usePrefersReducedMotion(): boolean {
  const [reducedMotion, setReducedMotion] = useState(() => (
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  ))

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const updatePreference = () => setReducedMotion(mediaQuery.matches)
    updatePreference()
    mediaQuery.addEventListener('change', updatePreference)
    return () => mediaQuery.removeEventListener('change', updatePreference)
  }, [])

  return reducedMotion
}
