const compactNumberFormatter = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
})

const exactNumberFormatter = new Intl.NumberFormat('en-US')

const shortDateFormatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  timeZone: 'UTC',
})

export function formatCompactNumber(value: number): string {
  return compactNumberFormatter.format(value)
}

export function formatExactNumber(value: number): string {
  return exactNumberFormatter.format(value)
}

export function formatPageviews(value: number): string {
  return `${formatCompactNumber(value)} pageviews`
}

export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatSizeIndex(value: number): string {
  return `${value.toFixed(1)}%`
}

export function formatDate(value: string): string {
  return shortDateFormatter.format(new Date(`${value}T00:00:00Z`))
}

export function humanizeCode(value: string): string {
  const normalized = value.replaceAll('_', ' ')
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}
