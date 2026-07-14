import { formatDate, formatExactNumber } from '../formatters'

interface PageviewSparklineProps {
  dailyViews: Record<string, number>
  articleTitle: string
}

const WIDTH = 156
const HEIGHT = 44
const PADDING = 3

export function PageviewSparkline({
  dailyViews,
  articleTitle,
}: PageviewSparklineProps) {
  const entries = Object.entries(dailyViews).sort(([left], [right]) =>
    left.localeCompare(right),
  )

  if (entries.length < 2) {
    return <span className="sparkline-empty">Daily trend unavailable</span>
  }

  const values = entries.map(([, value]) => value)
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const range = Math.max(maximum - minimum, 1)
  const drawableWidth = WIDTH - PADDING * 2
  const drawableHeight = HEIGHT - PADDING * 2
  const points = values
    .map((value, index) => {
      const x = PADDING + (index / (values.length - 1)) * drawableWidth
      const y = PADDING + ((maximum - value) / range) * drawableHeight
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  const firstDate = formatDate(entries[0][0])
  const lastDate = formatDate(entries.at(-1)?.[0] ?? entries[0][0])
  const accessibleLabel = `${articleTitle} daily pageviews from ${firstDate} to ${lastDate}; minimum ${formatExactNumber(minimum)}, maximum ${formatExactNumber(maximum)}.`

  return (
    <svg
      className="sparkline"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label={accessibleLabel}
      preserveAspectRatio="none"
    >
      <line
        className="sparkline-baseline"
        x1={PADDING}
        x2={WIDTH - PADDING}
        y1={HEIGHT - PADDING}
        y2={HEIGHT - PADDING}
      />
      <polyline className="sparkline-line" points={points} />
      {points.split(' ').map((point, index) => {
        const [cx, cy] = point.split(',')
        return (
          <circle
            className="sparkline-point"
            cx={cx}
            cy={cy}
            r="1.8"
            key={entries[index][0]}
          />
        )
      })}
    </svg>
  )
}
