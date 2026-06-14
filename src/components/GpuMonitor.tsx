import { useState, useEffect, useRef } from 'react'
import { Cpu, MemoryStick, Thermometer, Zap } from 'lucide-react'

// ── Types ──────────────────────────────────────────────────────────────────────
interface GpuSnapshot {
  ts: number
  gpus: Array<{
    index: number                // globally unique across clusters (used as history key)
    localIndex: number           // the GPU's own index on its node (0, 1, 2…)
    nodeIndex: number            // which node this GPU is on
    nodeHost: string             // IP/hostname of the node
    cluster: 'on-prem' | 'modal' // which Ray cluster this GPU belongs to
    name: string
    utilGpu: number    // 0-100 (raw nvidia-smi util.gpu)
    activityPct?: number // 0-100 (power-derived, catches data-loader phases)
    smPct?: number      // 0-100 (nvidia-smi dmon sm_pct — 1s averaged SM util)
    dmonMemPct?: number // 0-100 (dmon mem_pct — memory-controller util)
    powerW?: number     // current power draw in Watts
    tdpW?: number       // GPU's TDP — denominator for power %
    memUsedMiB: number
    memTotalMiB: number
    tempC: number
  }>
  ramUsedGb: number
  ramTotalGb: number
  nodesTotal: number
}

// ── Color palette per GPU (cycled) ────────────────────────────────────────────
const GPU_LINE_COLORS = [
  '#6366f1', // indigo
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#84cc16', // lime
]

// SM chart uses a separate warm-toned palette so SM lines can't be confused
// with the activity lines on the chart above.
const SM_LINE_COLORS = [
  '#f97316', // orange
  '#dc2626', // red-600
  '#eab308', // yellow
  '#db2777', // pink
  '#7c2d12', // dark orange
  '#fb923c', // light orange
  '#a16207', // dark amber
  '#facc15', // bright yellow
]

// ── Sparkline (used inside per-GPU cards) — real-time animated ────────────────
function Sparkline({
  data, color, maxVal = 100, height = 48, width = 160, nPoints = 40,
}: {
  data: number[]
  color: string
  maxVal?: number
  height?: number
  width?: number
  nPoints?: number
}) {
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const PAD = 2
  // Pad to a stable nPoints so CSS path transition animates smoothly
  // between updates instead of snapping.
  const padded: number[] = (() => {
    if (data.length >= nPoints) return data.slice(-nPoints)
    const seed = data[0] ?? 0
    return [...Array(nPoints - data.length).fill(seed), ...data]
  })()

  const xs = padded.map((_, i) => PAD + (i / (nPoints - 1)) * (width - 2 * PAD))
  const ys = padded.map(v => height - PAD - ((v / maxVal) * (height - 2 * PAD)))
  const lastX = xs[xs.length - 1]
  const lastY = ys[ys.length - 1]

  const line = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  const area = `${line} L${lastX.toFixed(1)},${(height - PAD).toFixed(1)} L${PAD},${(height - PAD).toFixed(1)} Z`
  const gradId = `gml-spark-${color.replace(/[^a-z0-9]/gi, '')}-${Math.random().toString(36).slice(2, 6)}`

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path className="gml-area" d={area} fill={`url(#${gradId})`} />
      <path className="gml-line" d={line} stroke={color} strokeWidth="1.5" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      {/* Expanding ring on the live dot — pulses every 1.6s */}
      <circle
        key={`spark-ring-${tick}`}
        className="gml-dot-ring"
        cx={lastX}
        cy={lastY}
        r="2.5"
        fill="none"
        stroke={color}
        strokeWidth="1.2"
        opacity="0.7"
        pointerEvents="none"
      />
      {/* Live dot — gentle pulse */}
      <circle
        key={`spark-dot-${tick}`}
        className="gml-dot-live"
        cx={lastX}
        cy={lastY}
        r="2.5"
        fill={color}
      />
    </svg>
  )
}

// ── Keyframes for live chart animations (injected once per component) ─────────
const LIVE_CHART_STYLES = `
@keyframes gml-pulse-dot {
  0%, 100% { r: 3; opacity: 1; }
  50%      { r: 5; opacity: 0.6; }
}
@keyframes gml-pulse-ring {
  0%   { r: 3;  opacity: 0.7; }
  100% { r: 10; opacity: 0; }
}
@keyframes gml-pulse-badge {
  0%, 100% { opacity: 1;   transform: scale(1); }
  50%      { opacity: 0.4; transform: scale(0.7); }
}
@keyframes gml-sweep {
  0%   { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
.gml-line  { transition: d 450ms cubic-bezier(0.4, 0, 0.2, 1); }
.gml-area  { transition: d 450ms cubic-bezier(0.4, 0, 0.2, 1), opacity 250ms ease; }
.gml-dot-live  { animation: gml-pulse-dot 1.6s ease-in-out infinite; }
.gml-dot-ring  { animation: gml-pulse-ring 1.6s ease-out infinite; transform-box: fill-box; transform-origin: center; }
.gml-badge-dot { animation: gml-pulse-badge 1.4s ease-in-out infinite; }
.gml-sweep-bar  {
  position: absolute; top: 0; left: 0; height: 100%; width: 30%;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%);
  animation: gml-sweep 2.2s linear infinite;
  pointer-events: none;
}
`

// ── Time-series chart with axes, gridlines, multi-line overlay, tooltip ───────
function TimeSeriesChart({
  series,                     // [{ key, label, color, data: number[] }]
  height = 180,
  windowSec = 60,             // width of the visible window
  maxVal = 100,
  yUnit = '%',
  showArea = true,            // show animated area fill under the first series
}: {
  series: Array<{ key: string; label: string; color: string; data: number[] }>
  height?: number
  windowSec?: number
  maxVal?: number
  yUnit?: string
  showArea?: boolean
}) {
  const [hover, setHover] = useState<{ x: number; idx: number } | null>(null)
  const [tick, setTick] = useState(0)  // forces re-render so pulse anim stays in sync
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)

  // Auto-size to container width
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(entries => {
      for (const e of entries) setWidth(Math.max(320, e.contentRect.width))
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  // Tick driver — re-renders every 1s so the rightmost-dot pulse animation
  // visibly pulses even when no new data has arrived (gives the chart a
  // "this is alive" feel independent of the underlying data rate).
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const PAD_L = 36
  const PAD_R = 12
  const PAD_T = 8
  const PAD_B = 22
  const innerW = width - PAD_L - PAD_R
  const innerH = height - PAD_T - PAD_B
  const n = Math.max(0, ...series.map(s => s.data.length))
  if (n < 2) {
    return <div ref={containerRef} style={{ height, background: 'var(--bg-elevated)', borderRadius: 8 }} />
  }

  // Always render exactly `windowSec` points (pad with the first known value
  // when we have less history). Keeping a stable point count means the SVG
  // path structure never changes, so CSS `transition: d` smoothly animates
  // between updates instead of snapping.
  const points = windowSec
  const xs = (i: number) => PAD_L + (i / (points - 1)) * innerW
  const ys = (v: number) => PAD_T + innerH - (Math.max(0, Math.min(maxVal, v)) / maxVal) * innerH

  // Y gridlines at 0/25/50/75/100
  const yTicks = [0, 25, 50, 75, 100]

  // X-axis time labels
  const xLabels = [0, Math.floor(points / 4), Math.floor(points / 2), Math.floor(3 * points / 4), points - 1]
    .filter((v, i, a) => a.indexOf(v) === i)
    .map(i => ({ i, label: i === points - 1 ? 'now' : `-${points - 1 - i}s` }))

  // Build the value-aligned array for a series. If the series has fewer than
  // `points` samples, left-pad with the first known value so the line starts
  // flat at the left edge.
  const padded = (data: number[]): number[] => {
    if (data.length >= points) return data.slice(-points)
    const pad = points - data.length
    const seed = data[0] ?? 0
    return [...Array(pad).fill(seed), ...data]
  }

  const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
    const x = e.clientX - rect.left
    if (x < PAD_L || x > PAD_L + innerW) { setHover(null); return }
    const ratio = (x - PAD_L) / innerW
    const idx = Math.round(ratio * (points - 1))
    setHover({ x: PAD_L + ratio * innerW, idx })
  }

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <style>{LIVE_CHART_STYLES}</style>
      <svg
        width={width}
        height={height}
        style={{ display: 'block', cursor: 'crosshair' }}
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
      >
        {/* Y gridlines + labels */}
        {yTicks.map(t => (
          <g key={t}>
            <line
              x1={PAD_L} y1={ys(t)} x2={width - PAD_R} y2={ys(t)}
              stroke="var(--border-default)" strokeWidth="1" strokeDasharray={t === 0 ? '0' : '2 4'} opacity={t === 0 ? 0.8 : 0.4}
            />
            <text
              x={PAD_L - 6} y={ys(t) + 3}
              fontSize="9" textAnchor="end"
              fill="var(--text-muted)" fontFamily="var(--font-mono)"
            >
              {t}{yUnit}
            </text>
          </g>
        ))}

        {/* X time labels */}
        {xLabels.map(({ i, label }) => (
          <text
            key={i}
            x={xs(i)} y={height - 6}
            fontSize="9" textAnchor="middle"
            fill="var(--text-muted)" fontFamily="var(--font-mono)"
          >
            {label}
          </text>
        ))}

        {/* Optional animated area fill under the first series — gives the
            chart a visible "breathing" effect when values change. */}
        {showArea && series[0] && (() => {
          const data = padded(series[0].data)
          const dArea = data.map((v, i) => `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ')
            + ` L${xs(data.length - 1).toFixed(1)},${(height - PAD_B).toFixed(1)}`
            + ` L${PAD_L.toFixed(1)},${(height - PAD_B).toFixed(1)} Z`
          const gradId = `gml-area-${series[0].key.replace(/[^a-z0-9]/gi, '')}`
          return (
            <>
              <defs>
                <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={series[0].color} stopOpacity="0.25" />
                  <stop offset="100%" stopColor={series[0].color} stopOpacity="0" />
                </linearGradient>
              </defs>
              <path className="gml-area" d={dArea} fill={`url(#${gradId})`} />
            </>
          )
        })()}

        {/* Series lines + pulsing live dot at the rightmost point */}
        {series.map(s => {
          const data = padded(s.data)
          const d = data.map((v, i) => `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ')
          const last = data[data.length - 1]
          return (
            <g key={s.key}>
              <path
                className="gml-line"
                d={d}
                stroke={s.color}
                strokeWidth="1.75"
                fill="none"
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={hover ? 0.55 : 0.95}
              />
              {/* Expanding ring on the live dot — gives a clear "this just
                  updated" pulse so the eye can pick out the current point. */}
              <circle
                className="gml-dot-ring"
                key={`ring-${tick}`}
                cx={xs(data.length - 1)}
                cy={ys(last)}
                r="3"
                fill="none"
                stroke={s.color}
                strokeWidth="1.5"
                opacity="0.7"
                pointerEvents="none"
              />
              <circle
                className="gml-dot-live"
                key={`dot-${tick}`}
                cx={xs(data.length - 1)}
                cy={ys(last)}
                r="3"
                fill={s.color}
                stroke="var(--bg-surface)"
                strokeWidth="1.5"
              />
            </g>
          )
        })}

        {/* Hover crosshair + dots */}
        {hover && (
          <g pointerEvents="none">
            <line
              x1={hover.x} y1={PAD_T} x2={hover.x} y2={height - PAD_B}
              stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" opacity="0.6"
            />
            {series.map(s => {
              const data = padded(s.data)
              const v = data[hover.idx]
              if (v === undefined) return null
              return (
                <circle key={s.key} cx={hover.x} cy={ys(v)} r="3.5" fill={s.color} stroke="var(--bg-surface)" strokeWidth="1.5" />
              )
            })}
          </g>
        )}
      </svg>

      {/* Subtle "data sweeping in" overlay — independent of data values,
          gives the whole chart a constant sense of motion. */}
      <div className="gml-sweep-bar" style={{ left: PAD_L, width: innerW }} />

      {/* Tooltip */}
      {hover && (
        <div style={{
          position: 'absolute', left: Math.min(hover.x + 8, width - 180), top: 4,
          background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
          borderRadius: 6, padding: '6px 10px', fontSize: 11, pointerEvents: 'none',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)', minWidth: 140, zIndex: 5,
        }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>
            t-{points - 1 - hover.idx}s
          </div>
          {series.map(s => {
            const data = padded(s.data)
            const v = data[hover.idx]
            if (v === undefined) return null
            return (
              <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 6, lineHeight: 1.6 }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: s.color, flexShrink: 0 }} />
                <span style={{ color: 'var(--text-secondary)', flex: 1 }}>{s.label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {Math.round(v)}{yUnit}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── GPU Card ──────────────────────────────────────────────────────────────────
function GpuCard({
  gpu, utilHistory, memHistory,
}: {
  gpu: GpuSnapshot['gpus'][number]
  utilHistory: number[]
  memHistory: number[]
}) {
  const memPct = gpu.memTotalMiB > 0 ? (gpu.memUsedMiB / gpu.memTotalMiB) * 100 : 0
  const memUsedGb = (gpu.memUsedMiB / 1024).toFixed(1)
  const memTotalGb = (gpu.memTotalMiB / 1024).toFixed(0)
  // Prefer activity_pct (power-derived) when the team's gpu-status feed is
  // available — it captures activity even when nvidia-smi's 1s sample lands
  // on an idle moment between kernels. Fall back to utilGpu otherwise.
  const displayPct = (gpu as any).activityPct ?? gpu.utilGpu
  const utilColor = displayPct > 80 ? 'var(--danger)' : displayPct > 50 ? 'var(--warning)' : 'var(--success)'
  const powerW = (gpu as any).powerW ?? gpu.powerW
  const tdpW = (gpu as any).tdpW
  const powerPct = tdpW && powerW ? Math.round((powerW / tdpW) * 100) : null

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-default)',
      borderRadius: 12,
      padding: '14px 16px',
      minWidth: 0,
    }}>
      {/* GPU name + node host + cluster badge + temp + power */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
            <span style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4,
              background: gpu.cluster === 'modal' ? '#8b5cf620' : '#f59e0b20',
              color:      gpu.cluster === 'modal' ? '#8b5cf6' : '#f59e0b',
              textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              {gpu.cluster}
            </span>
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              node {gpu.nodeIndex} · {gpu.nodeHost || '?'}
            </span>
          </div>
          <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            GPU {gpu.localIndex} · {gpu.name}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexShrink: 0 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 11, color: 'var(--text-muted)' }} title="Temperature">
            <Thermometer size={11} /> {gpu.tempC}°C
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 11, color: 'var(--text-muted)' }} title="Power draw">
            <Zap size={11} /> {gpu.powerW}W
          </span>
        </div>
      </div>

      {/* Utilization — nvidia-smi's util.gpu is a 1s rolling window; the
          sidecar samples 4× per poll and reports max(compute, mem-controller)
          so we capture peak activity even between kernel bursts. When the
          team's gpu-status endpoint is available, it also exposes a
          power-derived activity_pct that catches data-loader phases too. */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
            <Cpu size={11} /> GPU Activity
            {powerPct !== null && (
              <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400 }} title={`Power draw: ${powerW?.toFixed(0)}W / TDP ${tdpW}W`}>
                · {powerW?.toFixed(0)}W
              </span>
            )}
          </span>
          <span style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono)', color: utilColor }}>
            {displayPct}%
          </span>
        </div>
        {typeof gpu.smPct === 'number' && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: -2, marginBottom: 4, fontFamily: 'var(--font-mono)' }}>
            SM <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{gpu.smPct}%</span>
            {typeof gpu.dmonMemPct === 'number' && (
              <> · MEM-ctrl <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{gpu.dmonMemPct}%</span></>
            )}
          </div>
        )}
        <div style={{ position: 'relative', height: 48 }}>
          <Sparkline data={utilHistory} color={utilColor === 'var(--danger)' ? '#ef4444' : utilColor === 'var(--warning)' ? '#f59e0b' : '#10b981'} maxVal={100} height={48} width={280} />
        </div>
      </div>

      {/* VRAM */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
            <MemoryStick size={11} /> VRAM
          </span>
          {gpu.memTotalMiB > 0 ? (
            <span style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              {memUsedGb} / {memTotalGb} GB
              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 4 }}>({memPct.toFixed(0)}%)</span>
            </span>
          ) : (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>no metrics</span>
          )}
        </div>
        {gpu.memTotalMiB > 0 && (
          <>
            <div style={{ position: 'relative', height: 36 }}>
              <Sparkline data={memHistory} color="#6366f1" maxVal={gpu.memTotalMiB} height={36} width={280} />
            </div>
            {/* VRAM bar */}
            <div style={{ height: 4, borderRadius: 2, background: 'var(--bg-elevated)', marginTop: 6, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 2,
                width: `${memPct}%`,
                background: memPct > 85 ? 'var(--danger)' : memPct > 60 ? 'var(--warning)' : '#6366f1',
                transition: 'width 0.5s ease',
              }} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── RAM Card ──────────────────────────────────────────────────────────────────
function RamCard({ usedGb, totalGb, history }: { usedGb: number; totalGb: number; history: number[] }) {
  const pct = (usedGb / totalGb) * 100
  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 12, padding: '14px 16px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
          <MemoryStick size={11} /> System RAM
        </span>
        <span style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
          {usedGb.toFixed(0)} / {totalGb.toFixed(0)} GB
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 4 }}>({pct.toFixed(0)}%)</span>
        </span>
      </div>
      <Sparkline data={history} color="#8b5cf6" maxVal={totalGb} height={36} width={280} />
      <div style={{ height: 4, borderRadius: 2, background: 'var(--bg-elevated)', marginTop: 6, overflow: 'hidden' }}>
        <div style={{ height: '100%', borderRadius: 2, width: `${pct}%`, background: '#8b5cf6', transition: 'width 0.5s ease' }} />
      </div>
    </div>
  )
}

// ── Main Export ───────────────────────────────────────────────────────────────
export default function GpuMonitor({ active }: { active: boolean }) {
  const [latest, setLatest] = useState<GpuSnapshot | null>(null)
  const [utilHist, setUtilHist] = useState<Record<number, number[]>>({})
  const [smHist,   setSmHist]   = useState<Record<number, number[]>>({})
  const [memHist, setMemHist] = useState<Record<number, number[]>>({})
  const [ramHist, setRamHist] = useState<number[]>([])
  const [error, setError] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const MAX_POINTS = 60

  /**
   * Pulls the Ray /nodes?view=summary from `endpointUrl` and appends
   * its GPUs to `acc`. Tag each GPU with the source cluster ("on-prem"
   * or "modal") so the UI can label cards correctly. Returns the
   * number of nodes that were actually present.
   */
  const poll = async () => {
    try {
      const [gpuRes, csRes] = await Promise.all([
        fetch('/api/ray/gpu-stats'),
        fetch('/api/ray/api/cluster_status'),
      ])
      if (!gpuRes.ok) { setError(true); return }
      const gpuData = await gpuRes.json()
      const gpus: GpuSnapshot['gpus'] = (gpuData.nodes ?? []).flatMap((node: any, nIdx: number) => {
        const details: any[] = node.gpus_detail ?? []
        const count = details.length || (node.gpus_allocated ?? 0)
        return Array.from({ length: count }, (_, gi) => {
          const d = details[gi] as any | undefined
          // activity_pct comes from the team's gpu-status feed (power-derived)
          // and is the most reliable activity indicator. util_pct is the raw
          // nvidia-smi 1-second sample and can be 0 between kernels. The card
          // prefers activityPct when present.
          const rawUtil = d?.util_pct ?? (gpuData.cluster?.gpu_util_pct ?? 0)
          const activity = d?.activity_pct
          return {
            index:       nIdx * 100 + gi,
            localIndex:  d?.index ?? gi,
            nodeIndex:   nIdx,
            nodeHost:    node.hostname ?? node.ip ?? '',
            cluster:     'on-prem' as const,
            name:        d?.name ?? 'H200',
            utilGpu:     rawUtil,
            activityPct: activity,
            smPct:       d?.sm_pct,
            dmonMemPct:  d?.dmon_mem_pct,
            powerW:      d?.power_w ?? d?.power_watt ?? 0,
            tdpW:        d?.tdp_w,
            memUsedMiB:  d?.mem_used_mb ?? 0,
            memTotalMiB: d?.mem_total_mb ?? 80000,
            tempC:       d?.temp_c ?? 0,
          }
        })
      })

      let ramUsedGb = 0, ramTotalGb = 0, nodesTotal = (gpuData.nodes ?? []).length
      if (csRes.ok) {
        const cs = await csRes.json()
        const usage: Record<string, [number, number]> = cs?.data?.clusterStatus?.loadMetricsReport?.usage ?? {}
        ramTotalGb = (usage['memory']?.[1] ?? 0) / 1e9
        ramUsedGb  = (usage['memory']?.[0] ?? 0) / 1e9
        const usageByNode = cs?.data?.clusterStatus?.loadMetricsReport?.usageByNode
        if (usageByNode && typeof usageByNode === 'object') {
          nodesTotal = Object.keys(usageByNode).length || nodesTotal
        }
      }

      const gpuTotal = gpuData.gpus_total ?? 0
      const nodeCount = (gpuData.nodes ?? []).length || gpuTotal
      // Only show error if the API call genuinely failed — having zero
      // allocated GPUs is perfectly normal when no training jobs are running.
      if (gpus.length === 0 && gpuTotal === 0 && nodeCount === 0) { setError(true); return }
      setError(false)

      const snap: GpuSnapshot = { ts: Date.now(), gpus, ramUsedGb, ramTotalGb, nodesTotal }
      setLatest(snap)
      setUtilHist(prev => {
        const next = { ...prev }
        for (const g of gpus) {
          // Sparkline tracks the value the card actually displays.
          const v = (g as any).activityPct ?? g.utilGpu
          next[g.index] = [...(prev[g.index] ?? []), v].slice(-MAX_POINTS)
        }
        return next
      })
      setSmHist(prev => {
        const next = { ...prev }
        for (const g of gpus) {
          // smPct may be undefined if the upstream feed didn't include dmon
          // — fall back to the same activity value so the chart never has
          // gaps. The dmon feed's sm_pct is the canonical SM util signal.
          const v = (g as any).smPct ?? (g as any).activityPct ?? g.utilGpu
          next[g.index] = [...(prev[g.index] ?? []), v].slice(-MAX_POINTS)
        }
        return next
      })
      setMemHist(prev => {
        const next = { ...prev }
        for (const g of gpus) next[g.index] = [...(prev[g.index] ?? []), g.memUsedMiB].slice(-MAX_POINTS)
        return next
      })
      setRamHist(prev => [...prev, ramUsedGb].slice(-MAX_POINTS))
    } catch {
      setError(true)
    }
  }

  useEffect(() => {
    if (!active) {
      if (timerRef.current) clearInterval(timerRef.current)
      return
    }
    poll()
    timerRef.current = setInterval(poll, 2000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [active])

  if (error) {
    return (
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
        borderRadius: 12, padding: '16px 20px', fontSize: 12, color: 'var(--text-muted)',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <Cpu size={14} /> Ray cluster ไม่ตอบสนอง — ตรวจสอบการเชื่อมต่อที่หน้า Ray Cluster
      </div>
    )
  }

  if (!latest) {
    return (
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
        borderRadius: 12, padding: '16px 20px', fontSize: 12, color: 'var(--text-muted)',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <Cpu size={14} /> กำลังโหลด GPU monitor...
      </div>
    )
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--success)', boxShadow: '0 0 6px var(--success)', animation: 'pulse 2s infinite' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
          Live GPU / Memory &mdash; Ray Cluster
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
          อัพเดตทุก 2s &middot; {latest.nodesTotal} node{latest.nodesTotal === 1 ? '' : 's'} &middot; {latest.gpus.length} GPU
        </span>
      </div>

      {/* Cluster-wide time-series chart: one line per GPU + cluster aggregate */}
      {latest.gpus.length > 0 && (
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
          borderRadius: 12, padding: '14px 16px', marginBottom: 12,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
            <Cpu size={12} color="var(--text-muted)" />
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              GPU Activity — last 60s
            </span>
            {/* LIVE badge with pulsing dot — reassures user the chart is
                actually streaming, even when all GPUs are at a similar value
                and the lines look flat. */}
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              marginLeft: 6, padding: '1px 6px', borderRadius: 4,
              background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)',
              fontSize: 9, fontWeight: 700, color: 'var(--danger)', letterSpacing: '0.08em',
            }}>
              <span className="gml-badge-dot" style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--danger)' }} />
              LIVE
            </span>
            <div style={{ display: 'flex', gap: 10, marginLeft: 'auto', flexWrap: 'wrap' }}>
              {(() => {
                // Cluster aggregate line: average across GPUs at each time index.
                const len = Math.max(0, ...latest.gpus.map(g => (utilHist[g.index] ?? []).length))
                if (len < 2) return null
                const agg: number[] = []
                for (let i = 0; i < len; i++) {
                  let s = 0, c = 0
                  for (const g of latest.gpus) {
                    const h = utilHist[g.index] ?? []
                    if (h[i] !== undefined) { s += h[i]; c++ }
                  }
                  agg.push(c > 0 ? s / c : 0)
                }
                return (
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-muted)' }}>
                    <span style={{ width: 10, height: 2, background: 'var(--text-muted)', borderRadius: 1 }} />
                    <span style={{ fontFamily: 'var(--font-mono)' }}>avg {Math.round(agg[agg.length - 1] ?? 0)}%</span>
                  </span>
                )
              })()}
              {latest.gpus.map((g, i) => (
                <span key={g.index} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-muted)' }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: GPU_LINE_COLORS[i % GPU_LINE_COLORS.length] }} />
                  <span>GPU {g.localIndex}</span>
                </span>
              ))}
            </div>
          </div>
          <TimeSeriesChart
            height={180}
            windowSec={60}
            maxVal={100}
            series={(() => {
              const list = latest.gpus.map((g, i) => ({
                key: `gpu-${g.index}`,
                label: `GPU ${g.localIndex}`,
                color: GPU_LINE_COLORS[i % GPU_LINE_COLORS.length],
                data: utilHist[g.index] ?? [],
              }))
              // Add the cluster-average line on top, drawn last so it sits above
              // the per-GPU lines visually.
              const len = Math.max(0, ...latest.gpus.map(g => (utilHist[g.index] ?? []).length))
              if (len >= 2) {
                const agg: number[] = []
                for (let i = 0; i < len; i++) {
                  let s = 0, c = 0
                  for (const g of latest.gpus) {
                    const h = utilHist[g.index] ?? []
                    if (h[i] !== undefined) { s += h[i]; c++ }
                  }
                  agg.push(c > 0 ? s / c : 0)
                }
                list.push({ key: 'avg', label: 'cluster avg', color: '#94a3b8', data: agg })
              }
              return list
            })()}
          />
        </div>
      )}

      {/* SM Utilization chart — same per-GPU overlay, distinct palette so
          SM lines don't get confused with the activity lines above. Sourced
          from nvidia-smi dmon sm_pct (1s averaged, less spiky than
          util.gpu). Cluster average is the gold dashed line. */}
      {latest.gpus.length > 0 && (
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
          borderRadius: 12, padding: '14px 16px', marginBottom: 12,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
            <Cpu size={12} color="var(--text-muted)" />
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              SM Utilization — last 60s
            </span>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              marginLeft: 6, padding: '1px 6px', borderRadius: 4,
              background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)',
              fontSize: 9, fontWeight: 700, color: 'var(--danger)', letterSpacing: '0.08em',
            }}>
              <span className="gml-badge-dot" style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--danger)' }} />
              LIVE
            </span>
            <div style={{ display: 'flex', gap: 10, marginLeft: 'auto', flexWrap: 'wrap' }}>
              {(() => {
                const len = Math.max(0, ...latest.gpus.map(g => (smHist[g.index] ?? []).length))
                if (len < 2) return null
                const agg: number[] = []
                for (let i = 0; i < len; i++) {
                  let s = 0, c = 0
                  for (const g of latest.gpus) {
                    const h = smHist[g.index] ?? []
                    if (h[i] !== undefined) { s += h[i]; c++ }
                  }
                  agg.push(c > 0 ? s / c : 0)
                }
                return (
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-muted)' }}>
                    <span style={{ width: 10, height: 2, background: '#fbbf24', borderRadius: 1 }} />
                    <span style={{ fontFamily: 'var(--font-mono)' }}>avg {Math.round(agg[agg.length - 1] ?? 0)}%</span>
                  </span>
                )
              })()}
              {latest.gpus.map((g, i) => (
                <span key={g.index} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-muted)' }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: SM_LINE_COLORS[i % SM_LINE_COLORS.length] }} />
                  <span>GPU {g.localIndex}</span>
                </span>
              ))}
            </div>
          </div>
          <TimeSeriesChart
            height={140}
            windowSec={60}
            maxVal={100}
            showArea={false}
            series={(() => {
              const list = latest.gpus.map((g, i) => ({
                key: `sm-${g.index}`,
                label: `GPU ${g.localIndex}`,
                color: SM_LINE_COLORS[i % SM_LINE_COLORS.length],
                data: smHist[g.index] ?? [],
              }))
              const len = Math.max(0, ...latest.gpus.map(g => (smHist[g.index] ?? []).length))
              if (len >= 2) {
                const agg: number[] = []
                for (let i = 0; i < len; i++) {
                  let s = 0, c = 0
                  for (const g of latest.gpus) {
                    const h = smHist[g.index] ?? []
                    if (h[i] !== undefined) { s += h[i]; c++ }
                  }
                  agg.push(c > 0 ? s / c : 0)
                }
                list.push({ key: 'sm-avg', label: 'cluster avg', color: '#fbbf24', data: agg })
              }
              return list
            })()}
          />
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
        {latest.gpus.map(gpu => (
          <GpuCard
            key={gpu.index}
            gpu={gpu}
            utilHistory={utilHist[gpu.index] ?? []}
            memHistory={memHist[gpu.index] ?? []}
          />
        ))}
        <RamCard
          usedGb={latest.ramUsedGb}
          totalGb={latest.ramTotalGb}
          history={ramHist}
        />
      </div>
    </div>
  )
}
