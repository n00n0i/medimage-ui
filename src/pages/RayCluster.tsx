import { useState, useEffect, useCallback, useRef } from 'react'
import { Activity, Cpu, HardDrive, RefreshCw, ExternalLink, Server, Zap, Settings, X, Check } from 'lucide-react'

const DEFAULT_RAY_URL = 'http://100.68.53.118:8265'
const STORAGE_KEY = 'ray_head_url'
const MAX_HISTORY = 40  // 40 × 10 s ≈ 6.7 min

// ──────────────── Ray Dashboard REST API types ────────────────

interface RayVersion {
  ray_version: string
  ray_commit: string
  session_name: string
}

interface RayGpu {
  index: number              // actual field name from API
  name: string
  utilizationGpu?: number    // 0–100 %
  memoryUsed?: number        // MB
  memoryTotal?: number       // MB
  temperatureC?: number
  powerMw?: number
}

interface RayNodeSummary {
  hostname: string
  ip: string
  cpu: number                // 0–100 %
  gpus?: RayGpu[]
  // [total_bytes, available_bytes, pct_used, shared_bytes]
  mem?: [number, number, number, number]
  raylet?: {
    nodeId: string
    state: string            // "ALIVE" | "DEAD"
    numWorkers: number
    resourcesTotal?: Record<string, number>  // { CPU: 80, GPU: 2, ... }
  }
}

interface RayClusterData {
  autoscalingStatus?: string | null
  clusterStatus?: {
    loadMetricsReport?: {
      usage?: Record<string, [number, number]>
    }
  }
}

interface MetricPoint {
  t: number    // ms timestamp
  cpu: number  // 0–100 cluster CPU allocation %
  gpu: number  // 0–100 avg GPU compute utilisation %
  mem: number  // 0–100 cluster memory allocation %
}

// ──────────────── helpers ─────────────────────────────────────

// Extract Python version from GCS cmdline path e.g. "/home/ray/anaconda3/lib/python3.10/..."
function extractPythonVersion(nodes: RayNodeSummary[]): string | null {
  for (const node of nodes) {
    const cmdline = (node as unknown as { gcs?: { cmdline?: string[] } }).gcs?.cmdline ?? []
    for (const arg of cmdline) {
      const m = arg.match(/\/python(\d+\.\d+)\//)
      if (m) return m[1]
    }
  }
  return null
}

const gb = (bytes: number) => (bytes / 1e9).toFixed(1)
const pct = (used: number, total: number) =>
  total > 0 ? Math.round((used / total) * 100) : 0

// ──────────────── Realtime chart ──────────────────────────────

interface ChartSeries {
  label: string
  color: string
  data: number[]  // 0–100 values
}

const CHART_H = 130
const CHART_PAD = { top: 10, right: 54, bottom: 22, left: 4 }

function smoothPath(pts: [number, number][]): string {
  if (pts.length < 2) return ''
  let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`
  for (let i = 1; i < pts.length; i++) {
    const p0 = pts[Math.max(0, i - 2)]
    const p1 = pts[i - 1]
    const p2 = pts[i]
    const p3 = pts[Math.min(pts.length - 1, i + 1)]
    const cp1x = p1[0] + (p2[0] - p0[0]) / 6
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6
    d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`
  }
  return d
}

function RealtimeChart({ series, timeLabels }: {
  series: ChartSeries[]
  timeLabels?: [string, string]
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [chartW, setChartW] = useState(580)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => setChartW(entries[0].contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const plotW = chartW - CHART_PAD.left - CHART_PAD.right
  const plotH = CHART_H - CHART_PAD.top - CHART_PAD.bottom

  const toXY = (data: number[], i: number): [number, number] => {
    const n = data.length
    const span = Math.max(n - 1, MAX_HISTORY - 1)
    const offset = MAX_HISTORY - n
    const x = CHART_PAD.left + ((i + offset) / span) * plotW
    const y = CHART_PAD.top + (1 - Math.min(data[i], 100) / 100) * plotH
    return [x, y]
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg width={chartW} height={CHART_H} style={{ display: 'block', overflow: 'visible' }}>
        <defs>
          {series.map(s => (
            <linearGradient key={s.label} id={`rg-${s.label}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity={0.22} />
              <stop offset="100%" stopColor={s.color} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>

        {/* Horizontal grid lines */}
        {[0, 25, 50, 75, 100].map(level => {
          const y = CHART_PAD.top + (1 - level / 100) * plotH
          return (
            <g key={level}>
              <line
                x1={CHART_PAD.left} y1={y}
                x2={chartW - CHART_PAD.right} y2={y}
                stroke="var(--border)" strokeOpacity={0.6} strokeWidth={1}
                strokeDasharray={level === 0 || level === 100 ? undefined : '3 5'}
              />
              <text x={chartW - CHART_PAD.right + 6} y={y + 3.5} fontSize={9} fill="var(--text-muted)" opacity={0.55}>
                {level}%
              </text>
            </g>
          )
        })}

        {/* Time labels */}
        {timeLabels && (
          <>
            <text x={CHART_PAD.left} y={CHART_H - 4} fontSize={9} fill="var(--text-muted)" opacity={0.45}>
              {timeLabels[0]}
            </text>
            <text x={chartW - CHART_PAD.right} y={CHART_H - 4} fontSize={9} fill="var(--text-muted)" opacity={0.45} textAnchor="end">
              {timeLabels[1]}
            </text>
          </>
        )}

        {/* Series */}
        {series.map(s => {
          if (s.data.length < 2) return null
          const pts = s.data.map((_, i) => toXY(s.data, i))
          const linePath = smoothPath(pts)
          const firstPt = pts[0]
          const lastPt = pts[pts.length - 1]
          const bottomY = CHART_PAD.top + plotH
          const areaD = `${linePath} L ${lastPt[0].toFixed(1)} ${bottomY} L ${firstPt[0].toFixed(1)} ${bottomY} Z`
          const curY = lastPt[1]
          const curVal = s.data[s.data.length - 1]
          const labelY = Math.max(CHART_PAD.top + 8, Math.min(CHART_H - CHART_PAD.bottom - 4, curY + 3.5))
          return (
            <g key={s.label}>
              <path d={areaD} fill={`url(#rg-${s.label})`} />
              <path d={linePath} fill="none" stroke={s.color} strokeWidth={1.5}
                strokeLinecap="round" strokeLinejoin="round" />
              <circle cx={lastPt[0]} cy={curY} r={2.5} fill={s.color} />
              <text x={chartW - CHART_PAD.right + 8} y={labelY}
                fontSize={10} fill={s.color} fontWeight={600}>
                {Math.round(curVal)}%
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function UtilBar({ label, used, total, unit = '' }: {
  label: string; used: number; total: number; unit?: string
}) {
  const p = pct(used, total)
  const barColor = p > 80 ? 'var(--danger)' : p > 60 ? 'var(--warning)' : 'var(--primary)'
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontWeight: 500, color: p > 80 ? 'var(--danger)' : 'var(--text-primary)' }}>
          {used}{unit} / {total}{unit}
          <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8 }}>{p}%</span>
        </span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${p}%`, background: barColor }} />
      </div>
    </div>
  )
}

function MiniBar({ label, pct: p }: { label: string; pct: number }) {
  const barColor = p > 80 ? 'var(--danger)' : p > 60 ? 'var(--warning)' : 'var(--primary)'
  return (
    <div style={{ flex: 1, minWidth: 80 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>
        <span>{label}</span>
        <span>{Math.round(Math.min(p, 100))}%</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: 'var(--border)' }}>
        <div style={{
          height: '100%', borderRadius: 2,
          width: `${Math.min(p, 100)}%`,
          background: barColor,
          transition: 'width 0.4s ease',
        }} />
      </div>
    </div>
  )
}

function StatCard({ icon, label, value, sub, accent = false }: {
  icon: React.ReactNode; label: string; value: string | number; sub: string; accent?: boolean
}) {
  return (
    <div className="stat-card">
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: accent ? 'var(--accent)' : 'var(--text-muted)', marginBottom: 6 }}>
        {icon}
        <span className="stat-label">{label}</span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  )
}

// ──────────────── Settings modal ─────────────────────────────

function SettingsModal({
  currentUrl, onSave, onClose,
}: {
  currentUrl: string
  onSave: (url: string) => void
  onClose: () => void
}) {
  const [value, setValue] = useState(currentUrl)
  const inputRef = useRef<HTMLInputElement>(null)
  useEffect(() => { inputRef.current?.focus() }, [])

  const handleSave = () => {
    const trimmed = value.trim().replace(/\/$/, '')
    if (trimmed) onSave(trimmed)
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 50,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.55)',
    }} onClick={onClose}>
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 12, padding: 24, width: 420, maxWidth: '90vw',
        boxShadow: '0 24px 60px rgba(0,0,0,0.5)',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>Ray Head Configuration</h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 4 }}>
            <X size={16} />
          </button>
        </div>

        <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
          Ray Dashboard URL
        </label>
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleSave(); if (e.key === 'Escape') onClose() }}
          placeholder="http://100.68.x.x:8265"
          style={{
            width: '100%', boxSizing: 'border-box',
            padding: '8px 12px', borderRadius: 7,
            border: '1px solid var(--border)',
            background: 'var(--bg-elevated)',
            color: 'var(--text-primary)',
            fontSize: 13, fontFamily: 'var(--font-mono)',
            outline: 'none',
          }}
        />
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
          Port 8265 (Ray Dashboard). Saved to localStorage.
        </p>

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 20 }}>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={handleSave}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Check size={13} /> Save
          </button>
        </div>
      </div>
    </div>
  )
}

// ──────────────── component ───────────────────────────────────

export default function RayCluster() {
  const [rayUrl, setRayUrl] = useState<string>(
    () => localStorage.getItem(STORAGE_KEY) ?? DEFAULT_RAY_URL
  )
  const [result, setResult] = useState<RayClusterData | null>(null)
  const [nodes, setNodes] = useState<RayNodeSummary[]>([])
  const [version, setVersion] = useState<RayVersion | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const historyRef = useRef<MetricPoint[]>([])
  const [history, setHistory] = useState<MetricPoint[]>([])

  const fetchData = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true)
    try {
      const [sRes, nRes, vRes] = await Promise.all([
        fetch(`/api/ray/api/cluster_status`),
        fetch(`/api/ray/nodes?view=summary`),
        fetch(`/api/ray/api/version`),
      ])
      if (!sRes.ok) throw new Error(`cluster_status ${sRes.status}`)
      if (!nRes.ok) throw new Error(`nodes ${nRes.status}`)
      const sData = await sRes.json()
      const nData = await nRes.json()
      if (sData.data) setResult(sData.data)
      if (nData.data?.summary) setNodes(nData.data.summary)
      if (vRes.ok) setVersion(await vRes.json())
      // Record metric point for realtime graph
      const freshUsage: Record<string, [number, number]> = sData.data?.clusterStatus?.loadMetricsReport?.usage ?? {}
      const cpuPct  = freshUsage['CPU']?.[1]    ? (freshUsage['CPU'][0]    / freshUsage['CPU'][1])    * 100 : 0
      const gpuAlloc = freshUsage['GPU']?.[1]   ? (freshUsage['GPU'][0]    / freshUsage['GPU'][1])    * 100 : 0
      const memPct   = freshUsage['memory']?.[1] ? (freshUsage['memory'][0] / freshUsage['memory'][1]) * 100 : 0
      const summary: RayNodeSummary[] = nData.data?.summary ?? []
      const allGpuUtils: number[] = summary.flatMap(n => n.gpus?.map(g => g.utilizationGpu ?? 0) ?? [])
      const gpuUtil = allGpuUtils.length > 0
        ? allGpuUtils.reduce((a, b) => a + b, 0) / allGpuUtils.length
        : gpuAlloc
      const point: MetricPoint = { t: Date.now(), cpu: cpuPct, gpu: gpuUtil, mem: memPct }
      const next = [...historyRef.current, point].slice(-MAX_HISTORY)
      historyRef.current = next
      setHistory([...next])
      setError(null)
      setLastUpdated(new Date())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  // Re-fetch when rayUrl changes
  useEffect(() => {
    setLoading(true)
    setResult(null)
    setVersion(null)
    setError(null)
    fetchData()
    const id = setInterval(() => fetchData(), 10_000)
    return () => clearInterval(id)
  }, [fetchData, rayUrl])

  const handleSaveUrl = (url: string) => {
    localStorage.setItem(STORAGE_KEY, url)
    setRayUrl(url)
    setShowSettings(false)
  }

  const usage    = result?.clusterStatus?.loadMetricsReport?.usage ?? {}
  const cpuUsed  = Math.round(usage['CPU']?.[0] ?? 0)
  const cpuTotal = Math.round(usage['CPU']?.[1] ?? 0)
  const gpuUsed  = usage['GPU']?.[0] ?? 0
  const gpuTotal = usage['GPU']?.[1] ?? 0
  const memTotal = usage['memory']?.[1] ?? 0
  const memUsed  = usage['memory']?.[0] ?? 0
  const aliveNodes = nodes.filter(n => n.raylet?.state === 'ALIVE').length
  const isConnected = !error && result !== null
  const pythonVersion = extractPythonVersion(nodes)

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 240 }}>
      <RefreshCw className="animate-spin" size={22} style={{ color: 'var(--primary)' }} />
    </div>
  )

  return (
    <div style={{ maxWidth: 900 }}>
      {showSettings && (
        <SettingsModal
          currentUrl={rayUrl}
          onSave={handleSaveUrl}
          onClose={() => setShowSettings(false)}
        />
      )}

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>Ray Cluster</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>Real-time resource utilisation across Ray head and worker nodes.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {lastUpdated && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => fetchData(true)}
            disabled={refreshing}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setShowSettings(true)}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <Settings size={13} />
            Configure
          </button>
          <a href={rayUrl} target="_blank" rel="noopener noreferrer">
            <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ExternalLink size={13} />
              Ray Dashboard
            </button>
          </a>
        </div>
      </div>

      {/* Version + head info bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
        padding: '8px 14px', borderRadius: 8, marginBottom: 12,
        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        fontSize: 12,
      }}>
        <span style={{ color: 'var(--text-muted)' }}>
          Head: <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{rayUrl}</span>
        </span>
        {version && (
          <>
            <span style={{ color: 'var(--border)' }}>·</span>
            <span style={{ color: 'var(--text-muted)' }}>
              Ray <span style={{ color: 'var(--primary)', fontWeight: 500 }}>{version.ray_version}</span>
            </span>
            {pythonVersion && (
              <>
                <span style={{ color: 'var(--border)' }}>·</span>
                <span style={{ color: 'var(--text-muted)' }}>
                  Python <span style={{ color: 'var(--primary)', fontWeight: 500 }}>{pythonVersion}</span>
                </span>
              </>
            )}
            <span style={{ color: 'var(--border)' }}>·</span>
            <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {version.session_name.replace('session_', '')}
            </span>
          </>
        )}
      </div>

      {/* Connection status banner */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '8px 14px', borderRadius: 8, marginBottom: 20,
        background: isConnected
          ? 'color-mix(in oklch, var(--success) 12%, transparent)'
          : 'color-mix(in oklch, var(--danger) 12%, transparent)',
        border: `1px solid ${isConnected
          ? 'color-mix(in oklch, var(--success) 25%, transparent)'
          : 'color-mix(in oklch, var(--danger) 25%, transparent)'}`,
      }}>
        <div style={{
          width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
          background: isConnected ? 'var(--success)' : 'var(--danger)',
          boxShadow: isConnected ? '0 0 6px var(--success)' : '0 0 6px var(--danger)',
        }} />
        <span style={{ fontSize: 13, fontWeight: 500, color: isConnected ? 'var(--success)' : 'var(--danger)' }}>
          {isConnected
            ? `Connected · ${aliveNodes} node${aliveNodes !== 1 ? 's' : ''} alive`
            : `Unreachable · ${error}`}
        </span>
      </div>

      {/* Disconnected empty state */}
      {!isConnected ? (
        <div className="card" style={{ textAlign: 'center', padding: '48px 0' }}>
          <Activity style={{ margin: '0 auto 16px', color: 'var(--text-muted)' }} size={36} />
          <h3 style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>Cluster unreachable</h3>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 340, margin: '0 auto 16px' }}>
            Cannot connect to Ray Dashboard at <code style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{rayUrl}</code>.
            Check network connectivity or VPN.
          </p>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
            <button className="btn btn-secondary btn-sm" onClick={() => fetchData(true)}>Try again</button>
            <button className="btn btn-secondary btn-sm" onClick={() => setShowSettings(true)}
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Settings size={13} /> Change URL
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* Stat cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
            <StatCard icon={<Cpu size={14} />}       label="CPUs"   value={`${cpuUsed} / ${cpuTotal}`}  sub="used / total" />
            <StatCard icon={<Zap size={14} />}        label="GPUs"   value={`${gpuUsed} / ${gpuTotal}`}  sub="used / total" accent />
            <StatCard icon={<HardDrive size={14} />}  label="Memory" value={`${gb(memUsed)} GB`}          sub={`of ${gb(memTotal)} GB`} />
            <StatCard icon={<Server size={14} />}     label="Nodes"  value={`${aliveNodes} / ${nodes.length}`} sub="alive / total" />
          </div>

          {/* Realtime usage graph */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
                Resource Usage
                <span style={{ marginLeft: 8, fontSize: 10, fontWeight: 400, color: 'var(--text-muted)' }}>live · 10 s interval</span>
              </h3>
              <div style={{ display: 'flex', gap: 14 }}>
                {[
                  { label: 'CPU alloc',   color: 'var(--primary)' },
                  { label: 'GPU compute', color: 'var(--accent)'  },
                  { label: 'Memory',      color: 'oklch(65% 0.14 160)' },
                ].map(({ label, color }) => (
                  <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
                    <span style={{ width: 14, height: 2, borderRadius: 1, background: color, display: 'inline-block', flexShrink: 0 }} />
                    {label}
                  </span>
                ))}
              </div>
            </div>
            {history.length < 2 ? (
              <div style={{ height: CHART_H, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Collecting data… next update in ~10 s</span>
              </div>
            ) : (
              <RealtimeChart
                series={[
                  { label: 'CPU', color: 'var(--primary)',         data: history.map(p => p.cpu) },
                  { label: 'GPU', color: 'var(--accent)',          data: history.map(p => p.gpu) },
                  { label: 'MEM', color: 'oklch(65% 0.14 160)',    data: history.map(p => p.mem) },
                ]}
                timeLabels={[
                  new Date(history[0].t).toLocaleTimeString(),
                  new Date(history[history.length - 1].t).toLocaleTimeString(),
                ]}
              />
            )}
          </div>

          {/* Cluster utilisation bars */}
          <div className="card" style={{ marginBottom: 16 }}>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 16 }}>Cluster Utilisation</h3>
            <UtilBar label="CPU" used={cpuUsed} total={cpuTotal} />
            {gpuTotal > 0 && <UtilBar label="GPU" used={gpuUsed} total={gpuTotal} />}
            <UtilBar label="Memory" used={parseFloat(gb(memUsed))} total={parseFloat(gb(memTotal))} unit=" GB" />
          </div>

          {/* Node list */}
          {nodes.length > 0 && (
            <div className="card">
              <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 14 }}>
                Nodes{' '}
                <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: 12 }}>({nodes.length})</span>
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {nodes.map((node) => {
                  const alive = node.raylet?.state === 'ALIVE'
                  const gpuCount = node.gpus?.length ?? 0
                  return (
                    <div key={node.raylet?.nodeId ?? node.ip} style={{
                      padding: '12px 14px', borderRadius: 8,
                      background: 'var(--bg-elevated)',
                      border: '1px solid var(--border)',
                    }}>
                      {/* Node header row */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: gpuCount > 0 ? 10 : 8 }}>
                        <div style={{
                          width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                          background: alive ? 'var(--success)' : 'var(--danger)',
                        }} />
                        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {node.hostname}
                        </span>
                        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          {node.ip}
                        </span>
                        <div style={{ marginLeft: 'auto', display: 'flex', gap: 14, fontSize: 12, color: 'var(--text-muted)' }}>
                          {node.raylet && (
                            <>
                              <span>{Math.round(node.raylet.resourcesTotal?.['CPU'] ?? 0)} CPUs</span>
                              {(node.raylet.resourcesTotal?.['GPU'] ?? 0) > 0 && (
                                <span>{Math.round(node.raylet.resourcesTotal!['GPU'])} GPUs</span>
                              )}
                              <span>{node.raylet.numWorkers} workers</span>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Mini utilisation bars */}
                      <div style={{ display: 'flex', gap: 16 }}>
                        <MiniBar label="CPU" pct={node.cpu} />
                        {/* mem[2] = pct used (e.g. 10.0 = 10%) */}
                        {(node.mem?.[2] ?? 0) > 0 && <MiniBar label="MEM" pct={node.mem![2]} />}
                        {node.gpus?.map(gpu => (
                          <MiniBar key={gpu.index} label={`GPU${gpu.index}`} pct={gpu.utilizationGpu ?? 0} />
                        ))}
                      </div>

                      {/* GPU name badges */}
                      {gpuCount > 0 && (
                        <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                          {node.gpus!.map(gpu => (
                            <span key={gpu.index} style={{
                              fontSize: 11, color: 'var(--accent)', padding: '2px 8px',
                              borderRadius: 4,
                              background: 'color-mix(in oklch, var(--accent) 10%, transparent)',
                              border: '1px solid color-mix(in oklch, var(--accent) 20%, transparent)',
                              fontFamily: 'var(--font-mono)',
                            }}>
                              {gpu.name} · {gpu.memoryUsed ?? 0}/{gpu.memoryTotal ?? 0} MB
                              {gpu.temperatureC != null && ` · ${gpu.temperatureC}°C`}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

