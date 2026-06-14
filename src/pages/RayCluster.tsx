import { useState, useEffect, useCallback, useRef } from 'react'
import { Activity, Cpu, HardDrive, RefreshCw, ExternalLink, Server, Zap, Settings, X, Check, Plus, Copy, Terminal, Loader } from 'lucide-react'
import { getPrefOr, useUserPref } from '../lib/userPrefs'

const DEFAULT_RAY_URL = 'http://100.68.53.118:8265'
const PREF_KEY        = 'ray_head_url'
const MAX_HISTORY     = 40  // 40 × 10 s ≈ 6.7 min
const SETTINGS_API    = '/api/settings'

// ──────────────── Persistent settings (SQLite user_prefs) ────────────────
// Migrated from localStorage. The URL lives in the user_prefs table
// (scoped per Keycloak user). The in-memory mirror in userPrefs.ts is
// populated on mount so reads are sync.

async function loadRayUrl(): Promise<string> {
  // Backwards-compat: migrate from old global /api/settings to per-user
  // /api/user-prefs if the legacy row exists.
  try {
    const res = await fetch(`${SETTINGS_API}/ray_head_url`)
    if (res.ok) {
      const d = await res.json()
      if (d?.value) {
        // Save into per-user prefs and remove the global row
        try {
          await fetch(`/api/user-prefs/${PREF_KEY}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value: d.value }),
          })
        } catch { /* best-effort */ }
        return d.value
      }
    }
  } catch { /* ignore */ }
  return getPrefOr(PREF_KEY, DEFAULT_RAY_URL)
}

async function saveRayUrl(url: string): Promise<void> {
  // Persist to per-user prefs via the shared hook
  try {
    await fetch(`/api/user-prefs/${PREF_KEY}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: url }),
    })
  } catch { /* best-effort */ }
}

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
  isRealUtil?: boolean       // true if utilizationGpu comes from nvidia-smi (not allocation)
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
    isHeadNode?: boolean     // Ray API: true on head node
    resourcesTotal?: Record<string, number>     // { CPU: 80, GPU: 2, ... }
    resourcesAvailable?: Record<string, number> // free resources
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

// ──────────────── Add Worker tab ─────────────────────────────

const inputStyle: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box',
  padding: '7px 10px', borderRadius: 7,
  border: '1px solid var(--border)',
  background: 'var(--bg-surface, #fff)',
  color: 'var(--text-primary)',
  fontSize: 13, fontFamily: 'var(--font-mono)',
  outline: 'none',
  transition: 'border-color 0.15s',
}

function AddWorkerTab({ headUrl }: { headUrl: string }) {
  const headHost = (() => {
    try { return new URL(headUrl).hostname } catch { return headUrl.replace(/https?:\/\//, '').split(':')[0] }
  })()

  const [cpus, setCpus]       = useState('8')
  const [gpus, setGpus]       = useState('1')
  const [gpuUtil, setGpuUtil] = useState('100')   // % advertised GPU utilisation cap
  const [block, setBlock]     = useState(true)
  const [copied, setCopied]   = useState(false)

  const cpuFlag  = cpus.trim()  ? ` --num-cpus=${cpus.trim()}`  : ''
  // num-gpus accepts decimals: 0.5 = 50% of 1 GPU, 1 = full GPU
  const gpuNum   = parseFloat(gpus) || 0
  const utilPct  = Math.min(100, Math.max(1, parseInt(gpuUtil) || 100))
  const gpuEffective = gpuNum > 0 ? (gpuNum * utilPct / 100).toFixed(2).replace(/\.?0+$/, '') : '0'
  const gpuFlag  = gpuNum > 0 ? ` --num-gpus=${gpuEffective}` : ''
  const blockFlag = block ? ' --block' : ''
  const cmd = `ray start --address=${headHost}:6379${cpuFlag}${gpuFlag}${blockFlag}`

  const copy = async () => {
    await navigator.clipboard.writeText(cmd)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div>
      {/* Prerequisites note */}
      <div style={{
        padding: '8px 12px', borderRadius: 7, marginBottom: 16,
        background: 'rgba(188,208,189,0.08)',
        border: '1px solid rgba(188,208,189,0.20)',
        fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6,
      }}>
        Worker must have <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>ray</code> installed
        and network access to <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>{headHost}:6379</code> (GCS port).
      </div>

      {/* Resource inputs */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 14 }}>
        <div>
          <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', marginBottom: 5 }}>CPUs</label>
          <input
            type="number" min={1} max={256} value={cpus}
            onChange={e => setCpus(e.target.value)}
            style={inputStyle}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', marginBottom: 5 }}>GPUs</label>
          <input
            type="number" min={0} max={32} value={gpus}
            onChange={e => setGpus(e.target.value)}
            style={inputStyle}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', marginBottom: 5 }}>
            GPU Utilization %
            <span style={{ marginLeft: 4, color: 'var(--text-muted)', fontWeight: 400 }}>(1–100)</span>
          </label>
          <input
            type="number" min={1} max={100} value={gpuUtil}
            onChange={e => setGpuUtil(e.target.value)}
            style={{ ...inputStyle, cursor: 'text' }}
            onFocus={e => e.target.style.borderColor = 'var(--primary)'}
            onBlur={e => e.target.style.borderColor = 'var(--border)'}
          />
        </div>
      </div>
      {parseFloat(gpus) > 0 && parseInt(gpuUtil) < 100 && (
        <div style={{
          fontSize: 11, color: 'var(--text-muted)', marginBottom: 12,
          padding: '6px 10px', borderRadius: 6,
          background: 'rgba(188,208,189,0.08)',
          border: '1px solid rgba(188,208,189,0.20)',
        }}>
          <span style={{ color: 'var(--accent)', fontWeight: 500 }}>--num-gpus={gpuEffective}</span>
          {' '}→ Ray จะโฆษณา {gpuEffective} GPU ต่อ node
          {' '}&nbsp;·&nbsp; task ที่ใช้ <code style={{ fontFamily: 'var(--font-mono)' }}>num_gpus=0.5</code> รันพร้อมกันได้ {Math.floor(parseFloat(gpuEffective) / 0.5)} task
        </div>
      )}

      {/* --block toggle */}
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, cursor: 'pointer' }}>
        <input type="checkbox" checked={block} onChange={e => setBlock(e.target.checked)}
          style={{ accentColor: 'var(--primary)', width: 14, height: 14 }} />
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>--block</code>
          {' '}(keep process in foreground / systemd/screen)
        </span>
      </label>

      {/* Generated command */}
      <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
        <Terminal size={11} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
        Command to run on worker machine
      </label>
      <div style={{ position: 'relative' }}>
        <pre style={{
          margin: 0, padding: '10px 44px 10px 12px',
          borderRadius: 7, border: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
          fontSize: 12, fontFamily: 'var(--font-mono)',
          color: 'var(--text-primary)',
          whiteSpace: 'pre-wrap', wordBreak: 'break-all',
          lineHeight: 1.6,
        }}>{cmd}</pre>
        <button
          onClick={copy}
          style={{
            position: 'absolute', top: 8, right: 8,
            background: copied
              ? 'rgba(214,222,209,0.15)'
              : 'rgba(188,208,189,0.10)',
            border: `1px solid ${copied ? 'rgba(214,222,209,0.30)' : 'var(--border)'}`,
            borderRadius: 5, padding: '3px 8px',
            cursor: 'pointer', fontSize: 11,
            color: copied ? 'var(--success)' : 'var(--text-secondary)',
            display: 'flex', alignItems: 'center', gap: 4,
            transition: 'all 0.2s ease',
          }}
        >
          {copied ? <Check size={11} /> : <Copy size={11} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      {/* Step list */}
      <ol style={{ margin: '14px 0 0', padding: '0 0 0 18px', fontSize: 11, color: 'var(--text-muted)', lineHeight: 2 }}>
        <li>SSH into the worker machine</li>
        <li>Run the command above (as the same user that has <code style={{ fontFamily: 'var(--font-mono)' }}>ray</code> in PATH)</li>
        <li>Refresh this page — node appears in the list within ~15 s</li>
      </ol>
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
  const [tab, setTab]     = useState<'head' | 'worker'>('head')
  const [value, setValue] = useState(currentUrl)
  const [saving, setSaving] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  useEffect(() => { if (tab === 'head') inputRef.current?.focus() }, [tab])

  const handleSave = async () => {
    const trimmed = value.trim().replace(/\/$/, '')
    if (!trimmed) return
    setSaving(true)
    await saveRayUrl(trimmed)
    setSaving(false)
    onSave(trimmed)
  }

  const tabBtn = (active: boolean): React.CSSProperties => ({
    flex: 1, padding: '7px 0', fontSize: 12, fontWeight: active ? 600 : 400,
    background: active ? 'var(--bg-card)' : 'transparent',
    border: 'none', borderRadius: 6,
    color: active ? 'var(--text-primary)' : 'var(--text-muted)',
    cursor: 'pointer', transition: 'all 0.15s',
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
  })

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 50,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.55)',
    }} onClick={onClose}>
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 12, padding: 24, width: 520, maxWidth: '94vw',
        maxHeight: '90vh', overflowY: 'auto',
        boxShadow: '0 24px 60px rgba(0,0,0,0.5)',
      }} onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>Cluster Configuration</h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 4 }}>
            <X size={16} />
          </button>
        </div>

        {/* Tab bar */}
        <div style={{
          display: 'flex', gap: 4, padding: 4, borderRadius: 8, marginBottom: 20,
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        }}>
          <button style={tabBtn(tab === 'head')} onClick={() => setTab('head')}>
            <Settings size={12} /> Head URL
          </button>
          <button style={tabBtn(tab === 'worker')} onClick={() => setTab('worker')}>
            <Plus size={12} /> Add Worker Node
          </button>
        </div>

        {/* Tab: Head URL */}
        {tab === 'head' && (
          <>
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
              style={inputStyle}
            />
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
              Port 8265 (Ray Dashboard). Persisted to SQLite via backend; localStorage used as cache.
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 20 }}>
              <button className="btn btn-secondary btn-sm" onClick={onClose} disabled={saving}>Cancel</button>
              <button className="btn btn-primary btn-sm" onClick={handleSave} disabled={saving}
                style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                {saving ? <RefreshCw size={12} className="animate-spin" /> : <Check size={13} />}
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </>
        )}

        {/* Tab: Add Worker Node */}
        {tab === 'worker' && (
          <>
            <AddWorkerTab headUrl={currentUrl} />
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 20 }}>
              <button className="btn btn-secondary btn-sm" onClick={onClose}>Close</button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ──────────────── component ───────────────────────────────────
// ──────────────────────────────────────────────────────────────

export default function RayCluster() {
  // Init from the in-memory prefs mirror; mirror is populated on mount
  // by GET /api/user-prefs. The hook re-renders us when it loads.
  const [storedUrl, setStoredUrl] = useUserPref(PREF_KEY, DEFAULT_RAY_URL)
  const [rayUrl, setRayUrl] = useState<string>(storedUrl)
  const [result, setResult] = useState<RayClusterData | null>(null)
  const [nodes, setNodes] = useState<RayNodeSummary[]>([])
  const [gpuStatsData, setGpuStatsData] = useState<any>(null)
  const [version, setVersion] = useState<RayVersion | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [clearingDead, setClearingDead] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const historyRef = useRef<MetricPoint[]>([])
  const [history, setHistory] = useState<MetricPoint[]>([])

  // Sync rayUrl with the prefs mirror once the user-prefs load finishes,
  // and migrate any legacy global /api/settings value to per-user.
  useEffect(() => {
    if (storedUrl) setRayUrl(storedUrl)
    void loadRayUrl().then(url => {
      if (url && url !== rayUrl) setRayUrl(url)
    })
  }, [storedUrl])

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

      // The Ray Dashboard /api/cluster_status may wrap the payload in a
      // "data" key or return it flat. Normalise so we always have the
      // clusterStatus object directly.
      const clusterPayload = sData.data ?? sData
      if (clusterPayload) setResult(clusterPayload)

      // Fetch GPU stats once (used for both node merging and chart metric)
      let gpuStats: any = null
      try {
        const gRes = await fetch('/api/ray/gpu-stats')
        if (gRes.ok) { gpuStats = await gRes.json(); setGpuStatsData(gpuStats) }
      } catch { /* ignore */ }

      if (nData.data?.summary) {
        // Merge GPU stats into nodes
        const gpuNodes: any[] = gpuStats?.nodes ?? []
        for (const node of nData.data.summary) {
          // If Ray already reports per-node GPU data, keep it
          if (node.gpus && node.gpus.length > 0) continue

          const gn = gpuNodes.find((n: any) => n.hostname === node.hostname || n.ip === node.ip)
          const gpusFromResources = Math.round(node.raylet?.resourcesTotal?.['GPU'] ?? 0)
          const gpusAllocated = gn ? gn.gpus_allocated : 0
          const gpuCount = Math.max(gpusAllocated, gpusFromResources, 0)
          const gpusDetail: any[] = gn?.gpus_detail ?? []

          if (gpuCount > 0) {
            // If sidecar provided per-GPU detail, use it — but only take the GPUs
            // that belong to this node (NVLink clusters show ALL GPUs on every node).
            // Limit to the number Ray reports for this node (gpusAllocated).
            if (gpusDetail.length > 0) {
              const nodeGpus = gpusDetail.slice(0, gpuCount).map((g: any, i: number) => ({
                index: g.index ?? i,
                name: g.name ?? 'H200',
                utilizationGpu: g.util_pct ?? 0,
                memoryUsed: g.mem_used_mb ?? 0,
                memoryTotal: g.mem_total_mb ?? 80000,
                isRealUtil: true,
              }))
              node.gpus = nodeGpus
            } else {
              node.gpus = Array.from({ length: gpuCount }, (_, i) => ({
                index: i,
                name: 'H200',
                utilizationGpu: 0,
                memoryUsed: 0,
                memoryTotal: 80000,
                isRealUtil: false,
              }))
            }
          }
        }
        setNodes(nData.data.summary)
      }
      if (vRes.ok) setVersion(await vRes.json())

      // ── Compute metric point for the realtime graph ──
      // Walk through all possible nesting of the cluster status payload.
      const cs = clusterPayload?.clusterStatus ?? clusterPayload
      const usage: Record<string, [number, number]> = (cs as any)?.loadMetricsReport?.usage ?? {}
      const cpuPct  = (usage['CPU']?.[1] ?? 0) > 0    ? (usage['CPU'][0]    / usage['CPU'][1])    * 100 : 0
      const memPct   = (usage['memory']?.[1] ?? 0) > 0 ? (usage['memory'][0] / usage['memory'][1]) * 100 : 0
      let gpuUtil = 0
      if ((gpuStats?.cluster?.gpu_active_count ?? 0) > 0) {
        gpuUtil = gpuStats.cluster.gpu_util_pct ?? 0
        // Clear stale history that used allocation % (100%) instead of util %
        if (historyRef.current.some(p => p.gpu > 50 && Math.abs(p.gpu - gpuUtil) > 50)) {
          historyRef.current = []
          setHistory([])
        }
      }
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

  async function clearDeadNodes() {
    setClearingDead(true)
    try {
      await fetch('/api/ray/clear-dead', { method: 'POST' })
      await fetchData(true)
    } finally { setClearingDead(false) }
  }

  const handleSaveUrl = (url: string) => {
    setRayUrl(url)
    setStoredUrl(url)   // sync to user_prefs so other tabs/devices see it
    setShowSettings(false)
  }

  const cs: any = result?.clusterStatus ?? result
  const usage    = cs?.loadMetricsReport?.usage ?? {}
  const cpuUsed  = Math.round(usage['CPU']?.[0] ?? 0)
  const cpuTotal = Math.round(usage['CPU']?.[1] ?? 0)
  const gpuTotal = usage['GPU']?.[1] ?? 0
  const hasGpuUtil = (gpuStatsData?.cluster?.gpu_active_count ?? 0) > 0
  const gpuUtilPct = hasGpuUtil ? (gpuStatsData.cluster.gpu_util_pct || 0) : 0
  const memTotal = usage['memory']?.[1] ?? 0
  const memUsed  = usage['memory']?.[0] ?? 0
  const aliveNodes = nodes.filter(n => n.raylet?.state === 'ALIVE' && (n.ip || n.hostname)).length
  const isConnected = !error && result !== null
  const pythonVersion = extractPythonVersion(nodes)

  // Identify head node by isHeadNode flag or IP match against configured URL
  const headIp = (() => {
    try { return new URL(rayUrl).hostname } catch { return rayUrl.replace(/https?:\/\//, '').split(':')[0] }
  })()
  const isHead = (node: RayNodeSummary) =>
    node.raylet?.isHeadNode === true || node.ip === headIp

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
          ? 'rgba(214,222,209,0.12)'
          : 'rgba(255,180,201,0.12)',
        border: `1px solid ${isConnected
          ? 'rgba(214,222,209,0.25)'
          : 'rgba(255,180,201,0.25)'}`,
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
            <StatCard icon={<Zap size={14} />}        label="GPUs"   value={hasGpuUtil ? `${Math.round(gpuUtilPct)}%` : `${gpuTotal}`}  sub={hasGpuUtil ? "utilized" : "allocated to Ray"} accent />
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
                  { label: 'CPU alloc',     color: '#60a5fa' },
                  { label: hasGpuUtil ? 'GPU util' : 'GPU alloc', color: '#f97316' },
                  { label: 'Memory %',       color: '#a78bfa' },
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
                  { label: 'CPU', color: '#60a5fa',         data: history.map(p => p.cpu) },
                  { label: 'GPU', color: '#f97316',          data: history.map(p => p.gpu) },
                  { label: 'MEM', color: '#a78bfa',    data: history.map(p => p.mem) },
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
            {gpuTotal > 0 && (
              <div style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>GPU</span>
                  <span style={{ fontWeight: 500, color: 'var(--accent)' }}>
                    {hasGpuUtil ? `${Math.round(gpuUtilPct)}% utilized` : `${gpuTotal} allocated`}
                  </span>
                </div>
                <div className="progress-bar">
                  <div className="progress-bar-fill" style={{ width: `${hasGpuUtil ? Math.min(gpuUtilPct, 100) : 0}%`, background: 'var(--primary)' }} />
                </div>
              </div>
            )}
            <UtilBar label="Memory" used={parseFloat(gb(memUsed))} total={parseFloat(gb(memTotal))} unit=" GB" />
          </div>

          {/* Node list */}
          {nodes.length > 0 && (
            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
                  Nodes{' '}
                  <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: 12 }}>({aliveNodes} alive{nodes.length - aliveNodes > 0 ? ` · ${nodes.length - aliveNodes} dead` : ''})</span>
                </h3>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  {nodes.some(n => n.raylet?.state !== 'ALIVE') && (
                    <button
                      className="btn btn-sm"
                      style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, background: '#ef444415', color: '#ef4444', border: '1px solid #ef444430' }}
                      onClick={clearDeadNodes}
                      disabled={clearingDead}
                    >
                      {clearingDead ? <Loader size={11} className="animate-spin" /> : <X size={11} />}
                      Clear Dead
                    </button>
                  )}
                  {aliveNodes < 2 && (
                    <span style={{
                      fontSize: 11,
                      padding: '2px 10px', borderRadius: 20,
                      background: 'rgba(255,206,216,0.10)',
                      border: '1px solid rgba(255,206,216,0.25)',
                      color: 'var(--warning)',
                    }}>
                      Head only · no workers connected
                    </span>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {nodes
                  .filter(n => n.raylet?.state === 'ALIVE' && (n.ip || n.hostname))
                  .filter((n, i, arr) => {
                    const key = n.raylet?.nodeId || n.ip
                    return !key || arr.findIndex(x => (x.raylet?.nodeId || x.ip) === key) === i
                  })
                  .map((node) => {
                  const alive = true
                  const gpuCount = node.gpus?.length ?? 0
                  const head = isHead(node)
                  return (
                    <div key={node.raylet?.nodeId ?? node.ip} style={{
                      padding: '12px 14px', borderRadius: 8,
                      background: 'var(--bg-elevated)',
                      border: `1px solid ${head ? 'rgba(188,208,189,0.10)' : 'var(--border)'}`,
                    }}>
                      {/* Node header row */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: gpuCount > 0 ? 10 : 8 }}>
                        <div style={{
                          width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                          background: alive ? 'var(--success)' : 'var(--danger)',
                        }} />
                        {/* HEAD / WORKER badge */}
                        <span style={{
                          fontSize: 10, fontWeight: 600, padding: '1px 7px', borderRadius: 4,
                          letterSpacing: '0.04em',
                          background: head
                            ? 'rgba(188,208,189,0.15)'
                            : 'rgba(188,208,189,0.10)',
                          color: head ? 'var(--primary)' : 'var(--accent)',
                          border: `1px solid ${head
                            ? 'rgba(188,208,189,0.30)'
                            : 'rgba(188,208,189,0.20)'}`,
                          flexShrink: 0,
                        }}>
                          {head ? 'HEAD' : 'WORKER'}
                        </span>
                        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {node.hostname ?? <span style={{ color: 'var(--text-muted)' }}>unknown</span>}
                        </span>
                        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          {node.ip ?? (node.raylet?.nodeId ? node.raylet.nodeId.slice(0, 8) : '—')}
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
                              background: 'rgba(188,208,189,0.10)',
                              border: '1px solid rgba(188,208,189,0.20)',
                              fontFamily: 'var(--font-mono)',
                            }}>
                              {gpu.name}
                              {gpu.isRealUtil ? (
                                <span style={{ marginLeft: 4 }}>
                                  · {(gpu.memoryUsed ?? 0) > 1024 ? `${((gpu.memoryUsed ?? 0) / 1024).toFixed(0)}/${((gpu.memoryTotal ?? 80000) / 1024).toFixed(0)} GB` : `${gpu.memoryUsed ?? 0}/${gpu.memoryTotal ?? 80000} MB`}
                                  <span style={{
                                    marginLeft: 6,
                                    color: (gpu.utilizationGpu ?? 0) > 80 ? 'var(--danger)' : (gpu.utilizationGpu ?? 0) > 40 ? 'var(--warning)' : (gpu.utilizationGpu ?? 0) > 0 ? 'var(--success)' : 'var(--text-muted)',
                                    fontWeight: 600,
                                  }}>
                                    {(gpu.utilizationGpu ?? 0) > 0 ? `${Math.round(gpu.utilizationGpu ?? 0)}% util` : 'idle'}
                                  </span>
                                </span>
                              ) : (
                                <span style={{ marginLeft: 4, color: 'var(--text-muted)' }}>· reserved</span>
                              )}
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

          {/* Add Worker Node card */}
          <div className="card" style={{ marginTop: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <Plus size={14} style={{ color: 'var(--accent)' }} />
              <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>Add Worker Node</h3>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>— join this cluster via Ray</span>
            </div>
            <AddWorkerTab headUrl={rayUrl} />
          </div>
        </>
      )}

    </div>
  )
}
