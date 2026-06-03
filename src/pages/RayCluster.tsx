import { useState, useEffect, useCallback, useRef } from 'react'
import { Activity, Cpu, HardDrive, RefreshCw, ExternalLink, Server, Zap, Settings, X, Check, Plus, Copy, Terminal, Cloud, Eye, EyeOff, ChevronDown, ChevronUp } from 'lucide-react'

const DEFAULT_RAY_URL = 'http://100.68.53.118:8265'
const STORAGE_KEY     = 'ray_head_url'
const MAX_HISTORY     = 40  // 40 × 10 s ≈ 6.7 min
const SETTINGS_API    = '/api/settings'

// ──────────────── Persistent settings (backend SQLite → localStorage cache) ────

async function loadRayUrl(): Promise<string> {
  try {
    const res = await fetch(`${SETTINGS_API}/ray_head_url`)
    if (res.ok) {
      const d = await res.json()
      if (d?.value) {
        localStorage.setItem(STORAGE_KEY, d.value)
        return d.value
      }
    }
  } catch { /* backend down — fall through to cache */ }
  return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_RAY_URL
}

async function saveRayUrl(url: string): Promise<void> {
  localStorage.setItem(STORAGE_KEY, url)   // instant cache
  try {
    await fetch(`${SETTINGS_API}/ray_head_url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: url }),
    })
  } catch { /* offline — localStorage is the fallback */ }
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
  background: 'var(--bg-elevated)',
  color: 'var(--text-primary)',
  fontSize: 13, fontFamily: 'var(--font-mono)',
  outline: 'none',
}

function AddWorkerTab({ headUrl }: { headUrl: string }) {
  const headHost = (() => {
    try { return new URL(headUrl).hostname } catch { return headUrl.replace(/https?:\/\//, '').split(':')[0] }
  })()

  const [cpus, setCpus]       = useState('8')
  const [gpus, setGpus]       = useState('0')
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
        background: 'color-mix(in oklch, var(--primary) 8%, transparent)',
        border: '1px solid color-mix(in oklch, var(--primary) 20%, transparent)',
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
            disabled={parseFloat(gpus) === 0}
            style={{ ...inputStyle, opacity: parseFloat(gpus) === 0 ? 0.4 : 1 }}
          />
        </div>
      </div>
      {parseFloat(gpus) > 0 && parseInt(gpuUtil) < 100 && (
        <div style={{
          fontSize: 11, color: 'var(--text-muted)', marginBottom: 12,
          padding: '6px 10px', borderRadius: 6,
          background: 'color-mix(in oklch, var(--accent) 8%, transparent)',
          border: '1px solid color-mix(in oklch, var(--accent) 20%, transparent)',
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
              ? 'color-mix(in oklch, var(--success) 15%, transparent)'
              : 'color-mix(in oklch, var(--primary) 10%, transparent)',
            border: `1px solid ${copied ? 'color-mix(in oklch, var(--success) 30%, transparent)' : 'var(--border)'}`,
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

// ──────────────── Modal.com Cloud Deploy ─────────────────────

const MODAL_GPUS = [
  { id: 'cpu',      label: 'CPU Only',  vram: null, price: 0.06,  badge: 'Budget',    color: 'var(--text-muted)' },
  { id: 'T4',       label: 'T4',        vram: 16,   price: 0.59,  badge: 'Entry',     color: 'oklch(65% 0.14 160)' },
  { id: 'L4',       label: 'L4',        vram: 24,   price: 0.80,  badge: 'Efficient', color: 'oklch(65% 0.14 160)' },
  { id: 'A10G',     label: 'A10G',      vram: 24,   price: 1.10,  badge: 'Balanced',  color: 'var(--primary)' },
  { id: 'A100-40GB',label: 'A100 40G',  vram: 40,   price: 3.04,  badge: 'High End',  color: 'var(--accent)' },
  { id: 'A100-80GB',label: 'A100 80G',  vram: 80,   price: 4.20,  badge: 'High End',  color: 'var(--accent)' },
  { id: 'H100',     label: 'H100',      vram: 80,   price: 3.95,  badge: 'Top Tier',  color: 'oklch(65% 0.20 30)' },
] as const

type ModalGpuId = typeof MODAL_GPUS[number]['id']
type ModalStatus = 'idle' | 'deploying' | 'running' | 'stopping' | 'error'

interface ModalConfig {
  tokenId: string
  tokenSecret: string
  gpuType: ModalGpuId
  numWorkers: number
}

const MODAL_KEY = 'modal_deploy_config'

function loadModalConfig(): ModalConfig {
  try {
    return { tokenId: '', tokenSecret: '', gpuType: 'T4', numWorkers: 1, ...JSON.parse(localStorage.getItem(MODAL_KEY) ?? '{}') }
  } catch {
    return { tokenId: '', tokenSecret: '', gpuType: 'T4', numWorkers: 1 }
  }
}

const STATUS_LABEL: Record<ModalStatus, string> = {
  idle:      'Not deployed',
  deploying: 'Deploying to Modal…',
  running:   'Running on Modal',
  stopping:  'Stopping…',
  error:     'Deploy failed',
}
const STATUS_COLOR: Record<ModalStatus, string> = {
  idle:      'var(--text-muted)',
  deploying: 'var(--warning)',
  running:   'var(--success)',
  stopping:  'var(--warning)',
  error:     'var(--danger)',
}

function ModalDeployPanel() {
  const [cfg, setCfg] = useState<ModalConfig>(loadModalConfig)
  const [showSecret, setShowSecret] = useState(false)
  const [status, setStatus] = useState<ModalStatus>('idle')
  const [rayUrl, setRayUrl] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [errMsg, setErrMsg] = useState<string | null>(null)

  // Persist config (skip secrets for security — user re-enters on reload)
  useEffect(() => {
    localStorage.setItem(MODAL_KEY, JSON.stringify({ ...cfg, tokenSecret: '' }))
  }, [cfg])

  // Poll status when active
  useEffect(() => {
    if (status === 'idle') return
    const id = setInterval(async () => {
      try {
        const r = await fetch('/api/modal/status')
        if (!r.ok) return
        const d = await r.json()
        setStatus(d.status as ModalStatus)
        if (d.ray_url) setRayUrl(d.ray_url)
        if (d.logs?.length) setLogs(d.logs)
      } catch { /* ignore */ }
    }, 3000)
    return () => clearInterval(id)
  }, [status])

  // Sync status on mount
  useEffect(() => {
    fetch('/api/modal/status').then(r => r.ok ? r.json() : null).then(d => {
      if (!d) return
      setStatus(d.status as ModalStatus)
      if (d.ray_url) setRayUrl(d.ray_url)
      if (d.logs?.length) setLogs(d.logs)
    }).catch(() => {})
  }, [])

  const handleStart = async () => {
    setStatus('deploying')
    setErrMsg(null)
    setLogs([])
    setRayUrl(null)
    try {
      const r = await fetch('/api/modal/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token_id: cfg.tokenId,
          token_secret: cfg.tokenSecret,
          gpu_type: cfg.gpuType,
          num_workers: cfg.numWorkers,
        }),
      })
      const d = await r.json()
      if (!r.ok) {
        setStatus('error')
        setErrMsg(d.detail ?? 'Deploy failed')
      }
    } catch (e) {
      setStatus('error')
      setErrMsg((e as Error).message)
    }
  }

  const handleStop = async () => {
    setStatus('stopping')
    try {
      await fetch('/api/modal/stop', { method: 'POST' })
    } catch { /* ignore */ }
    setStatus('idle')
    setRayUrl(null)
  }

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '7px 10px', borderRadius: 7, fontSize: 12,
    background: 'var(--bg-base)', border: '1px solid var(--border)',
    color: 'var(--text-primary)', outline: 'none', boxSizing: 'border-box',
    fontFamily: 'var(--font-mono)',
  }

  const canStart = cfg.tokenId.trim().length > 0 && cfg.tokenSecret.trim().length > 0
  const isActive = status === 'running' || status === 'deploying' || status === 'stopping'
  const selectedGpu = MODAL_GPUS.find(g => g.id === cfg.gpuType)!

  return (
    <div className="card" style={{ marginTop: 16 }}>
      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <Cloud size={14} style={{ color: 'var(--primary)' }} />
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
          Modal.com Cloud Deploy
        </h3>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>— serverless GPU Ray cluster</span>
        <a
          href="https://modal.com/settings"
          target="_blank"
          rel="noopener noreferrer"
          style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: 4, textDecoration: 'none' }}
        >
          Get Token <ExternalLink size={10} />
        </a>
      </div>

      {/* ── Credentials ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 16 }}>
        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Token ID</label>
          <input
            style={inputStyle}
            type="text"
            value={cfg.tokenId}
            onChange={e => setCfg(c => ({ ...c, tokenId: e.target.value }))}
            placeholder="ak-xxxxxxxxxxxxxxxxxxxxxxxx"
            autoComplete="off"
          />
        </div>
        <div style={{ position: 'relative' }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Token Secret</label>
          <input
            style={{ ...inputStyle, paddingRight: 32 }}
            type={showSecret ? 'text' : 'password'}
            value={cfg.tokenSecret}
            onChange={e => setCfg(c => ({ ...c, tokenSecret: e.target.value }))}
            placeholder="as-xxxxxxxxxxxxxxxxxxxxxxxx"
            autoComplete="new-password"
          />
          <button
            onClick={() => setShowSecret(s => !s)}
            style={{ position: 'absolute', right: 8, top: 22, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 2 }}
          >
            {showSecret ? <EyeOff size={13} /> : <Eye size={13} />}
          </button>
        </div>
      </div>

      {/* ── GPU Card Selector ── */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 8 }}>Instance Type</label>
        <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
          {MODAL_GPUS.map(gpu => {
            const sel = cfg.gpuType === gpu.id
            return (
              <button
                key={gpu.id}
                onClick={() => setCfg(c => ({ ...c, gpuType: gpu.id as ModalGpuId }))}
                disabled={isActive}
                style={{
                  flexShrink: 0, padding: '8px 10px', borderRadius: 8, cursor: isActive ? 'default' : 'pointer',
                  border: `1.5px solid ${sel ? gpu.color : 'var(--border)'}`,
                  background: sel ? `color-mix(in oklch, ${gpu.color} 10%, var(--bg-elevated))` : 'var(--bg-elevated)',
                  textAlign: 'center', minWidth: 76,
                  transition: 'border-color 0.15s, background 0.15s',
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 700, color: sel ? gpu.color : 'var(--text-primary)', marginBottom: 3 }}>
                  {gpu.label}
                </div>
                {gpu.vram ? (
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{gpu.vram} GB VRAM</div>
                ) : (
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>CPU only</div>
                )}
                <div style={{ fontSize: 10, color: gpu.color, fontWeight: 600, marginTop: 2 }}>${gpu.price}/hr</div>
                <div style={{
                  fontSize: 9, marginTop: 3, padding: '1px 5px', borderRadius: 3, display: 'inline-block',
                  background: `color-mix(in oklch, ${gpu.color} 15%, transparent)`,
                  color: gpu.color, fontWeight: 600,
                }}>
                  {gpu.badge}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Workers + Action Row ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        {/* Workers counter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Workers:</span>
          <button
            className="btn btn-secondary btn-sm"
            disabled={isActive}
            onClick={() => setCfg(c => ({ ...c, numWorkers: Math.max(0, c.numWorkers - 1) }))}
            style={{ padding: '2px 8px', fontWeight: 700 }}
          >−</button>
          <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', minWidth: 22, textAlign: 'center' }}>
            {cfg.numWorkers}
          </span>
          <button
            className="btn btn-secondary btn-sm"
            disabled={isActive}
            onClick={() => setCfg(c => ({ ...c, numWorkers: Math.min(8, c.numWorkers + 1) }))}
            style={{ padding: '2px 8px', fontWeight: 700 }}
          >+</button>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            (~${((cfg.numWorkers + 1) * selectedGpu.price).toFixed(2)}/hr total)
          </span>
        </div>

        {/* Start / Stop */}
        {status === 'running' ? (
          <button
            className="btn btn-sm"
            onClick={handleStop}
            style={{ background: 'var(--danger)', color: '#fff', border: 'none', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <X size={12} /> Stop Cluster
          </button>
        ) : (
          <button
            className="btn btn-sm"
            onClick={handleStart}
            disabled={!canStart || status === 'deploying' || status === 'stopping'}
            style={{
              background: canStart ? 'var(--primary)' : undefined,
              color: canStart ? '#fff' : undefined,
              border: 'none', display: 'flex', alignItems: 'center', gap: 6,
              opacity: (!canStart || isActive) ? 0.55 : 1,
            }}
          >
            {status === 'deploying' ? (
              <><RefreshCw size={12} className="animate-spin" /> Deploying…</>
            ) : status === 'stopping' ? (
              <><RefreshCw size={12} className="animate-spin" /> Stopping…</>
            ) : (
              <><Cloud size={12} /> Start Cluster</>
            )}
          </button>
        )}

        {/* Ray Dashboard link when running */}
        {rayUrl && (
          <a
            href={rayUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <ExternalLink size={12} /> Ray Dashboard
          </a>
        )}
      </div>

      {/* ── Status Bar ── */}
      {status !== 'idle' && (
        <div style={{
          marginTop: 12, display: 'flex', alignItems: 'center', gap: 8,
          padding: '7px 12px', borderRadius: 7,
          background: `color-mix(in oklch, ${STATUS_COLOR[status]} 8%, transparent)`,
          border: `1px solid color-mix(in oklch, ${STATUS_COLOR[status]} 20%, transparent)`,
        }}>
          <div style={{
            width: 7, height: 7, borderRadius: '50%',
            background: STATUS_COLOR[status],
            boxShadow: status === 'running' ? `0 0 6px ${STATUS_COLOR[status]}` : undefined,
            flexShrink: 0,
            animation: (status === 'deploying' || status === 'stopping') ? 'pulse 1.4s infinite' : undefined,
          }} />
          <span style={{ fontSize: 12, color: STATUS_COLOR[status], fontWeight: 500, flex: 1 }}>
            {errMsg ?? STATUS_LABEL[status]}
            {rayUrl && status === 'running' && (
              <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {rayUrl.replace('https://', '')}
              </span>
            )}
          </span>
          {logs.length > 0 && (
            <button
              onClick={() => setShowLogs(s => !s)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 3, fontSize: 11 }}
            >
              Logs {showLogs ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
          )}
        </div>
      )}

      {/* ── Deploy Logs ── */}
      {showLogs && logs.length > 0 && (
        <pre style={{
          marginTop: 8, maxHeight: 160, overflowY: 'auto',
          padding: '8px 10px', borderRadius: 7, fontSize: 10,
          background: 'var(--bg-base)', border: '1px solid var(--border)',
          color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)',
          whiteSpace: 'pre-wrap', wordBreak: 'break-all',
        }}>
          {logs.join('\n')}
        </pre>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────

export default function RayCluster() {
  // Init synchronously from localStorage cache; async override from backend on mount
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

  // Load authoritative URL from backend on mount
  useEffect(() => {
    loadRayUrl().then(url => {
      if (url !== (localStorage.getItem(STORAGE_KEY) ?? DEFAULT_RAY_URL)) setRayUrl(url)
    })
  }, [])

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
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 14 }}>
                <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                  Nodes{' '}
                  <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: 12 }}>({nodes.length})</span>
                </h3>
                {aliveNodes === 1 && (
                  <span style={{
                    fontSize: 11,
                    padding: '2px 10px', borderRadius: 20,
                    background: 'color-mix(in oklch, var(--warning) 10%, transparent)',
                    border: '1px solid color-mix(in oklch, var(--warning) 25%, transparent)',
                    color: 'var(--warning)',
                  }}>
                    Head only · no workers connected
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {nodes.map((node) => {
                  const alive = node.raylet?.state === 'ALIVE'
                  const gpuCount = node.gpus?.length ?? 0
                  const head = isHead(node)
                  return (
                    <div key={node.raylet?.nodeId ?? node.ip} style={{
                      padding: '12px 14px', borderRadius: 8,
                      background: 'var(--bg-elevated)',
                      border: `1px solid ${head ? 'color-mix(in oklch, var(--primary) 30%, var(--border))' : 'var(--border)'}`,
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
                            ? 'color-mix(in oklch, var(--primary) 15%, transparent)'
                            : 'color-mix(in oklch, var(--accent) 10%, transparent)',
                          color: head ? 'var(--primary)' : 'var(--accent)',
                          border: `1px solid ${head
                            ? 'color-mix(in oklch, var(--primary) 30%, transparent)'
                            : 'color-mix(in oklch, var(--accent) 20%, transparent)'}`,
                          flexShrink: 0,
                        }}>
                          {head ? 'HEAD' : 'WORKER'}
                        </span>
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
                              {gpu.utilizationGpu != null && (
                                <span style={{
                                  marginLeft: 6,
                                  color: gpu.utilizationGpu > 80 ? 'var(--danger)' : gpu.utilizationGpu > 40 ? 'var(--warning)' : 'var(--success)',
                                  fontWeight: 600,
                                }}>
                                  {gpu.utilizationGpu}%
                                </span>
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

          {/* Modal.com Cloud Deploy */}
          <ModalDeployPanel />
        </>
      )}

      {/* Modal.com Cloud Deploy — always visible (shown below disconnected state too) */}
      {!isConnected && <ModalDeployPanel />}
    </div>
  )
}
