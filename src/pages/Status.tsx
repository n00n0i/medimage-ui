import { useState, useEffect, useCallback } from 'react'
import {
  Server, HardDrive, Cpu, BookOpen, Tag, RefreshCw,
  CheckCircle, XCircle, ExternalLink, Layers, Activity, Zap,
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────

type SvcStatus = 'ok' | 'error' | 'checking'

interface SvcState {
  status: SvcStatus
  ms: number | null
  detail: string | null
  lastChecked: number | null
}

interface ServiceDef {
  id: string
  label: string
  description: string
  containerName: string
  portLabel: string
  navUrl: string | null
  icon: React.ReactNode
}

// ── Service definitions ───────────────────────────────────────────────────

const SERVICES: ServiceDef[] = [
  {
    id: 'api',
    label: 'MedImage API',
    description: 'Training & jobs backend',
    containerName: 'medimage-medimage-api-1',
    portLabel: ':8000',
    navUrl: '/jobs',
    icon: <Server size={18} />,
  },
  {
    id: 'minio',
    label: 'MinIO Storage',
    description: 'S3-compatible object storage',
    containerName: 'medimage-minio-1',
    portLabel: ':9000 S3 · :9001 Console',
    navUrl: '/storage',
    icon: <HardDrive size={18} />,
  },
  {
    id: 'jupyter',
    label: 'Jupyter Lab',
    description: 'Notebook server for ML experiments',
    containerName: 'medimage-jupyter-1',
    portLabel: ':8888',
    navUrl: '/notebook',
    icon: <BookOpen size={18} />,
  },
  {
    id: 'labelstudio',
    label: 'Label Studio',
    description: 'Data annotation & labeling',
    containerName: 'medimage-label-studio-1',
    portLabel: ':8085',
    navUrl: 'http://localhost:8085',
    icon: <Tag size={18} />,
  },
  {
    id: 'ray',
    label: 'Ray Cluster',
    description: 'Distributed GPU compute cluster',
    containerName: '100.68.53.118 (tailscale)',
    portLabel: ':8265 dashboard',
    navUrl: '/ray-cluster',
    icon: <Cpu size={18} />,
  },
]

// ── Probe helper ──────────────────────────────────────────────────────────

async function probe(
  url: string,
  opts?: RequestInit,
): Promise<{ ok: boolean; ms: number; body?: any }> {
  const start = performance.now()
  try {
    const res = await fetch(url, { ...opts, cache: 'no-store' })
    const ms = Math.round(performance.now() - start)
    let body: any
    try { body = await res.json() } catch { /* non-JSON ok */ }
    return { ok: res.ok, ms, body }
  } catch {
    return { ok: false, ms: Math.round(performance.now() - start) }
  }
}

// ── Individual service checks ─────────────────────────────────────────────

async function checkApi(): Promise<{ ok: boolean; ms: number; detail: string | null }> {
  const { ok, ms, body } = await probe('/api/jobs')
  const count = Array.isArray(body?.jobs) ? body.jobs.length : null
  return {
    ok,
    ms,
    detail: ok
      ? count !== null
        ? `FastAPI · ${count} job${count !== 1 ? 's' : ''} in DB`
        : 'FastAPI · SQLite'
      : null,
  }
}

async function checkMinio(): Promise<{ ok: boolean; ms: number; detail: string | null }> {
  const { ok, ms } = await probe('/api/minio/minio/health/live')
  return { ok, ms, detail: ok ? 'S3 API healthy' : null }
}

async function checkJupyter(): Promise<{ ok: boolean; ms: number; detail: string | null }> {
  const { ok, ms, body } = await probe('/jupyter/api/kernels', {
    headers: { Authorization: 'token medimage2026' },
  })
  if (!ok) return { ok, ms, detail: null }
  let detail = 'Running'
  if (Array.isArray(body)) {
    const n = body.length
    const busy = body.filter((k: any) => k.execution_state === 'busy').length
    const idle = body.filter((k: any) => k.execution_state === 'idle').length
    detail = n === 0
      ? 'No active kernels'
      : `${n} kernel${n !== 1 ? 's' : ''} · ${busy} busy · ${idle} idle`
  }
  return { ok, ms, detail }
}

async function checkLabelStudio(): Promise<{ ok: boolean; ms: number; detail: string | null }> {
  const [health, version] = await Promise.all([
    probe('/api/ls/api/health/'),
    probe('/api/ls/version/'),
  ])
  const ver: string | null = version.body?.release ?? null
  return {
    ok: health.ok,
    ms: health.ms,
    detail: health.ok ? (ver ? `Label Studio v${ver}` : 'Running') : null,
  }
}

async function checkRay(): Promise<{ ok: boolean; ms: number; detail: string | null }> {
  const start = performance.now()
  try {
    const [cRes, nRes] = await Promise.all([
      probe('/api/ray/api/cluster_status'),
      probe('/api/ray/nodes?view=summary'),
    ])
    const ms = Math.round(performance.now() - start)
    if (!cRes.ok) return { ok: false, ms, detail: null }

    const nodes: any[] = nRes.body?.data?.summary ?? []
    const alive = nodes.filter((n: any) => n.raylet?.state === 'ALIVE').length
    const usage = cRes.body?.data?.clusterStatus?.loadMetricsReport?.usage ?? {}
    const gpuTotal = Math.round(usage['GPU']?.[1] ?? 0)
    const gpuUsed  = Number((usage['GPU']?.[0] ?? 0).toFixed(1))
    const cpuTotal = Math.round(usage['CPU']?.[1] ?? 0)
    const cpuUsed  = Math.round(usage['CPU']?.[0] ?? 0)

    const detail = [
      `${alive}/${nodes.length} node${nodes.length !== 1 ? 's' : ''}`,
      `CPU ${cpuUsed}/${cpuTotal}`,
      gpuTotal > 0 ? `GPU ${gpuUsed}/${gpuTotal}` : null,
    ].filter(Boolean).join(' · ')

    return { ok: true, ms, detail }
  } catch {
    return { ok: false, ms: Math.round(performance.now() - start), detail: null }
  }
}

// ── Combined checker ──────────────────────────────────────────────────────

async function checkAll(): Promise<Record<string, { ok: boolean; ms: number; detail: string | null }>> {
  const [api, minio, jupyter, ls, ray] = await Promise.all([
    checkApi(),
    checkMinio(),
    checkJupyter(),
    checkLabelStudio(),
    checkRay(),
  ])
  return { api, minio, jupyter, labelstudio: ls, ray }
}

// ── Sub-components ────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: SvcStatus }) {
  if (status === 'checking') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <RefreshCw size={12} className="animate-spin" style={{ color: 'var(--text-muted)' }} />
        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 500 }}>Checking…</span>
      </div>
    )
  }
  const ok = status === 'ok'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
      <div style={{
        width: 7, height: 7, borderRadius: '50%',
        background: ok ? 'var(--success)' : 'var(--danger)',
        boxShadow: ok ? '0 0 6px var(--success)' : '0 0 6px var(--danger)',
      }} />
      <span style={{ fontSize: 11, fontWeight: 600, color: ok ? 'var(--success)' : 'var(--danger)' }}>
        {ok ? 'Online' : 'Offline'}
      </span>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function Status() {
  const initState = (): Record<string, SvcState> => {
    const s: Record<string, SvcState> = {}
    SERVICES.forEach(svc => { s[svc.id] = { status: 'checking', ms: null, detail: null, lastChecked: null } })
    return s
  }

  const [states, setStates] = useState<Record<string, SvcState>>(initState)
  const [refreshing, setRefreshing] = useState(false)
  const [lastRefresh, setLastRefresh] = useState<number | null>(null)

  const refresh = useCallback(async () => {
    setRefreshing(true)
    setStates(prev => {
      const next = { ...prev }
      SERVICES.forEach(s => { next[s.id] = { ...next[s.id], status: 'checking' } })
      return next
    })
    const results = await checkAll()
    const now = Date.now()
    setStates(prev => {
      const next = { ...prev }
      for (const [id, r] of Object.entries(results)) {
        next[id] = { status: r.ok ? 'ok' : 'error', ms: r.ms, detail: r.detail, lastChecked: now }
      }
      return next
    })
    setLastRefresh(now)
    setRefreshing(false)
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 30_000)
    return () => clearInterval(id)
  }, [refresh])

  const anyError    = Object.values(states).some(s => s.status === 'error')
  const onlineCount = Object.values(states).filter(s => s.status === 'ok').length

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
            System Status
          </h1>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            {lastRefresh
              ? `Last updated ${new Date(lastRefresh).toLocaleTimeString()} · auto-refresh every 30s`
              : 'Checking services…'}
          </div>
        </div>
        <button
          className="btn btn-secondary"
          onClick={refresh}
          disabled={refreshing}
          style={{ fontSize: 13 }}
        >
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Summary banner */}
      <div
        className="card"
        style={{
          marginBottom: 24,
          padding: '14px 18px',
          background: anyError
            ? 'color-mix(in oklch, var(--danger) 8%, var(--bg-surface))'
            : 'color-mix(in oklch, var(--success) 8%, var(--bg-surface))',
          border: `1px solid ${anyError ? 'color-mix(in oklch, var(--danger) 40%, var(--border-default))' : 'color-mix(in oklch, var(--success) 40%, var(--border-default))'}`,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        {anyError
          ? <XCircle size={16} style={{ color: 'var(--danger)', flexShrink: 0 }} />
          : <CheckCircle size={16} style={{ color: 'var(--success)', flexShrink: 0 }} />
        }
        <span style={{ fontWeight: 600, fontSize: 14, color: anyError ? 'var(--danger)' : 'var(--success)' }}>
          {anyError ? 'One or more services are offline' : 'All systems operational'}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 2 }}>
          {onlineCount} / {SERVICES.length} online
        </span>
      </div>

      {/* Service cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
        {SERVICES.map(svc => {
          const state = states[svc.id]
          const isOk = state.status === 'ok'
          const isErr = state.status === 'error'

          return (
            <div
              key={svc.id}
              className="card"
              style={{
                padding: '18px 20px',
                borderColor: isErr ? 'color-mix(in oklch, var(--danger) 30%, var(--border-default))' : undefined,
              }}
            >
              {/* Card header */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10, marginBottom: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                    background: isOk
                      ? 'color-mix(in oklch, var(--primary) 12%, var(--bg-surface))'
                      : isErr
                        ? 'color-mix(in oklch, var(--danger) 10%, var(--bg-surface))'
                        : 'var(--bg-elevated)',
                    color: isOk ? 'var(--primary)' : isErr ? 'var(--danger)' : 'var(--text-muted)',
                  }}>
                    {svc.icon}
                  </div>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
                      {svc.label}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                      {svc.description}
                    </div>
                  </div>
                </div>
                <StatusBadge status={state.status} />
              </div>

              {/* Divider */}
              <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 12 }}>
                {/* Container name */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)', marginBottom: 5 }}>
                  <Layers size={11} style={{ flexShrink: 0 }} />
                  <span style={{ fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {svc.containerName}
                  </span>
                </div>

                {/* Port */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                  <Activity size={11} style={{ flexShrink: 0 }} />
                  <span style={{ fontFamily: 'var(--font-mono)' }}>{svc.portLabel}</span>
                </div>

                {/* Service-specific detail */}
                {state.detail && (
                  <div style={{
                    fontSize: 12, color: 'var(--text-secondary)',
                    padding: '5px 8px', borderRadius: 6,
                    background: 'var(--bg-elevated)',
                    marginBottom: 8,
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}>
                    <Zap size={11} style={{ color: 'var(--primary)', flexShrink: 0 }} />
                    {state.detail}
                  </div>
                )}

                {/* Response time */}
                {state.ms !== null && (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4, marginBottom: 10 }}>
                    <span>Response:</span>
                    <span style={{
                      fontFamily: 'var(--font-mono)', fontWeight: 500,
                      color: state.ms > 800 ? 'var(--danger)' : state.ms > 300 ? 'oklch(0.75 0.15 60)' : 'var(--success)',
                    }}>
                      {state.ms} ms
                    </span>
                  </div>
                )}

                {/* Open button */}
                {svc.navUrl && (
                  <a
                    href={svc.navUrl}
                    target={svc.navUrl.startsWith('http') ? '_blank' : undefined}
                    rel={svc.navUrl.startsWith('http') ? 'noopener noreferrer' : undefined}
                    className="btn btn-secondary btn-sm"
                    style={{
                      width: '100%', justifyContent: 'center', textDecoration: 'none',
                      fontSize: 12, padding: '5px 0',
                      display: 'flex', alignItems: 'center', gap: 5,
                      opacity: isErr ? 0.5 : 1,
                      pointerEvents: isErr ? 'none' : 'auto',
                    }}
                  >
                    <ExternalLink size={11} />Open
                  </a>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
