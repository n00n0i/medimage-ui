import { useState, useEffect } from 'react'
import { LayoutDashboard, Activity, Database, Brain, Cpu, HardDrive, Zap, Server, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'
import { fetchDiskInfo, type DiskInfo } from '../lib/minioClient'

interface DashboardStats {
  label_studio: { total_tasks: number; labeled: number; projects: number; error: string | null }
  jobs:         { total: number; by_status: Record<string, number> }
  storage:      { buckets: Array<{ name: string; size_bytes: number; object_count: number }>; error: string | null }
  ray:          { url: string }
}

interface RayClusterSummary {
  cpuUsed: number; cpuTotal: number
  gpuUsed: number; gpuTotal: number; gpuAllocPct: number; gpuUtilPct: number; gpuActiveCount: number; gpuTotalCount: number
  memUsedGb: number; memTotalGb: number
  nodesAlive: number; nodesTotal: number
  connected: boolean
}

async function fetchRayCluster(_url: string): Promise<RayClusterSummary> {
  const proxied = '/api/ray/api/cluster_status'
  const nodes   = '/api/ray/nodes?view=summary'
  const [sRes, nRes, gpuRes] = await Promise.all([
    fetch(proxied),
    fetch(nodes),
    fetch('/api/ray/gpu-stats').catch(() => null),
  ])
  if (!sRes.ok || !nRes.ok) throw new Error('unreachable')
  const sData = await sRes.json()
  const nData = await nRes.json()
  const usage: Record<string, [number, number]> = sData.data?.clusterStatus?.loadMetricsReport?.usage ?? {}
  const nodesArr: any[] = nData.data?.summary ?? []
  // Real GPU compute utilization from nvidia-smi (if available)
  let gpuComputeUtil = 0
  let gpuRealUtil = 0
  let gpuActiveCount = 0
  let gpuTotalCount = 0
  if (gpuRes && gpuRes.ok) {
    const gd = await gpuRes.json()
    if (gd.cluster?.gpu_allocation_pct != null) {
      gpuComputeUtil = gd.cluster.gpu_allocation_pct
    }
    if (gd.cluster?.gpu_util_pct != null) {
      gpuRealUtil = gd.cluster.gpu_util_pct
    }
    if (gd.cluster?.gpu_active_count != null) {
      gpuActiveCount = gd.cluster.gpu_active_count
    }
    if (gd.cluster?.gpu_total_count != null) {
      gpuTotalCount = gd.cluster.gpu_total_count
    }
  }
   return {
    cpuUsed:    Math.round(usage['CPU']?.[0]    ?? 0),
    cpuTotal:   Math.round(usage['CPU']?.[1]    ?? 0),
    gpuUsed:    usage['GPU']?.[0] ?? 0,
    gpuTotal:   usage['GPU']?.[1] ?? 0,
    gpuAllocPct: gpuComputeUtil > 0 ? gpuComputeUtil : (usage['GPU']?.[1] ? (usage['GPU']?.[0] / usage['GPU'][1]) * 100 : 0),
    gpuUtilPct: gpuRealUtil,
    gpuActiveCount,
    gpuTotalCount: gpuTotalCount || (usage['GPU']?.[1] ?? 0),
    memUsedGb:  (usage['memory']?.[0] ?? 0) / 1e9,
    memTotalGb: (usage['memory']?.[1] ?? 0) / 1e9,
    nodesAlive: nodesArr.filter((n: any) => n.raylet?.state === 'ALIVE').length,
    nodesTotal: nodesArr.length,
    connected:  true,
  }
}

function pct(used: number, total: number) {
  return total > 0 ? Math.round((used / total) * 100) : 0
}

function UtilRow({ label, p, detail }: { label: string; p: number; detail: string }) {
  const bar = p > 80 ? 'var(--danger)' : p > 60 ? 'var(--warning)' : 'var(--primary)'
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 6 }}>
        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontWeight: 500, color: p > 80 ? 'var(--danger)' : 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{p}%</span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${p}%`, background: bar }} />
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{detail}</div>
    </div>
  )
}

export default function Dashboard() {
  const [stats,    setStats]    = useState<DashboardStats | null>(null)
  const [ray,      setRay]      = useState<RayClusterSummary | null>(null)
  const [rayUrl,   setRayUrl]   = useState<string>('')
  const [diskInfo, setDiskInfo] = useState<DiskInfo | null>(null)
  const [loading,  setLoading]  = useState(true)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        // Single aggregated call
        const res = await fetch('/api/dashboard/stats')
        if (res.ok) {
          const data: DashboardStats = await res.json()
          setStats(data)
          setRayUrl(data.ray?.url ?? '')

          // Disk info (host filesystem free/total) for the storage bar
          fetchDiskInfo().then(d => setDiskInfo(d)).catch(() => {})

          // Ray cluster live metrics — only if we have a URL
          if (data.ray?.url) {
            fetchRayCluster(data.ray.url).then(setRay).catch(() => setRay(null))
          }
        }
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
    const id = setInterval(fetchAll, 30000)
    return () => clearInterval(id)
  }, [])

  function fmtBytes(b: number) {
    if (b >= 1e12) return `${(b / 1e12).toFixed(1)} TB`
    if (b >= 1e9)  return `${(b / 1e9).toFixed(1)} GB`
    if (b >= 1e6)  return `${(b / 1e6).toFixed(0)} MB`
    return `${b} B`
  }

  const storageBuckets = stats?.storage.buckets ?? []
  const storageTotalBytes = storageBuckets.reduce((s, b) => s + b.size_bytes, 0)
  const storageUsedBytes  = diskInfo?.totalBytes ? Math.min(storageTotalBytes, diskInfo.totalBytes) : storageTotalBytes
  const storagePct = (diskInfo?.totalBytes ?? 0) > 0
    ? Math.min(Math.round((storageUsedBytes / diskInfo!.totalBytes) * 100), 100)
    : 0

  const ls      = stats?.label_studio
  const jobs    = stats?.jobs
  const running = jobs?.by_status['running'] ?? 0
  const queued  = jobs?.by_status['queued']  ?? 0
  const done    = jobs?.by_status['completed'] ?? 0
  const failed  = jobs?.by_status['error'] ?? 0
  const active  = running + queued

  const statCards = [
    { label: 'Total Images', value: (ls?.total_tasks ?? 0).toLocaleString(), sub: ls?.projects ? `${ls.projects} project${ls.projects !== 1 ? 's' : ''}` : null, subColor: 'var(--text-muted)', icon: <Database size={15} /> },
    { label: 'Labeled',      value: (ls?.labeled     ?? 0).toLocaleString(), sub: ls?.total_tasks ? `${Math.round((ls.labeled / ls.total_tasks) * 100)}% of ${ls.total_tasks}` : null, subColor: 'var(--success)', icon: <CheckCircle2 size={15} /> },
    { label: 'Active Jobs',  value: String(active),                         sub: running > 0 || queued > 0 ? `${running} running · ${queued} queued` : null, subColor: 'var(--text-muted)', icon: <Brain size={15} /> },
    { label: 'Completed',    value: String(done),                           sub: failed > 0 ? `${failed} failed` : null, subColor: failed > 0 ? 'var(--danger)' : 'var(--text-muted)', icon: <LayoutDashboard size={15} /> },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner" />
      </div>
    )
  }

  const cpuPct = ray ? pct(ray.cpuUsed, ray.cpuTotal) : 0
  const hasGpuUtil = ray ? ray.gpuActiveCount > 0 : false
  const gpuPct = ray ? (hasGpuUtil ? ray.gpuUtilPct : (ray.gpuTotal > 0 ? (ray.gpuUsed / ray.gpuTotal) * 100 : 0)) : 0
  const gpuDetail = ray ? (hasGpuUtil ? `${Math.round(ray.gpuUtilPct)}% utilized` : `${ray.gpuUsed} / ${ray.gpuTotal} allocated`) : ''
  const memPct = ray ? pct(ray.memUsedGb, ray.memTotalGb) : 0

  return (
    <div className="max-w-6xl mx-auto">
      <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 20 }}>Dashboard</h1>

      {/* Stat Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {statCards.map((card, i) => (
          <div key={i} className="stat-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', marginBottom: 6 }}>
              {card.icon}
              <span className="stat-label">{card.label}</span>
            </div>
            <div className="stat-value">{card.value}</div>
            {card.sub && <div className="stat-sub" style={{ color: card.subColor }}>{card.sub}</div>}
          </div>
        ))}
      </div>

      {/* Error banner for partial failures */}
      {(ls?.error || stats?.storage.error) && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 14px', marginBottom: 16, borderRadius: 8,
          background: 'var(--warning-dim, #f59e0b10)',
          border: '1px solid var(--warning, #f59e0b40)',
          color: 'var(--text-secondary)', fontSize: 13,
        }}>
          <AlertTriangle size={16} style={{ color: 'var(--warning)', flexShrink: 0 }} />
          <span>
            {ls?.error && <>Label Studio: {ls.error}. </>}
            {stats?.storage.error && <>MinIO: {stats.storage.error}. </>}
            Some stats may be incomplete.
          </span>
        </div>
      )}

      {/* Job status breakdown */}
      {jobs && jobs.total > 0 && (
        <div className="card mb-6">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', margin: 0 }}>Job Status</h3>
            <a href="/jobs" style={{ fontSize: 12, color: 'var(--primary-hover)', textDecoration: 'none' }}>View all →</a>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {([
              { key: 'running',   label: 'Running',   count: running, icon: <Activity size={14} />,       color: 'var(--primary)' },
              { key: 'queued',    label: 'Queued',    count: queued,  icon: <Brain size={14} />,         color: 'var(--text-secondary)' },
              { key: 'completed', label: 'Completed', count: done,    icon: <CheckCircle2 size={14} />,  color: 'var(--success)' },
              { key: 'error',     label: 'Failed',    count: failed,  icon: <XCircle size={14} />,       color: 'var(--danger)' },
            ] as const).map(s => (
              <div key={s.key} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '10px 12px', borderRadius: 8,
                background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
              }}>
                <span style={{ color: s.color, flexShrink: 0 }}>{s.icon}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{s.label}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: s.color, fontVariantNumeric: 'tabular-nums' }}>{s.count}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ray Cluster + Storage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 items-stretch">

        {/* Ray Cluster — live from Ray API */}
        <div className="card flex flex-col">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', margin: 0 }}>Ray Cluster</h3>
            {ray && (
              <div style={{ display: 'flex', gap: 10, fontSize: 11, color: 'var(--text-muted)' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Cpu size={11} />{ray.cpuTotal} CPU</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Zap size={11} style={{ color: 'var(--accent)' }} />{ray.gpuTotal} GPU</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Server size={11} />{ray.nodesAlive}/{ray.nodesTotal}</span>
              </div>
            )}
          </div>

          {ray ? (
            <div className="space-y-4 flex-1">
              <UtilRow label="CPU Allocation"  p={cpuPct} detail={`${ray.cpuUsed} / ${ray.cpuTotal} cores`} />
              {ray.gpuTotal > 0 && (
                <UtilRow label="GPU" p={gpuPct} detail={gpuDetail} />
              )}
              <UtilRow label="Memory" p={memPct} detail={`${ray.memUsedGb.toFixed(1)} / ${ray.memTotalGb.toFixed(1)} GB`} />
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-muted)' }}>
              <HardDrive size={14} />
              <span>Cluster unreachable — check the Ray URL in Ray Cluster settings</span>
            </div>
          )}

          <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)' }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', flexShrink: 0, background: ray ? 'var(--success)' : 'var(--danger)' }} />
              <span style={{ fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{rayUrl || '—'}</span>
            </div>
          </div>
        </div>

        {/* Storage */}
        <div className="card flex flex-col">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', margin: 0 }}>Storage</h3>
            <a href="/storage" style={{ fontSize: 12, color: 'var(--primary-hover)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 4 }}>
              View all <HardDrive size={11} />
            </a>
          </div>
          <div className="space-y-4 flex-1">
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13, marginBottom: 8 }}>
                <span style={{ color: 'var(--text-secondary)' }}>MinIO · {storageBuckets.length} bucket{storageBuckets.length !== 1 ? 's' : ''}</span>
                <span style={{ color: 'var(--text-secondary)', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
                  {fmtBytes(storageUsedBytes)}{diskInfo ? ` / ${fmtBytes(diskInfo.totalBytes)}` : ''}
                </span>
              </div>
              <div className="progress-bar">
                <div className="progress-bar-fill" style={{ width: `${storagePct}%` }} />
              </div>
              {diskInfo && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  {storagePct}% used · {fmtBytes(diskInfo.freeBytes)} free
                </div>
              )}
            </div>
            {storageBuckets.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {storageBuckets.slice(0, 4).map((b) => (
                  <div key={b.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '55%' }}>{b.name}</span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                      {fmtBytes(b.size_bytes)} · {b.object_count.toLocaleString()} obj
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: 13, color: 'var(--text-muted)', paddingTop: 8 }}>No buckets found</div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', marginBottom: 14 }}>Quick Actions</h3>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <a href="/datasets" className="btn btn-secondary"><Database size={15} />Datasets</a>
          <a href="/storage" className="btn btn-secondary"><HardDrive size={15} />Storage</a>
          <a href="/train" className="btn btn-primary"><Brain size={15} />New Training Job</a>
          <a href="/jobs" className="btn btn-secondary"><Activity size={15} />View Jobs</a>
          {rayUrl && (
            <a href={rayUrl} target="_blank" rel="noopener noreferrer" className="btn btn-secondary">
              <LayoutDashboard size={15} />Ray Dashboard
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
