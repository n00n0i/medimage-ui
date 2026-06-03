import { useState, useEffect } from 'react'
import { LayoutDashboard, Activity, Database, Brain, Cpu, HardDrive, Zap, Server } from 'lucide-react'
import { listBuckets, listObjects, fetchDiskInfo, type DiskInfo } from '../lib/minioClient'

const RAY_STORAGE_KEY = 'ray_head_url'
const RAY_DEFAULT_URL = 'http://100.68.53.118:8265'

interface RayClusterSummary {
  cpuUsed: number; cpuTotal: number
  gpuUsed: number; gpuTotal: number
  memUsedGb: number; memTotalGb: number
  nodesAlive: number; nodesTotal: number
  connected: boolean
}

interface StorageInfo {
  bucket: string
  sizeBytes: number
}

interface JobsInfo {
  jobs: Array<{ status: string }>
}

async function fetchRayCluster(): Promise<RayClusterSummary> {
  const [sRes, nRes] = await Promise.all([
    fetch('/api/ray/api/cluster_status'),
    fetch('/api/ray/nodes?view=summary'),
  ])
  if (!sRes.ok || !nRes.ok) throw new Error('unreachable')
  const sData = await sRes.json()
  const nData = await nRes.json()
  const usage: Record<string, [number, number]> = sData.data?.clusterStatus?.loadMetricsReport?.usage ?? {}
  const nodes: any[] = nData.data?.summary ?? []
  return {
    cpuUsed:    Math.round(usage['CPU']?.[0]    ?? 0),
    cpuTotal:   Math.round(usage['CPU']?.[1]    ?? 0),
    gpuUsed:    usage['GPU']?.[0]    ?? 0,
    gpuTotal:   usage['GPU']?.[1]    ?? 0,
    memUsedGb:  (usage['memory']?.[0] ?? 0) / 1e9,
    memTotalGb: (usage['memory']?.[1] ?? 0) / 1e9,
    nodesAlive: nodes.filter((n: any) => n.raylet?.state === 'ALIVE').length,
    nodesTotal: nodes.length,
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
  const [ray, setRay]         = useState<RayClusterSummary | null>(null)
  const [rayUrl]              = useState(() => localStorage.getItem(RAY_STORAGE_KEY) ?? RAY_DEFAULT_URL)
  const [storage, setStorage] = useState<StorageInfo[]>([])
  const [diskInfo, setDiskInfo] = useState<DiskInfo | null>(null)
  const [jobs, setJobs]       = useState<JobsInfo['jobs']>([])
  const [lsStats, setLsStats] = useState<{ total: number; labeled: number }>({ total: 0, labeled: 0 })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        // Ray cluster (direct Ray API — same as Ray Cluster page)
        fetchRayCluster().then(setRay).catch(() => setRay(null))

        // MinIO storage — real bucket data
        listBuckets()
          .then(async (buckets) => {
            fetchDiskInfo().then(d => setDiskInfo(d)).catch(() => {})
            const infos: StorageInfo[] = await Promise.all(
              buckets.map(async (b) => {
                try {
                  const objs = await listObjects(b.name)
                  return { bucket: b.name, sizeBytes: objs.reduce((s, o) => s + o.size, 0) }
                } catch {
                  return { bucket: b.name, sizeBytes: 0 }
                }
              })
            )
            setStorage(infos)
          })
          .catch(() => {})

        try {
          const jobsRes = await fetch('/api/jobs').then(r => r.json()).catch(() => ({ jobs: [] }))
          if (jobsRes?.jobs) setJobs(jobsRes.jobs)
        } catch (_) {}

        try {
          const LS_TOKEN = 'medimage-ls-token-2026'
          const taskRes = await fetch('/api/ls/tasks?project=1&page_size=1000', {
            headers: { Authorization: `Token ${LS_TOKEN}` }
          }).catch(() => null)
          if (taskRes?.ok) {
            const taskData = await taskRes.json()
            const tasks: any[] = taskData.tasks || []
            setLsStats({ total: tasks.length, labeled: tasks.filter((t: any) => t.is_labeled).length })
          }
        } catch (_) {}
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
    const id = setInterval(fetchAll, 30000)
    return () => clearInterval(id)
  }, [])

  const storageTotalBytes = diskInfo?.totalBytes ?? 0
  const storageUsedBytes  = storage.reduce((s, b) => s + b.sizeBytes, 0)
  const storagePct = storageTotalBytes > 0
    ? Math.min(Math.round((storageUsedBytes / storageTotalBytes) * 100), 100)
    : 0

  function fmtBytes(b: number) {
    if (b >= 1e12) return `${(b / 1e12).toFixed(1)} TB`
    if (b >= 1e9)  return `${(b / 1e9).toFixed(1)} GB`
    if (b >= 1e6)  return `${(b / 1e6).toFixed(0)} MB`
    return `${b} B`
  }
  const activeJobs     = jobs.filter(j => j.status === 'running' || j.status === 'queued').length
  const completedJobs  = jobs.filter(j => j.status === 'completed').length

  const statCards = [
    { label: 'Total Images', value: lsStats.total.toLocaleString(),   sub: lsStats.total > 0 ? `${Math.round((lsStats.labeled / lsStats.total) * 100)}% labeled` : null, subColor: 'var(--success)', icon: <Database size={15} /> },
    { label: 'Labeled',      value: lsStats.labeled.toLocaleString(), sub: null, icon: <Activity size={15} /> },
    { label: 'Active Jobs',  value: activeJobs.toString(),            sub: null, icon: <Brain size={15} /> },
    { label: 'Completed',    value: completedJobs.toString(),         sub: null, icon: <LayoutDashboard size={15} /> },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner" />
      </div>
    )
  }

  const cpuPct = ray ? pct(ray.cpuUsed, ray.cpuTotal) : 0
  const gpuPct = ray ? pct(ray.gpuUsed, ray.gpuTotal) : 0
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
              {ray.gpuTotal > 0 && <UtilRow label="GPU Allocation" p={gpuPct} detail={`${ray.gpuUsed} / ${ray.gpuTotal} GPUs`} />}
              <UtilRow label="Memory" p={memPct} detail={`${ray.memUsedGb.toFixed(1)} / ${ray.memTotalGb.toFixed(1)} GB`} />
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-muted)' }}>
              <HardDrive size={14} />
              <span>Cluster unreachable — check VPN</span>
            </div>
          )}

          <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)' }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', flexShrink: 0, background: ray ? 'var(--success)' : 'var(--danger)' }} />
              <span style={{ fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{rayUrl}</span>
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
                <span style={{ color: 'var(--text-secondary)' }}>MinIO · {storage.length} bucket{storage.length !== 1 ? 's' : ''}</span>
                <span style={{ color: 'var(--text-secondary)', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
                  {storageTotalBytes > 0 ? `${fmtBytes(storageUsedBytes)} / ${fmtBytes(storageTotalBytes)}` : fmtBytes(storageUsedBytes)}
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
            {storage.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {storage.slice(0, 4).map((b) => (
                  <div key={b.bucket} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '65%' }}>{b.bucket}</span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{fmtBytes(b.sizeBytes)}</span>
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
          <a href={rayUrl} target="_blank" rel="noopener noreferrer" className="btn btn-secondary">
            <LayoutDashboard size={15} />Ray Dashboard
          </a>
        </div>
      </div>
    </div>
  )
}
