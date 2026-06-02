import { useState, useEffect } from 'react'
import { LayoutDashboard, Activity, Database, Brain } from 'lucide-react'

interface ClusterStatus {
  status: string
  cpus: number
  total_cpus: number
  gpus: number
  total_gpus: number
  memory_gb: number
  memory_total_gb: number
}

interface StorageInfo {
  bucket: string
  size_mb: number
}

interface JobsInfo {
  jobs: Array<{ status: string }>
}

export default function Dashboard() {
  const [cluster, setCluster] = useState<ClusterStatus | null>(null)
  const [storage, setStorage] = useState<StorageInfo[]>([])
  const [jobs, setJobs] = useState<JobsInfo['jobs']>([])
  const [lsStats, setLsStats] = useState<{ total: number; labeled: number }>({ total: 0, labeled: 0 })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [clusterRes, storageRes, jobsRes] = await Promise.all([
          fetch('/api/cluster/status').then(r => r.json()).catch(() => null),
          fetch('/api/storage/buckets').then(r => r.json()).catch(() => []),
          fetch('/api/jobs').then(r => r.json()).catch(() => ({ jobs: [] })),
        ])

        if (clusterRes && clusterRes.status !== 'error') {
          setCluster(clusterRes)
        }
        if (Array.isArray(storageRes)) {
          // Map API fields: API returns {name, objects}, we use {bucket, size_mb}
          setStorage(storageRes.map((b: any) => ({
            bucket: b.name,
            size_mb: (b.objects || 0) * 50, // approximate: 50MB per object average
          })))
        }
        if (jobsRes?.jobs) {
          setJobs(jobsRes.jobs)
        }

        // Fetch LS task stats for image counts
        try {
          const LS_TOKEN = '160d2644f4d45f84cd09f8931d20891e52f5e4cf'
          const taskRes = await fetch('/api/ls/tasks?project=1&page_size=1000', {
            headers: { Authorization: `Token ${LS_TOKEN}` }
          }).catch(() => null)
          if (taskRes?.ok) {
            const taskData = await taskRes.json()
            const tasks: any[] = taskData.tasks || []
            setLsStats({
              total: tasks.length,
              labeled: tasks.filter((t: any) => t.is_labeled).length,
            })
          }
        } catch (_) {}
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
    const interval = setInterval(fetchAll, 30000) // refresh every 30s
    return () => clearInterval(interval)
  }, [])

  const gpuUtilPct = cluster
    ? Math.round((cluster.gpus / Math.max(cluster.total_gpus, 1)) * 100)
    : 0
  const memUtilPct = cluster
    ? Math.round((cluster.memory_gb / Math.max(cluster.memory_total_gb, 1)) * 100)
    : 0
  const storageUsedGB = storage.reduce((sum, b) => sum + (b.size_mb || 0) / 1024, 0)
  const storageTotalGB = 100
  const storagePct = storageTotalGB > 0 ? Math.min(Math.round((storageUsedGB / storageTotalGB) * 100), 100) : 0
  const activeJobs = jobs.filter(j => j.status === 'running' || j.status === 'queued').length
  const completedJobs = jobs.filter(j => j.status === 'completed').length

  const statCards = [
    {
      label: 'Total Images',
      value: lsStats.total.toLocaleString(),
      sub: lsStats.total > 0 ? `${Math.round((lsStats.labeled / lsStats.total) * 100)}% labeled` : null,
      subColor: 'var(--success)',
      icon: <Database size={15} />,
    },
    {
      label: 'Labeled',
      value: lsStats.labeled.toLocaleString(),
      sub: null,
      icon: <Activity size={15} />,
    },
    {
      label: 'Active Jobs',
      value: activeJobs.toString(),
      sub: null,
      icon: <Brain size={15} />,
    },
    {
      label: 'Completed',
      value: completedJobs.toString(),
      sub: null,
      icon: <LayoutDashboard size={15} />,
    },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner" />
      </div>
    )
  }

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
            {card.sub && (
              <div className="stat-sub" style={{ color: card.subColor }}>
                {card.sub}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Ray Cluster + Storage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 items-stretch">
        {/* Ray Cluster Status */}
        <div className="card flex flex-col">
          <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', marginBottom: 16 }}>Ray Cluster</h3>
          <div className="space-y-4 flex-1">
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13, marginBottom: 8 }}>
                <span style={{ color: 'var(--text-secondary)' }}>GPU Utilization</span>
                <span style={{ color: 'var(--primary-hover)', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{gpuUtilPct}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-bar-fill" style={{ width: `${gpuUtilPct}%` }} />
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{cluster?.gpus ?? '?'} / {cluster?.total_gpus ?? '?'} GPUs</div>
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13, marginBottom: 8 }}>
                <span style={{ color: 'var(--text-secondary)' }}>Memory Usage</span>
                <span style={{ color: 'var(--primary-hover)', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{memUtilPct}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-bar-fill" style={{ width: `${memUtilPct}%` }} />
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                {cluster ? `${Math.round(cluster.memory_gb)} GB / ${Math.round(cluster.memory_total_gb)} GB` : 'Connecting...'}
              </div>
            </div>
          </div>
          <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)' }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: cluster ? 'var(--success)' : 'var(--danger)', flexShrink: 0 }} />
              <span style={{ fontFamily: 'var(--font-mono)' }}>100.68.53.118</span>
              {cluster && <span>{cluster.total_cpus} CPUs · {cluster.total_gpus} GPUs</span>}
            </div>
          </div>
        </div>

        {/* Storage */}
        <div className="card flex flex-col">
          <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', marginBottom: 16 }}>Storage</h3>
          <div className="space-y-4 flex-1">
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13, marginBottom: 8 }}>
                <span style={{ color: 'var(--text-secondary)' }}>MinIO</span>
                <span style={{ color: 'var(--text-secondary)', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
                  {storageUsedGB.toFixed(1)} / {storageTotalGB} GB
                </span>
              </div>
              <div className="progress-bar">
                <div className="progress-bar-fill" style={{ width: `${storagePct}%` }} />
              </div>
            </div>
            {storage.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {storage.slice(0, 4).map((b) => (
                  <div key={b.bucket} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '65%' }}>{b.bucket}</span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{(b.size_mb / 1024).toFixed(2)} GB</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: 13, color: 'var(--text-muted)', paddingTop: 8 }}>No buckets found — create a project to get started</div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', marginBottom: 14 }}>Quick Actions</h3>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <a href="/datasets" className="btn btn-secondary">
            <Database size={15} />
            Datasets
          </a>
          <a href="/train" className="btn btn-primary">
            <Brain size={15} />
            New Training Job
          </a>
          <a href="/jobs" className="btn btn-secondary">
            <Activity size={15} />
            View Jobs
          </a>
          <a
            href="http://100.68.53.118:8265"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
          >
            <LayoutDashboard size={15} />
            Ray Dashboard
          </a>
        </div>
      </div>
    </div>
  )
}
