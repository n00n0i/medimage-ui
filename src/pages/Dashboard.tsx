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
      subColor: 'text-green-500',
      icon: <Database size={18} className="text-indigo-400" />,
      valueColor: 'text-white',
    },
    {
      label: 'Labeled',
      value: lsStats.labeled.toLocaleString(),
      sub: null,
      icon: <Activity size={18} className="text-green-400" />,
      valueColor: 'text-green-400',
    },
    {
      label: 'Active Jobs',
      value: activeJobs.toString(),
      sub: null,
      icon: <Brain size={18} className="text-blue-400" />,
      valueColor: 'text-blue-400',
    },
    {
      label: 'Completed',
      value: completedJobs.toString(),
      sub: null,
      icon: <LayoutDashboard size={18} className="text-amber-400" />,
      valueColor: 'text-amber-400',
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
      <h1 className="text-2xl font-bold text-white mb-6">Dashboard</h1>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {statCards.map((card, i) => (
          <div key={i} className="card flex flex-col items-center text-center gap-2">
            <div className="flex items-center justify-center gap-2 mb-1">
              {card.icon}
            </div>
            <div className={`text-3xl font-bold ${card.valueColor}`}>
              {card.value}
            </div>
            <div className="text-xs text-gray-500">{card.label}</div>
            {card.sub && (
              <div className={`text-xs ${card.subColor} mt-0.5`}>
                {card.sub}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Ray Cluster + Storage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8 items-stretch">
        {/* Ray Cluster Status */}
        <div className="card flex flex-col">
          <h3 className="font-semibold text-white mb-4">Ray Cluster Status</h3>
          <div className="space-y-4 flex-1">
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-gray-400">GPU Utilization</span>
                <span className="text-indigo-400 font-medium">{gpuUtilPct}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${gpuUtilPct}%`, background: 'linear-gradient(90deg, #6366f1, #34d399)' }}
                />
              </div>
              <div className="text-xs text-gray-600 mt-1">{cluster?.gpus ?? '?'} / {cluster?.total_gpus ?? '?'} GPUs</div>
            </div>
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-gray-400">Memory Usage</span>
                <span className="text-indigo-400 font-medium">{memUtilPct}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${memUtilPct}%` }}
                />
              </div>
              <div className="text-xs text-gray-600 mt-1">
                {cluster ? `${Math.round(cluster.memory_gb)} GB / ${Math.round(cluster.memory_total_gb)} GB` : 'Loading...'}
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-800">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <div className={`w-2 h-2 rounded-full ${cluster ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <span>Head: 100.68.53.118 {cluster ? `| ${cluster.total_cpus} CPUs, ${cluster.total_gpus} GPUs` : '| Connecting...'}</span>
            </div>
          </div>
        </div>

        {/* Storage */}
        <div className="card flex flex-col">
          <h3 className="font-semibold text-white mb-4">Storage</h3>
          <div className="space-y-4 flex-1">
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-gray-400">MinIO Storage</span>
                <span className="text-cyan-400 font-medium">
                  {storageUsedGB.toFixed(1)} GB / {storageTotalGB} GB
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${storagePct}%`, background: 'linear-gradient(90deg, #06b6d4, #22d3ee)' }}
                />
              </div>
            </div>
            {storage.length > 0 ? (
              <div className="grid grid-cols-2 gap-2 text-xs">
                {storage.slice(0, 4).map((b) => (
                  <div key={b.bucket} className="bg-gray-900 rounded-lg p-2">
                    <div className="text-gray-500 text-xs mb-1 truncate">{b.bucket}</div>
                    <div className="text-indigo-400 font-medium">{(b.size_mb / 1024).toFixed(2)} GB</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-gray-600">No buckets found — create a project first</div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card flex flex-col">
        <h3 className="font-semibold text-white mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <a href="/datasets" className="btn btn-secondary flex-col items-center justify-center py-4 gap-1.5">
            <Database size={18} />
            <span className="text-xs mt-1">Manage Datasets</span>
          </a>
          <a href="/train" className="btn btn-primary flex-col items-center justify-center py-4 gap-1.5">
            <Brain size={18} />
            <span className="text-xs mt-1">New Training</span>
          </a>
          <a href="/jobs" className="btn btn-secondary flex-col items-center justify-center py-4 gap-1.5">
            <Activity size={18} />
            <span className="text-xs mt-1">View Jobs</span>
          </a>
          <a
            href="http://100.68.53.118:8265"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary flex-col items-center justify-center py-4 gap-1.5"
          >
            <LayoutDashboard size={18} />
            <span className="text-xs mt-1">Ray Dashboard</span>
          </a>
        </div>
      </div>
    </div>
  )
}
