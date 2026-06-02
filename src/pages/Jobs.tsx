
import { useState, useEffect } from 'react'
import { ListChecks, Clock, CheckCircle2, XCircle, Loader, RefreshCw } from 'lucide-react'

interface Job {
  id: string
  name: string
  training_type: string
  model: string
  engine: string
  status: 'queued' | 'running' | 'completed' | 'error'
  progress: number
  created_at: number | null
  started_at: number | null
  finished_at: number | null
  error: string | null
  dataset: string
  epochs: number
  batch_size: number
}

const STATUS_CONFIG = {
  queued:    { icon: Clock,        color: 'text-gray-400',   badge: 'badge-warning', label: 'Queued'    },
  running:   { icon: Loader,       color: 'text-blue-400',   badge: 'badge-primary', label: 'Running'   },
  completed: { icon: CheckCircle2, color: 'text-green-400', badge: 'badge-success', label: 'Completed' },
  error:     { icon: XCircle,      color: 'text-red-400',   badge: 'badge-danger',  label: 'Failed'    },
}

function formatTs(ts: number | null): string {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString('th-TH', { dateStyle: 'short', timeStyle: 'short' })
}

function elapsed(startTs: number | null, finishTs: number | null): string {
  const end = finishTs ? finishTs * 1000 : Date.now()
  const start = startTs ? startTs * 1000 : Date.now()
  const ms = end - start
  const h = Math.floor(ms / 3600000)
  const m = Math.floor((ms % 3600000) / 60000)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

export default function Jobs() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<string>('all')

  const fetchJobs = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/jobs')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setJobs(data.jobs || [])
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchJobs() }, [])

  const filtered = filter === 'all'
    ? jobs
    : jobs.filter((j: Job) => j.status === filter)

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Training Jobs</h1>
          <p className="text-gray-400 text-sm mt-1">ติดตามสถานะ training jobs บน Ray cluster</p>
        </div>
        <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
          <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
            {['all', 'running', 'completed', 'error', 'queued'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  filter === f
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
          <button className="btn btn-secondary flex items-center gap-1 whitespace-nowrap" onClick={fetchJobs}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="card border border-red-900 bg-red-950/30 mb-4">
          <p className="text-red-400 text-sm">Failed to load jobs: {error}</p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center h-48">
          <Loader className="animate-spin text-indigo-400" size={32} />
        </div>
      )}

      {/* Jobs list */}
      {!loading && (
        <div className="space-y-4">
          {filtered.map((job: Job) => {
            const cfg = STATUS_CONFIG[job.status as keyof typeof STATUS_CONFIG] ?? STATUS_CONFIG.queued
            const Icon = cfg.icon
            return (
              <div key={job.id} className="card">
                {/* Header */}
                <div className="flex flex-col sm:flex-row items-start sm:items-start justify-between gap-3 mb-4">
                  <div className="flex items-start gap-3">
                    <Icon size={20} className={`${cfg.color} flex-shrink-0 mt-0.5 animate-spin`} style={job.status !== 'running' ? { animation: 'none' } : {}} />
                    <div>
                      <h3 className="font-semibold text-white">{job.name}</h3>
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-1 text-xs text-gray-500 font-mono">
                        <span>{job.id}</span>
                        <span>·</span>
                        <span>{job.model || job.training_type}</span>
                        {job.engine && <><span>·</span><span>{job.engine}</span></>}
                        <span>·</span>
                        <span>{job.epochs} epochs · {job.batch_size} batch</span>
                      </div>
                    </div>
                  </div>
                  <span className={`badge ${cfg.badge} flex-shrink-0`}>{cfg.label}</span>
                </div>

                {/* Progress bar */}
                {job.status === 'running' && (
                  <div className="mb-4">
                    <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                      <span>Progress</span>
                      <span className="text-indigo-400 font-medium">{job.progress}%</span>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-bar-fill" style={{ width: `${job.progress}%` }} />
                    </div>
                  </div>
                )}

                {/* Error message */}
                {job.status === 'error' && job.error && (
                  <div className="mb-4 bg-red-950/50 border border-red-900 rounded-lg p-3">
                    <div className="text-xs text-red-400 font-medium mb-1">Error</div>
                    <div className="text-xs text-red-300">{job.error}</div>
                  </div>
                )}

                {/* Timing */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-xs text-gray-500 mb-4">
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                    <span>Queued: {formatTs(job.created_at)}</span>
                    {job.started_at && <span>Started: {formatTs(job.started_at)}</span>}
                    {job.finished_at && <span>Finished: {formatTs(job.finished_at)}</span>}
                  </div>
                  <span className="text-gray-400">Elapsed: {elapsed(job.started_at, job.finished_at)}</span>
                </div>

                {/* Actions */}
                <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-gray-800">
                  <a
                    href="http://100.68.53.118:8265"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary text-xs"
                  >
                    Ray Dashboard
                  </a>
                  {job.status === 'error' && (
                    <button className="btn btn-primary text-xs">Retry Job</button>
                  )}
                  {job.status === 'completed' && (
                    <button className="btn btn-secondary text-xs">View Logs</button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Empty state */}
      {!loading && filtered.length === 0 && !error && (
        <div className="text-center py-16 text-gray-500">
          <ListChecks size={48} className="mx-auto mb-4 opacity-50" />
          <p>{filter === 'all' ? 'ยังไม่มี training job — ไปที่ Train Model เพื่อสร้าง job' : `ไม่มี job ที่มีสถานะ '${filter}'`}</p>
          {filter !== 'all' && (
            <button className="btn btn-secondary mt-4" onClick={() => setFilter('all')}>
              แสดงทั้งหมด
            </button>
          )}
        </div>
      )}
    </div>
  )
}
