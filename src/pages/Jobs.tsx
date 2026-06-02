
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
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>Training Jobs</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>ติดตามสถานะ training jobs บน Ray cluster</p>
        </div>
        <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
          <div style={{ display: 'flex', gap: 2, background: 'var(--bg-elevated)', borderRadius: 8, padding: 4 }}>
            {['all', 'running', 'completed', 'error', 'queued'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: '5px 10px', borderRadius: 6, fontSize: 12, fontWeight: 500,
                  transition: 'all 0.12s ease', border: 'none', cursor: 'pointer',
                  background: filter === f ? 'var(--primary)' : 'transparent',
                  color: filter === f ? '#fff' : 'var(--text-muted)',
                }}
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
        <div className="card mb-4" style={{ borderColor: 'var(--danger-dim)', background: 'var(--danger-dim)' }}>
          <p style={{ fontSize: 13, color: 'var(--danger)' }}>Failed to load jobs: {error}</p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center h-48">
          <Loader className="animate-spin" size={28} style={{ color: 'var(--primary)' }} />
        </div>
      )}

      {/* Jobs list */}
      {!loading && (
        <div className="space-y-3">
          {filtered.map((job: Job) => {
            const cfg = STATUS_CONFIG[job.status as keyof typeof STATUS_CONFIG] ?? STATUS_CONFIG.queued
            const Icon = cfg.icon
            return (
              <div key={job.id} className="card">
                {/* Header */}
                <div className="flex flex-col sm:flex-row items-start sm:items-start justify-between gap-3 mb-3">
                  <div className="flex items-start gap-3">
                    <Icon size={16} style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: 2 }} className={job.status === 'running' ? 'animate-spin' : ''} />
                    <div>
                      <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>{job.name}</h3>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 8px', marginTop: 4, fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
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
                  <div className="mb-3">
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
                      <span>Progress</span>
                      <span style={{ color: 'var(--primary-hover)', fontWeight: 500 }}>{job.progress}%</span>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-bar-fill" style={{ width: `${job.progress}%` }} />
                    </div>
                  </div>
                )}

                {/* Error message */}
                {job.status === 'error' && job.error && (
                  <div style={{ marginBottom: 12, background: 'var(--danger-dim)', borderRadius: 8, padding: '10px 12px' }}>
                    <div style={{ fontSize: 12, color: 'var(--danger)', fontWeight: 500, marginBottom: 4 }}>Error</div>
                    <div style={{ fontSize: 12, color: 'var(--danger)' }}>{job.error}</div>
                  </div>
                )}

                {/* Timing */}
                <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '4px 12px', fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 12px' }}>
                    <span>Queued: {formatTs(job.created_at)}</span>
                    {job.started_at && <span>Started: {formatTs(job.started_at)}</span>}
                    {job.finished_at && <span>Finished: {formatTs(job.finished_at)}</span>}
                  </div>
                  <span>Elapsed: {elapsed(job.started_at, job.finished_at)}</span>
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
                  <a
                    href="http://100.68.53.118:8265"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary btn-sm"
                  >
                    Ray Dashboard
                  </a>
                  {job.status === 'error' && (
                    <button className="btn btn-primary btn-sm">Retry</button>
                  )}
                  {job.status === 'completed' && (
                    <button className="btn btn-secondary btn-sm">View Logs</button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Empty state */}
      {!loading && filtered.length === 0 && !error && (
        <div style={{ textAlign: 'center', padding: '64px 0', color: 'var(--text-muted)' }}>
          <ListChecks size={40} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
          <p style={{ fontSize: 14 }}>{filter === 'all' ? 'ยังไม่มี training job — ไปที่ Train Model เพื่อสร้าง job' : `ไม่มี job ที่มีสถานะ '${filter}'`}</p>
          {filter !== 'all' && (
            <button className="btn btn-secondary" style={{ marginTop: 16 }} onClick={() => setFilter('all')}>
              แสดงทั้งหมด
            </button>
          )}
        </div>
      )}
    </div>
  )
}
