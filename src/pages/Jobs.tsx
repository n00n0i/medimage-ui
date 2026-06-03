
import { useState, useEffect, useRef } from 'react'
import { ListChecks, Clock, CheckCircle2, XCircle, Loader, RefreshCw, Terminal, Trash2, CheckSquare, Square } from 'lucide-react'
import GpuMonitor from '../components/GpuMonitor'

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
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [deletingBulk, setDeletingBulk] = useState(false)
  const [logJob, setLogJob] = useState<{id: string; name: string} | null>(null)
  const [logText, setLogText] = useState<string>('')
  const [logLoading, setLogLoading] = useState(false)
  const logRef = useRef<HTMLPreElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchJobs = async (quiet = false) => {
    if (!quiet) setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/jobs')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setJobs(data.jobs || [])
    } catch (e: any) {
      setError(e.message)
    } finally {
      if (!quiet) setLoading(false)
    }
  }

  // Auto-refresh when any job is running
  useEffect(() => {
    fetchJobs()
    pollRef.current = setInterval(() => {
      fetchJobs(true)
    }, 3000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const fetchLog = async (jobId: string) => {
    setLogLoading(true)
    try {
      const res = await fetch(`/api/jobs/${jobId}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setLogText(data.log || '(no output yet)')
      setTimeout(() => { logRef.current?.scrollTo(0, logRef.current.scrollHeight) }, 50)
    } catch (e: any) {
      setLogText(`Error: ${e.message}`)
    } finally {
      setLogLoading(false)
    }
  }

  const openLog = (job: Job) => {
    setLogJob({ id: job.id, name: job.name })
    fetchLog(job.id)
  }

  const deleteJob = async (jobId: string) => {
    await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' })
    setSelected(prev => { const s = new Set(prev); s.delete(jobId); return s })
    fetchJobs(true)
  }

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const s = new Set(prev)
      s.has(id) ? s.delete(id) : s.add(id)
      return s
    })
  }

  const toggleSelectAll = () => {
    const ids = filtered.map(j => j.id)
    const allSelected = ids.every(id => selected.has(id))
    setSelected(allSelected ? new Set() : new Set(ids))
  }

  const deleteSelected = async () => {
    setDeletingBulk(true)
    await Promise.all([...selected].map(id => fetch(`/api/jobs/${id}`, { method: 'DELETE' })))
    setSelected(new Set())
    setDeletingBulk(false)
    fetchJobs(true)
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
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>ติดตามสถานะ training jobs · auto-refresh ทุก 3s</p>
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
          <button className="btn btn-secondary flex items-center gap-1 whitespace-nowrap" onClick={() => fetchJobs()}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Bulk action toolbar */}
      {selected.size > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px',
          background: 'var(--bg-elevated)', borderRadius: 10, marginBottom: 16,
          border: '1px solid var(--primary)', boxShadow: '0 0 0 3px var(--primary-dim, #6366f120)',
        }}>
          <CheckSquare size={16} color="var(--primary)" />
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
            เลือกไว้ {selected.size} job
          </span>
          <button className="btn btn-secondary btn-sm" onClick={() => setSelected(new Set())}>ยกเลิก</button>
          <button
            className="btn btn-sm"
            style={{ background: 'var(--danger)', color: '#fff', border: 'none', display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={deleteSelected}
            disabled={deletingBulk}
          >
            {deletingBulk ? <Loader size={13} className="animate-spin" /> : <Trash2 size={13} />}
            Delete {selected.size}
          </button>
        </div>
      )}

      {/* Live GPU Monitor — show when a job is running */}
      {jobs.some(j => j.status === 'running' || j.status === 'queued') && (
        <div className="card mb-6" style={{ padding: '16px 20px' }}>
          <GpuMonitor active={jobs.some(j => j.status === 'running')} />
        </div>
      )}

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
          {/* Select-all row */}
          {filtered.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingBottom: 8 }}>
              <button
                onClick={toggleSelectAll}
                style={{ background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: 12 }}
              >
                {filtered.every(j => selected.has(j.id))
                  ? <CheckSquare size={15} color="var(--primary)" />
                  : <Square size={15} />}
                เลือกทั้งหมด ({filtered.length})
              </button>
            </div>
          )}

          {filtered.map((job: Job) => {
            const cfg = STATUS_CONFIG[job.status as keyof typeof STATUS_CONFIG] ?? STATUS_CONFIG.queued
            const Icon = cfg.icon
            const isSelected = selected.has(job.id)
            return (
              <div key={job.id} className="card" style={{ outline: isSelected ? '2px solid var(--primary)' : 'none', outlineOffset: -2 }}>
                {/* Header */}
                <div className="flex flex-col sm:flex-row items-start sm:items-start justify-between gap-3 mb-3">
                  <div className="flex items-start gap-3">
                    <button
                      onClick={() => toggleSelect(job.id)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', flexShrink: 0, marginTop: 1, padding: 0, color: isSelected ? 'var(--primary)' : 'var(--text-muted)' }}
                    >
                      {isSelected ? <CheckSquare size={16} /> : <Square size={16} />}
                    </button>
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
                  <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6 }} onClick={() => openLog(job)}>
                    <Terminal size={13} /> Logs
                  </button>
                  <a
                    href="http://100.68.53.118:8265"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary btn-sm"
                  >
                    Ray Dashboard
                  </a>
                  {job.status !== 'running' && (
                    <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--danger)' }} onClick={() => deleteJob(job.id)}>
                      <Trash2 size={13} /> Delete
                    </button>
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
      {/* Log Modal */}
      {logJob && (
        <div className="modal-overlay" onClick={() => setLogJob(null)}>
          <div className="modal" style={{ maxWidth: 800, width: '95vw' }} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div>
                <h3 style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-primary)' }}>Training Logs</h3>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>{logJob.name}</p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-secondary btn-sm" onClick={() => fetchLog(logJob.id)}>
                  {logLoading ? <Loader size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => setLogJob(null)}>✕</button>
              </div>
            </div>
            <pre
              ref={logRef}
              style={{
                background: 'var(--bg-base)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 8,
                padding: '12px 16px',
                fontSize: 12,
                lineHeight: 1.6,
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-mono)',
                maxHeight: '60vh',
                overflowY: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
              }}
            >
              {logLoading && !logText ? 'Loading...' : logText || '(no output yet)'}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
