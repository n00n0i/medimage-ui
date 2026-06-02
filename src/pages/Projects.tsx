import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  FolderSync, Plus, RefreshCw, ChevronDown, ChevronUp,
  Database, ExternalLink, CheckCircle2, XCircle, AlertCircle,
  Clock, HardDrive, ArrowRight, Loader2, Image
} from 'lucide-react'

const LS_API = '/api/ls'
const LS_TOKEN = '160d2644f4d45f84cd09f8931d20891e52f5e4cf'

interface Storage {
  id: number
  title: string
  bucket: string
  prefix: string
  status: string
  last_sync: string | null
  last_sync_count: number
}

interface Project {
  id: number
  title: string
  ls_url?: string
  storages: Storage[]
  taskCount: number
  storageIds: number[]
  lastSync: string | null
  status: string
}

interface Toast { id: number; msg: string; type: 'success' | 'error' | 'info' }

// ─── Status Badge ───────────────────────────────────────────────
function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    completed: { cls: 'badge badge-success', icon: <CheckCircle2 size={11} />, label: 'Completed' },
    failed:    { cls: 'badge badge-danger',  icon: <XCircle size={11} />,       label: 'Failed' },
    started:   { cls: 'badge badge-warning', icon: <Loader2 size={11} className="animate-spin" />, label: 'Syncing' },
    unknown:   { cls: 'badge badge-neutral', icon: <AlertCircle size={11} />,   label: 'Unknown' },
  }
  const c = cfg[status] || cfg.unknown
  return (
    <span className={c.cls}>
      {c.icon}
      {c.label}
    </span>
  )
}

// ─── Sync Result Row ─────────────────────────────────────────────
function SyncResultRow({ sid, label, result }: { sid: number; label: string; result: { status: string; data?: any; msg?: string } | null }) {
  const ok = result?.status === 'ok'
  return (
    <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-[var(--bg-elevated)]">
      <div className="flex items-center gap-2.5">
        {ok
          ? <CheckCircle2 size={15} className="text-[var(--success)] flex-shrink-0" />
          : <XCircle size={15} className="text-[var(--danger)] flex-shrink-0" />
        }
        <div>
          <div className="text-sm font-medium text-[var(--text-primary)]">{label}</div>
          <div className="text-xs text-[var(--text-muted)] font-mono">Storage #{sid}</div>
        </div>
      </div>
      {result?.status === 'ok'
        ? <span className="text-xs text-[var(--success)] font-medium">{result.data?.status ?? 'OK'}</span>
        : <span className="text-xs text-[var(--danger)]">{result?.msg || 'Error'}</span>
      }
    </div>
  )
}

// ─── Main Component ──────────────────────────────────────────────
export default function Projects() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedProject, setExpandedProject] = useState<number | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [showCreate, setShowCreate] = useState(false)
  const [createForm, setCreateForm] = useState({ name: '', bucket: '', prefix: '' })
  const [creating, setCreating] = useState(false)
  const [syncingProject, setSyncingProject] = useState<number | null>(null)
  const [syncResults, setSyncResults] = useState<Record<number, { status: string; data?: any; msg?: string }>>({})

  const addToast = (msg: string, type: Toast['type']) => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4200)
  }

  useEffect(() => { fetchProjects() }, [])

  const fetchProjects = async () => {
    setLoading(true)
    try {
      // Get ls_url from backend, then enrich with LS storage/task data
      const backendRes = await axios.get('/api/projects').catch(() => ({ data: [] }))
      const backendProjects: any[] = Array.isArray(backendRes.data) ? backendRes.data : (backendRes.data.projects || [])

      const withDetails = await Promise.all(backendProjects.map(async (p: any) => {
        const [storRes, taskRes] = await Promise.all([
          axios.get(`${LS_API}/storages/s3?project=${p.id}`, {
            headers: { Authorization: `Token ${LS_TOKEN}` },
          }).catch(() => ({ data: [] })),
          axios.get(`${LS_API}/tasks?project=${p.id}&page_size=1`, {
            headers: { Authorization: `Token ${LS_TOKEN}` },
          }).catch(() => ({ data: { count: 0 } })),
        ])
        const storages = storRes.data
        const taskCount = taskRes.data.count || 0
        return {
          id: p.id,
          title: p.name,
          ls_url: p.ls_url || `http://100.68.221.236:8080/projects/${p.id}/settings`,
          storages,
          taskCount,
          storageIds: storages.map((s: any) => s.id),
          lastSync: storages[0]?.last_sync || null,
          status: storages[0]?.status || 'unknown',
        }
      }))

      if (withDetails.length === 0) {
        const [storRes, taskRes] = await Promise.all([
          axios.get(`${LS_API}/storages/s3?project=1`, {
            headers: { Authorization: `Token ${LS_TOKEN}` },
          }).catch(() => ({ data: [] })),
          axios.get(`${LS_API}/tasks?project=1&page_size=1`, {
            headers: { Authorization: `Token ${LS_TOKEN}` },
          }).catch(() => ({ data: { count: 0 } })),
        ])
        const storages = storRes.data
        withDetails.push({
          id: 1, title: 'Project 1', ls_url: 'http://100.68.221.236:8080/projects/1/settings', storages, taskCount: taskRes.data.count || 0,
          storageIds: storages.map((s: any) => s.id),
          lastSync: storages[0]?.last_sync || null,
          status: storages[0]?.status || 'unknown',
        })
      }

      setProjects(withDetails)
    } catch (e: any) {
      addToast(`Error: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const triggerSyncAll = async (project: Project) => {
    if (project.storageIds.length === 0) {
      addToast('No storage to sync', 'info')
      return
    }
    setSyncingProject(project.id)
    setSyncResults({})
    const results: Record<number, { status: string; data?: any; msg?: string }> = {}

    for (const sid of project.storageIds) {
      try {
        const res = await axios.post(`/api/sync/${sid}/`, {}, {
          headers: { Authorization: `Token ${LS_TOKEN}` },
        })
        results[sid] = { status: 'ok', data: res.data }
      } catch (e: any) {
        results[sid] = { status: 'error', msg: e.message }
      }
      setSyncResults({ ...results })
    }

    const ok = Object.values(results).filter(r => r.status === 'ok').length
    const fail = Object.values(results).filter(r => r.status === 'error').length
    addToast(
      fail === 0
        ? `Synced ${ok}/${project.storageIds.length} storage(s) successfully`
        : `${ok} synced, ${fail} failed`,
      fail === 0 ? 'success' : 'error'
    )
    setSyncingProject(null)
    setTimeout(fetchProjects, 2500)
  }

  const triggerSyncSingle = async (sid: number, label: string) => {
    try {
      await axios.post(`/api/sync/${sid}/`, {}, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
      })
      addToast(`Storage "${label}" synced ✓`, 'success')
      setTimeout(fetchProjects, 2500)
    } catch (e: any) {
      addToast(`Sync failed: ${e.message}`, 'error')
    }
  }

  const handleCreateProject = async () => {
    const { name, bucket, prefix } = createForm
    if (!name.trim() || !bucket.trim()) {
      addToast('Please enter project name and bucket', 'error')
      return
    }
    setCreating(true)
    try {
      addToast('Creating MinIO bucket...', 'info')
      await axios.post('http://100.68.221.236:9000/minio/upload/fname', null, {
        params: { bucket, prefix },
      }).catch(() => null)

      addToast('Creating Label Studio project...', 'info')
      const projRes = await axios.post(
        `${LS_API}/projects/`,
        { title: name.trim(), organization: 1 },
        { headers: { Authorization: `Token ${LS_TOKEN}`, 'Content-Type': 'application/json' } }
      )
      const newId = projRes.data.id

      addToast('Creating S3 storage...', 'info')
      await axios.post(
        `${LS_API}/storages/s3/`,
        {
          project: newId,
          title: name.trim(),
          bucket,
          prefix: prefix || '',
          s3_endpoint: 'http://100.68.221.236:9000',
          region_name: 'us-east-1',
          use_blob_urls: true,
          presign: true,
        },
        { headers: { Authorization: `Token ${LS_TOKEN}`, 'Content-Type': 'application/json' } }
      )

      addToast(`Project "${name}" created successfully`, 'success')
      setShowCreate(false)
      setCreateForm({ name: '', bucket: '', prefix: '' })
      setTimeout(fetchProjects, 1000)
    } catch (e: any) {
      addToast('Error: ' + (e.response?.data?.detail || e.message), 'error')
    } finally {
      setCreating(false)
    }
  }

  const formatDate = (d: string | null) =>
    d ? new Date(d).toLocaleString('th-TH', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'

  const totalImages = projects.reduce((sum, p) => sum + p.taskCount, 0)
  const completedSyncs = projects.filter(p => p.status === 'completed').length

  return (
    <div style={{ maxWidth: '100%' }}>

      {/* ── Header ──────────────────────────────────────────── */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>Projects</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>Manage datasets and sync from MinIO storage</p>
        </div>
        <div className="flex gap-3">
          <button className="btn btn-secondary btn-sm" onClick={fetchProjects}>
            <RefreshCw size={14} />
            Refresh
          </button>
          <button className="btn btn-primary btn-sm" onClick={() => setShowCreate(true)}>
            <Plus size={14} />
            New Project
          </button>
        </div>
      </div>

      {/* ── Stat Row ─────────────────────────────────────────── */}
      {!loading && projects.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="stat-card">
            <span className="stat-label">Projects</span>
            <span className="stat-value">{projects.length}</span>
            <span className="stat-sub">Total projects</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Total Images</span>
            <span className="stat-value">{totalImages.toLocaleString()}</span>
            <span className="stat-sub">Across all storages</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Completed Syncs</span>
            <span className="stat-value">{completedSyncs}<span style={{fontSize:'14px',fontWeight:500,color:'var(--text-muted)'}}>/{projects.length}</span></span>
            <span className="stat-sub">Storages synced</span>
          </div>
        </div>
      )}

      {/* ── Loading ──────────────────────────────────────────── */}
      {loading && (
        <div className="space-y-4">
          {[1,2,3].map(i => (
            <div key={i} className="card" style={{padding:'20px'}}>
              <div className="flex items-center justify-between">
                <div className="space-y-2 flex-1">
                  <div className="skeleton h-4 w-48" />
                  <div className="skeleton h-3 w-32" />
                </div>
                <div className="skeleton h-9 w-28 rounded-lg" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Project Cards Grid ───────────────────────────────────── */}
      {!loading && projects.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-8">
          {projects.map(p => (
            <div key={p.id} className="card flex flex-col" style={{ padding: 0, overflow: 'hidden', minHeight: 160 }}>

              {/* Card Header */}
              <div className="flex items-center justify-between p-5">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{background:'var(--primary-dim)'}}>
                    <FolderSync size={18} style={{color:'var(--primary)'}} />
                  </div>
                  <div className="min-w-0">
                    <h2 className="text-[var(--text-primary)] font-semibold text-[15px] truncate">
                      {p.title}
                    </h2>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
                        <HardDrive size={11} />
                        {p.storages.length} storage{p.storages.length !== 1 ? 's' : ''}
                      </span>
                      <span className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
                        <Image size={11} />
                        {p.taskCount.toLocaleString()} images
                      </span>
                      <span className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
                        <Clock size={11} />
                        {formatDate(p.lastSync)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0 ml-4">
                  <StatusBadge status={p.status} />
                  {p.storageIds.length > 0 && (
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => triggerSyncAll(p)}
                      disabled={syncingProject === p.id}
                    >
                      {syncingProject === p.id
                        ? <><Loader2 size={13} className="animate-spin" /> Syncing...</>
                        : <><FolderSync size={13} /> Sync All ({p.storageIds.length})</>
                      }
                    </button>
                  )}
                  <button
                    className="btn-icon"
                    onClick={() => setExpandedProject(expandedProject === p.id ? null : p.id)}
                    data-tooltip={expandedProject === p.id ? 'Collapse' : 'Show details'}
                  >
                    {expandedProject === p.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                </div>
              </div>

              {/* Sync Results */}
              {syncingProject === p.id && Object.keys(syncResults).length > 0 && (
                <div className="px-5 pb-4 space-y-2">
                  {p.storageIds.map(sid => {
                    const s = p.storages.find((s: any) => s.id === sid)
                    const label = s?.title || s?.bucket || `Storage ${sid}`
                    return (
                      <SyncResultRow key={sid} sid={sid} label={label} result={syncResults[sid]} />
                    )
                  })}
                </div>
              )}

              {/* Expanded Details */}
              {expandedProject === p.id && (
                <div className="border-t" style={{borderColor:'var(--border-subtle)'}}>
                  <div className="p-5">
                    {p.storages.length === 0 ? (
                      <div className="text-center py-8">
                        <Database size={32} className="mx-auto mb-3 opacity-30" />
                        <p className="text-sm text-[var(--text-muted)]">No storage connected</p>
                        <p className="text-xs text-[var(--text-muted)] mt-1">
                          Go to Label Studio to add an S3 storage
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Storage</th>
                              <th>Bucket / Prefix</th>
                              <th>Status</th>
                              <th>Synced</th>
                              <th>Last Sync</th>
                              <th></th>
                            </tr>
                          </thead>
                          <tbody>
                            {p.storages.map((s: any) => (
                              <tr key={s.id}>
                                <td>
                                  <span className="font-medium text-[var(--text-primary)]">
                                    {s.title || s.bucket}
                                  </span>
                                </td>
                                <td>
                                  <span className="font-mono text-xs text-[var(--text-muted)]">
                                    {s.bucket}{s.prefix ? `/${s.prefix}` : ''}
                                  </span>
                                </td>
                                <td><StatusBadge status={s.status} /></td>
                                <td className="text-xs text-[var(--text-secondary)]">
                                  {s.last_sync_count?.toLocaleString() ?? '—'}
                                </td>
                                <td className="text-xs text-[var(--text-muted)]">
                                  {formatDate(s.last_sync)}
                                </td>
                                <td>
                                  <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => triggerSyncSingle(s.id, s.title || s.bucket)}
                                    disabled={s.status === 'started' || syncingProject !== null}
                                  >
                                    {s.status === 'started'
                                      ? <><Loader2 size={11} className="animate-spin" /> Syncing</>
                                      : <><RefreshCw size={11} /> Sync</>
                                    }
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>

                        <div className="flex items-center justify-between pt-3 mt-3"
                          style={{borderTop:'1px solid var(--border-subtle)'}}>
                          <a
                            href={p.ls_url || `/ls/projects/${p.id}/settings`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 text-xs text-[var(--primary)] hover:text-[var(--primary-hover)] transition-colors"
                          >
                            <ExternalLink size={12} />
                            Open Label Studio
                            <ArrowRight size={11} />
                          </a>
                          <span className="text-xs text-[var(--text-muted)]">
                            Project ID: {p.id}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Empty State ──────────────────────────────────────── */}
      {!loading && projects.length === 0 && (
        <div className="card text-center py-16">
          <div className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center"
            style={{background:'var(--primary-dim)'}}>
            <FolderSync size={28} style={{color:'var(--primary)'}} />
          </div>
          <h3 className="text-[var(--text-primary)] font-semibold text-lg mb-2">
            No projects yet
          </h3>
          <p className="text-sm text-[var(--text-muted)] mb-6 max-w-sm mx-auto">
            Create your first project to start managing datasets and syncing data from MinIO
          </p>
          <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
            <Plus size={15} />
            Create First Project
          </button>
        </div>
      )}

      {/* ── Create Project Modal ─────────────────────────────── */}
      {showCreate && (
        <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) setShowCreate(false) }}>
          <div className="modal">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-5"
              style={{borderBottom:'1px solid var(--border-subtle)'}}>
              <div>
                <h2 className="text-[var(--text-primary)] font-bold text-lg">New Project</h2>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Creates Label Studio project + S3 storage + MinIO bucket
                </p>
              </div>
              <button className="btn-icon" onClick={() => setShowCreate(false)}>
                <XCircle size={18} />
              </button>
            </div>

            {/* Modal Body */}
            <div className="px-6 py-5 space-y-4">
              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Project Name</label>
                <input
                  type="text"
                  placeholder="e.g. CXR Chest X-Ray"
                  value={createForm.name}
                  onChange={e => setCreateForm(f => ({...f, name: e.target.value}))}
                  onKeyDown={e => e.key === 'Enter' && !creating && handleCreateProject()}
                />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>MinIO Bucket</label>
                <input
                  type="text"
                  placeholder="e.g. medimage-cxr"
                  value={createForm.bucket}
                  onChange={e => setCreateForm(f => ({...f, bucket: e.target.value}))}
                />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Prefix <span style={{ fontWeight: 400 }}>(optional)</span></label>
                <input
                  type="text"
                  placeholder="e.g. images/"
                  value={createForm.prefix}
                  onChange={e => setCreateForm(f => ({...f, prefix: e.target.value}))}
                />
              </div>

              {/* Info note */}
              <div className="flex gap-2.5 p-3 rounded-lg" style={{background:'var(--bg-elevated)',border:'1px solid var(--border-subtle)'}}>
                <AlertCircle size={14} style={{ color: 'var(--warning)', flexShrink: 0, marginTop: 2 }} />
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                  Make sure the MinIO bucket <span className="font-mono text-[var(--text-primary)]">"{createForm.bucket || '...'}"</span> already exists
                  before creating the project. You can create it via the{' '}
                  <a href="http://100.68.221.236:9001" target="_blank" rel="noopener noreferrer"
                    className="text-[var(--primary)] hover:underline">MinIO Console</a>.
                </p>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="flex gap-3 px-6 py-4"
              style={{borderTop:'1px solid var(--border-subtle)',background:'var(--bg-surface)'}}>
              <button className="btn btn-secondary flex-1" onClick={() => setShowCreate(false)}>
                Cancel
              </button>
              <button
                className="btn btn-primary flex-1"
                onClick={handleCreateProject}
                disabled={creating || !createForm.name.trim() || !createForm.bucket.trim()}
              >
                {creating
                  ? <><Loader2 size={14} className="animate-spin" /> Creating...</>
                  : <><Plus size={14} /> Create Project</>
                }
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Toast Container ──────────────────────────────────── */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast-${t.type}`}>
            {t.type === 'success' && <CheckCircle2 size={15} />}
            {t.type === 'error'   && <XCircle size={15} />}
            {t.type === 'info'    && <AlertCircle size={15} />}
            {t.msg}
          </div>
        ))}
      </div>
    </div>
  )
}
