import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { listBuckets, createBucket as s3CreateBucket } from '../lib/minioClient'
import {
  FolderSync, Plus, RefreshCw, ChevronDown, ChevronUp,
  Database, ExternalLink, CheckCircle2, XCircle, AlertCircle,
  Clock, Loader2, HardDrive, Trash2, ImageIcon, Layers,
} from 'lucide-react'

const LS_API   = '/api/ls'
const LS_TOKEN = 'medimage-ls-token-2026'

function deriveBucket(title: string, fallback: string) {
  return title.replace(/-bucket$/i, '').toLowerCase() || fallback
}

interface RawStorage {
  id: number
  title: string
  bucket: string
  prefix: string
  status: string
  last_sync: string | null
  last_sync_count: number | null
}
interface Storage extends RawStorage { derivedBucket: string }

interface Project {
  id: number
  title: string
  storages: Storage[]
  uniqueStorages: Storage[]
  taskCount: number
  lastSync: string | null
  syncStatus: string
}

interface Toast { id: number; msg: string; type: 'success' | 'error' | 'info' }

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    completed: { cls: 'badge badge-success', icon: <CheckCircle2 size={11} />, label: 'Synced'  },
    failed:    { cls: 'badge badge-danger',  icon: <XCircle size={11} />,      label: 'Failed'  },
    started:   { cls: 'badge badge-warning', icon: <Loader2 size={11} className="animate-spin" />, label: 'Syncing' },
    unknown:   { cls: 'badge badge-neutral', icon: <AlertCircle size={11} />,  label: 'No sync' },
  }
  const c = map[status] ?? map.unknown
  return <span className={c.cls}>{c.icon}{c.label}</span>
}

// Module-level cache: persists data across navigations so no flicker on re-visit
let _projectsCache: Project[] = []

export default function Projects() {
  const [projects,    setProjects]    = useState<Project[]>(_projectsCache)
  const [loading,     setLoading]     = useState(_projectsCache.length === 0)
  const [expanded,    setExpanded]    = useState<number | null>(null)
  const [toasts,      setToasts]      = useState<Toast[]>([])
  const [showCreate,  setShowCreate]  = useState(false)
  const [createForm,  setCreateForm]  = useState({ name: '', prefix: '' })
  const [creating,    setCreating]    = useState(false)
  const [syncingId,   setSyncingId]   = useState<number | null>(null)
  const [syncResults, setSyncResults] = useState<Record<number, { ok: boolean; msg?: string }>>({})
  const [minioBuckets,    setMinioBuckets]    = useState<string[]>([])
  const [bucketsLoading,  setBucketsLoading]  = useState(false)
  const [selectedBuckets, setSelectedBuckets] = useState<string[]>([])
  const [confirmDelete,   setConfirmDelete]   = useState<Project | null>(null)
  const [deleting,        setDeleting]        = useState(false)
  const [newBucketInline, setNewBucketInline] = useState('')
  const [creatingBucket,  setCreatingBucket]  = useState(false)

  const openInLS = useCallback((projectId: number) => {
    // Use <a target="_blank"> so VS Code built-in browser opens in external browser
    const a = document.createElement('a')
    a.href = `/api/ls-goto/${projectId}`
    a.target = '_blank'
    a.rel = 'noopener noreferrer'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }, [])

  const toast = useCallback((msg: string, type: Toast['type']) => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000)
  }, [])

  const enrichProject = useCallback(async (id: number, title: string): Promise<Project> => {
    const [storRes, taskRes] = await Promise.all([
      axios.get(`${LS_API}/storages/s3?project=${id}`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
      }).catch(() => ({ data: [] })),
      axios.get(`${LS_API}/tasks?project=${id}&page_size=1`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
      }).catch(() => ({ data: { count: 0 } })),
    ])
    const storages: Storage[] = (storRes.data as RawStorage[]).map(s => ({
      ...s, derivedBucket: deriveBucket(s.title, s.bucket),
    }))
    const seen = new Set<string>()
    const uniqueStorages = storages.filter(s => {
      if (seen.has(s.derivedBucket)) return false
      seen.add(s.derivedBucket)
      return true
    })
    return {
      id, title, storages, uniqueStorages,
      taskCount:  taskRes.data.count || 0,
      lastSync:   storages[0]?.last_sync  || null,
      syncStatus: storages[0]?.status     || 'unknown',
    }
  }, [])

  const fetchProjects = useCallback(async () => {
    if (_projectsCache.length === 0) setLoading(true)
    try {
      const lsRes = await axios.get(`${LS_API}/projects/`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
      })
      const raw: any[] = lsRes.data?.results ?? (Array.isArray(lsRes.data) ? lsRes.data : [])
      const list: Project[] = await Promise.all(
        raw.map((p: any) => enrichProject(p.id, p.title || `Project ${p.id}`))
      )
      _projectsCache = list
      setProjects(list)
    } catch (e: any) {
      toast(`Error loading projects: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [enrichProject, toast])

  useEffect(() => { fetchProjects() }, [fetchProjects])

  async function syncAll(project: Project) {
    if (project.uniqueStorages.length === 0) { toast('No storage to sync', 'info'); return }
    setSyncingId(project.id)
    setSyncResults({})
    const res: Record<number, { ok: boolean; msg?: string }> = {}
    for (const s of project.uniqueStorages) {
      try {
        await axios.post(`${LS_API}/storages/s3/${s.id}/sync`, {}, { headers: { Authorization: `Token ${LS_TOKEN}` } })
        res[s.id] = { ok: true }
      } catch (e: any) {
        res[s.id] = { ok: false, msg: e.message }
      }
      setSyncResults({ ...res })
    }
    const ok   = Object.values(res).filter(r => r.ok).length
    const fail = Object.values(res).filter(r => !r.ok).length
    toast(
      fail === 0 ? `Synced ${ok} storage${ok !== 1 ? 's' : ''} ✓` : `${ok} synced · ${fail} failed`,
      fail === 0 ? 'success' : 'error'
    )
    setSyncingId(null)
    setTimeout(fetchProjects, 2000)
  }

  async function syncOne(s: Storage) {
    try {
      await axios.post(`${LS_API}/storages/s3/${s.id}/sync`, {}, { headers: { Authorization: `Token ${LS_TOKEN}` } })
      toast(`"${s.derivedBucket}" synced ✓`, 'success')
      setTimeout(fetchProjects, 2000)
    } catch (e: any) {
      toast(`Sync failed: ${e.message}`, 'error')
    }
  }

  async function handleDelete(project: Project) {
    setDeleting(true)
    try {
      // Delete all S3 storages first
      for (const s of project.storages) {
        await axios.delete(`${LS_API}/storages/s3/${s.id}/`, {
          headers: { Authorization: `Token ${LS_TOKEN}` },
        }).catch(() => {})
      }
      // Delete the project
      await axios.delete(`${LS_API}/projects/${project.id}/`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
      })
      toast(`Project "${project.title}" deleted`, 'success')
      setConfirmDelete(null)
      setExpanded(null)
      fetchProjects()
    } catch (e: any) {
      toast('Delete failed: ' + (e.response?.data?.detail || e.message), 'error')
    } finally {
      setDeleting(false)
    }
  }

  async function handleCreate() {
    const { name, prefix } = createForm
    if (!name.trim() || selectedBuckets.length === 0) { toast('Project name and at least one bucket are required', 'error'); return }
    setCreating(true)
    try {
      toast('Creating Label Studio project…', 'info')
      const projRes = await axios.post(
        `${LS_API}/projects/`,
        { title: name.trim(), organization: 1 },
        { headers: { Authorization: `Token ${LS_TOKEN}`, 'Content-Type': 'application/json' } }
      )
      const newId = projRes.data.id
      toast(`Attaching ${selectedBuckets.length} bucket(s)…`, 'info')
      for (const bucket of selectedBuckets) {
        await axios.post(
          `${LS_API}/storages/s3/`,
          {
            project: newId,
            title:   `${bucket}-bucket`,
            bucket,  prefix: prefix || '',
            s3_endpoint: 'http://minio:9000', region_name: 'us-east-1',
            aws_access_key_id: 'minioadmin', aws_secret_access_key: 'minioadmin',
            use_blob_urls: true, presign: true,
          },
          { headers: { Authorization: `Token ${LS_TOKEN}`, 'Content-Type': 'application/json' } }
        )
      }
      toast(`Project "${name}" created ✓`, 'success')
      setShowCreate(false)
      setCreateForm({ name: '', prefix: '' })
      setSelectedBuckets([])
      setTimeout(fetchProjects, 1000)
    } catch (e: any) {
      const errDetail = e.response?.data?.detail
        || (e.response?.data ? JSON.stringify(e.response.data) : null)
        || e.message
      toast('Error: ' + errDetail, 'error')
    } finally {
      setCreating(false)
    }
  }

  async function handleInlineCreateBucket() {
    const name = newBucketInline.trim()
    if (!name) return
    setCreatingBucket(true)
    try {
      await s3CreateBucket(name)
      setMinioBuckets(prev => [...prev, name])
      setSelectedBuckets(prev => [...prev, name])
      setNewBucketInline('')
    } catch (e: any) {
      toast('Failed to create bucket: ' + (e?.message ?? 'unknown error'), 'error')
    } finally {
      setCreatingBucket(false)
    }
  }

  const fmt = (d: string | null) =>
    d ? new Date(d).toLocaleString('th-TH', {
      day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
    }) : '—'

  const totalImages = projects.reduce((s, p) => s + p.taskCount, 0)
  const syncedCount = projects.filter(p => p.syncStatus === 'completed').length

  return (
    <div style={{ maxWidth: '100%' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>Projects</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 3 }}>Label Studio annotation projects · synced from MinIO</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-secondary btn-sm" onClick={fetchProjects} disabled={loading}><RefreshCw size={14} />Refresh</button>
          <button className="btn btn-primary btn-sm" onClick={() => {
            setShowCreate(true)
            setSelectedBuckets([])
            setNewBucketInline('')
            setMinioBuckets([])
            setBucketsLoading(true)
            listBuckets().then(bs => {
              setMinioBuckets(bs.map(b => b.name))
            }).catch(() => {}).finally(() => setBucketsLoading(false))
          }}><Plus size={14} />New Project</button>
        </div>
      </div>

      {/* Stat cards */}
      {!loading && projects.length > 0 && (
        <div className="grid grid-cols-3 gap-4" style={{ marginBottom: 24 }}>
          <div className="stat-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', marginBottom: 6 }}>
              <Layers size={15} />
              <span className="stat-label">Projects</span>
            </div>
            <div className="stat-value">{projects.length}</div>
          </div>
          <div className="stat-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', marginBottom: 6 }}>
              <ImageIcon size={15} />
              <span className="stat-label">Total Images</span>
            </div>
            <div className="stat-value">{totalImages.toLocaleString()}</div>
          </div>
          <div className="stat-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', marginBottom: 6 }}>
              <CheckCircle2 size={15} />
              <span className="stat-label">Synced</span>
            </div>
            <div className="stat-value">
              <span style={{ color: syncedCount === projects.length ? 'var(--success)' : 'var(--text-primary)' }}>
                {syncedCount}
              </span>
              <span style={{ fontSize: 16, color: 'var(--text-muted)', fontWeight: 500 }}>/{projects.length}</span>
            </div>
          </div>
        </div>
      )}

      {/* Skeleton — only on first load */}
      {loading && projects.length === 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {[1, 2, 3].map(i => (
            <div key={i} className="card" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                <div className="skeleton" style={{ width: 40, height: 40, borderRadius: 10, flexShrink: 0 }} />
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div className="skeleton" style={{ height: 13, width: 160 }} />
                  <div className="skeleton" style={{ height: 10, width: 80 }} />
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
                <div className="skeleton" style={{ height: 22, width: 90, borderRadius: 4 }} />
                <div className="skeleton" style={{ height: 22, width: 70, borderRadius: 4 }} />
              </div>
              <div className="skeleton" style={{ height: 32, borderRadius: 8 }} />
            </div>
          ))}
        </div>
      )}

      {/* Project grid */}
      {!loading && projects.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {projects.map(p => {
            const isExpanded = expanded === p.id
            const isSyncing  = syncingId === p.id
            return (
              <div key={p.id} className="card" style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>

                {/* Card header */}
                <div style={{ padding: '18px 18px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0, flex: 1 }}>
                      <div style={{ width: 38, height: 38, borderRadius: 10, flexShrink: 0, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <FolderSync size={17} style={{ color: 'var(--primary)' }} />
                      </div>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.title}</div>
                        <div style={{ marginTop: 4 }}><StatusBadge status={p.syncStatus} /></div>
                      </div>
                    </div>
                    {/* Action icons */}
                    <div style={{ display: 'flex', gap: 2, flexShrink: 0, marginLeft: 8 }} onClick={e => e.stopPropagation()}>
                      <button
                        className="btn-icon"
                        onClick={() => setConfirmDelete(p)}
                        data-tooltip="Delete project"
                        style={{ color: 'var(--danger)', opacity: 0.65 }}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>

                  {/* Bucket chips — 2 column grid */}
                  {p.uniqueStorages.length > 0 && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 12 }}>
                      {p.uniqueStorages.map(s => (
                        <span key={s.derivedBucket} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--primary-hover)', background: 'var(--primary-dim)', border: '1px solid var(--primary-border)', borderRadius: 4, padding: '3px 8px', overflow: 'hidden', minWidth: 0 }}>
                          <HardDrive size={9} style={{ flexShrink: 0 }} />
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.derivedBucket}</span>
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Stats row */}
                  <div style={{ display: 'flex', gap: 20, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
                    <div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{p.taskCount.toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Images</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{p.uniqueStorages.length}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Buckets</div>
                    </div>
                    {p.lastSync && (
                      <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
                          <Clock size={10} />Last sync
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 3, fontVariantNumeric: 'tabular-nums' }}>{fmt(p.lastSync)}</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Sync progress */}
                {isSyncing && Object.keys(syncResults).length > 0 && (
                  <div style={{ padding: '0 16px 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {p.uniqueStorages.map(s => {
                      const r = syncResults[s.id]
                      if (!r) return null
                      return (
                        <div key={s.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', borderRadius: 6, background: 'var(--bg-elevated)' }}>
                          {r.ok
                            ? <CheckCircle2 size={12} style={{ color: 'var(--success)', flexShrink: 0 }} />
                            : <XCircle      size={12} style={{ color: 'var(--danger)',  flexShrink: 0 }} />
                          }
                          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{s.derivedBucket}</span>
                          <span style={{ fontSize: 11, marginLeft: 'auto', color: r.ok ? 'var(--success)' : 'var(--danger)' }}>{r.ok ? 'OK' : r.msg}</span>
                        </div>
                      )
                    })}
                  </div>
                )}

                {/* Expanded storage table — above footer */}
                {isExpanded && (
                  <div style={{ borderTop: '1px solid var(--border-subtle)', padding: '12px 14px', overflowX: 'auto' }}>
                    {p.uniqueStorages.length === 0 ? (
                      <div style={{ textAlign: 'center', padding: '20px 0', color: 'var(--text-muted)' }}>
                        <Database size={24} style={{ margin: '0 auto 8px', opacity: 0.25, display: 'block' }} />
                        <p style={{ fontSize: 12 }}>No storage connected</p>
                      </div>
                    ) : (
                      <table className="data-table" style={{ minWidth: 380 }}>
                        <thead>
                          <tr>
                            <th style={{ whiteSpace: 'nowrap' }}>Bucket</th>
                            <th style={{ whiteSpace: 'nowrap' }}>Status</th>
                            <th style={{ whiteSpace: 'nowrap' }}>Synced</th>
                            <th style={{ whiteSpace: 'nowrap' }}>Last sync</th>
                            <th></th>
                          </tr>
                        </thead>
                        <tbody>
                          {p.uniqueStorages.map(s => (
                            <tr key={s.derivedBucket}>
                              <td style={{ whiteSpace: 'nowrap' }}>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>{s.derivedBucket}</span>
                                {s.prefix && <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', marginLeft: 5 }}>/{s.prefix}</span>}
                              </td>
                              <td><StatusBadge status={s.status} /></td>
                              <td style={{ fontSize: 13, color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>{s.last_sync_count?.toLocaleString() ?? '—'}</td>
                              <td style={{ fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{fmt(s.last_sync)}</td>
                              <td>
                                <button className="btn btn-secondary btn-sm" onClick={() => syncOne(s)} disabled={s.status === 'started' || syncingId !== null}>
                                  {s.status === 'started'
                                    ? <><Loader2 size={11} className="animate-spin" />Syncing</>
                                    : <><RefreshCw size={11} />Sync</>
                                  }
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                )}

                {/* Footer buttons */}
                <div style={{ display: 'flex', gap: 8, padding: '12px 16px', marginTop: 'auto', borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}>
                  {p.uniqueStorages.length > 0 && (
                    <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={() => syncAll(p)} disabled={isSyncing || syncingId !== null}>
                      {isSyncing
                        ? <><Loader2 size={12} className="animate-spin" />Syncing…</>
                        : <><FolderSync size={12} />Sync All</>
                      }
                    </button>
                  )}
                  <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={() => openInLS(p.id)}>
                    <ExternalLink size={12} />Open in LS
                  </button>
                  <button
                    className="btn-icon"
                    onClick={() => setExpanded(isExpanded ? null : p.id)}
                    data-tooltip={isExpanded ? 'Hide details' : 'Show details'}
                    style={{ flexShrink: 0 }}
                  >
                    {isExpanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Empty state */}
      {!loading && projects.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '60px 24px' }}>
          <div style={{ width: 56, height: 56, borderRadius: 16, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
            <FolderSync size={24} style={{ color: 'var(--primary)' }} />
          </div>
          <h3 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>No projects yet</h3>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 320, margin: '0 auto 20px' }}>
            Create a project to start managing annotation datasets and syncing from MinIO.
          </p>
          <button className="btn btn-primary" onClick={() => {
            setShowCreate(true)
            setSelectedBuckets([])
            setNewBucketInline('')
            setMinioBuckets([])
            setBucketsLoading(true)
            listBuckets().then(bs => {
              setMinioBuckets(bs.map(b => b.name))
            }).catch(() => {}).finally(() => setBucketsLoading(false))
          }}><Plus size={15} />Create First Project</button>
        </div>
      )}

      {/* Confirm delete modal */}
      {confirmDelete && (
        <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget && !deleting) setConfirmDelete(null) }}>
          <div className="modal" style={{ maxWidth: 420 }}>
            <div style={{ padding: '24px 24px 20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                <div style={{ width: 40, height: 40, borderRadius: 10, background: 'rgba(255,180,201,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Trash2 size={18} style={{ color: 'var(--danger)' }} />
                </div>
                <div>
                  <h2 style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>Delete Project</h2>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>This action cannot be undone</p>
                </div>
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                Delete <strong style={{ color: 'var(--text-primary)' }}>{confirmDelete.title}</strong>
                {confirmDelete.storages.length > 0 && <> and <strong style={{ color: 'var(--text-primary)' }}>{confirmDelete.storages.length} dataset{confirmDelete.storages.length !== 1 ? 's' : ''}</strong> attached to it</>}?
              </p>
              {confirmDelete.storages.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
                  {confirmDelete.uniqueStorages.map(s => (
                    <span key={s.id} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--danger)', background: 'rgba(255,180,201,0.10)', border: '1px solid rgba(255,180,201,0.25)', borderRadius: 4, padding: '2px 7px' }}>
                      <HardDrive size={9} />{s.derivedBucket}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: 10, padding: '14px 24px', borderTop: '1px solid var(--border-subtle)' }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setConfirmDelete(null)} disabled={deleting}>Cancel</button>
              <button
                className="btn btn-sm"
                style={{ flex: 1, background: 'var(--danger)', color: '#fff', border: 'none', borderRadius: 8, fontWeight: 600, cursor: deleting ? 'not-allowed' : 'pointer', opacity: deleting ? 0.7 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}
                onClick={() => handleDelete(confirmDelete)}
                disabled={deleting}
              >
                {deleting ? <><Loader2 size={13} className="animate-spin" />Deleting…</> : <><Trash2 size={13} />Delete</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create modal */}
      {showCreate && (
        <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) setShowCreate(false) }}>
          <div className="modal">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 24px', borderBottom: '1px solid var(--border-subtle)' }}>
              <div>
                <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)' }}>New Project</h2>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>Creates a Label Studio project with S3 storage</p>
              </div>
              <button className="btn-icon" onClick={() => setShowCreate(false)}><XCircle size={18} /></button>
            </div>
            <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Project Name</label>
                <input type="text" placeholder="e.g. PCB Surface Defect" value={createForm.name}
                  onChange={e => setCreateForm(f => ({ ...f, name: e.target.value }))}
                  onKeyDown={e => e.key === 'Enter' && !creating && handleCreate()} />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', display: 'block', marginBottom: 8 }}>
                  MinIO Buckets
                  <span style={{ fontWeight: 400, marginLeft: 6 }}>(select one or more)</span>
                </label>
                {bucketsLoading ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 0', color: 'var(--text-muted)', fontSize: 13 }}>
                    <Loader2 size={13} className="animate-spin" />Loading buckets…
                  </div>
                ) : minioBuckets.length === 0 ? (
                  <div style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', fontSize: 13, color: 'var(--text-muted)' }}>
                    No buckets found. <a href="/storage" style={{ color: 'var(--primary)' }}>Create one on Storage page</a>.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {minioBuckets.map(b => {
                      const sel = selectedBuckets.includes(b)
                      return (
                        <button
                          key={b}
                          type="button"
                          onClick={() => setSelectedBuckets(prev =>
                            sel ? prev.filter(x => x !== b) : [...prev, b]
                          )}
                          style={{
                            display: 'inline-flex', alignItems: 'center', gap: 6,
                            fontSize: 12, fontFamily: 'var(--font-mono)',
                            padding: '5px 11px', borderRadius: 6, cursor: 'pointer',
                            border: sel ? '1.5px solid var(--primary)' : '1px solid var(--border-default)',
                            background: sel ? 'var(--primary-dim)' : 'var(--bg-elevated)',
                            color: sel ? 'var(--primary-hover)' : 'var(--text-secondary)',
                            transition: 'all 0.15s',
                          }}
                        >
                          <HardDrive size={11} />{b}
                          {sel && <CheckCircle2 size={11} style={{ marginLeft: 2 }} />}
                        </button>
                      )
                    })}
                  </div>
                )}
                {/* Inline new bucket creation */}
                <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
                  <input
                    type="text"
                    placeholder="new-bucket-name"
                    value={newBucketInline}
                    onChange={e => setNewBucketInline(e.target.value.toLowerCase().replace(/[^a-z0-9.-]/g, '-'))}
                    onKeyDown={async e => {
                      if (e.key === 'Enter' && newBucketInline.trim() && !creatingBucket) {
                        e.preventDefault()
                        await handleInlineCreateBucket()
                      }
                    }}
                    style={{ flex: 1, fontSize: 12, fontFamily: 'var(--font-mono)' }}
                  />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={!newBucketInline.trim() || creatingBucket}
                    onClick={handleInlineCreateBucket}
                    style={{ flexShrink: 0, whiteSpace: 'nowrap' }}
                  >
                    {creatingBucket ? <Loader2 size={11} className="animate-spin" /> : <Plus size={11} />}
                    New Bucket
                  </button>
                </div>
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Prefix <span style={{ fontWeight: 400 }}>(optional)</span></label>
                <input type="text" placeholder="e.g. images/" value={createForm.prefix}
                  onChange={e => setCreateForm(f => ({ ...f, prefix: e.target.value }))} />
              </div>
            </div>
            <div style={{ display: 'flex', gap: 10, padding: '14px 24px', borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setShowCreate(false)}>Cancel</button>
              <button className="btn btn-primary" style={{ flex: 1 }} onClick={handleCreate}
                disabled={creating || !createForm.name.trim() || selectedBuckets.length === 0}>
                {creating ? <><Loader2 size={14} className="animate-spin" />Creating…</> : <><Plus size={14} />Create Project</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toasts */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast-${t.type}`}>
            {t.type === 'success' && <CheckCircle2 size={15} />}
            {t.type === 'error'   && <XCircle      size={15} />}
            {t.type === 'info'    && <AlertCircle  size={15} />}
            {t.msg}
          </div>
        ))}
      </div>
    </div>
  )
}
