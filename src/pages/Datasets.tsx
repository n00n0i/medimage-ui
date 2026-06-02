import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  Database, RefreshCw, ExternalLink, CheckCircle2, XCircle,
  AlertCircle, Loader2, HardDrive, FolderSync,
} from 'lucide-react'

const LS_API   = '/api/ls'
const LS_TOKEN = 'medimage-ls-token-2026'
const LS_BASE  = 'http://localhost:8085'

interface LSProject {
  id: number
  title: string
  task_number: number
  finished_task_number: number
}

interface LSStorage {
  id: number
  title: string
  bucket: string
  status: string
  last_sync: string | null
  last_sync_count: number | null
}

interface Dataset {
  projectId: number
  projectTitle: string
  storages: LSStorage[]
  taskCount: number
  labeledCount: number
  lastSync: string | null
  syncStatus: string
}

async function fetchLSCreds(): Promise<{ email: string; password: string } | null> {
  try {
    const r = await fetch('/api/ls-config')
    return r.ok ? await r.json() : null
  } catch { return null }
}

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

export default function Datasets() {
  const [loading,  setLoading]  = useState(true)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [syncing,  setSyncing]  = useState<Set<number>>(new Set())
  const [toast,    setToast]    = useState<{ msg: string; type: string } | null>(null)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  const fetchDatasets = useCallback(async () => {
    setLoading(true)
    try {
      const projRes = await axios.get(`${LS_API}/projects/`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
        params: { page_size: 100 },
      })
      const projects: LSProject[] = projRes.data?.results ?? (Array.isArray(projRes.data) ? projRes.data : [])

      const results = await Promise.all(projects.map(async (p): Promise<Dataset> => {
        try {
          const storRes = await axios.get(`${LS_API}/storages/s3`, {
            headers: { Authorization: `Token ${LS_TOKEN}` },
            params: { project: p.id },
          })
          const storages: LSStorage[] = Array.isArray(storRes.data) ? storRes.data : []
          const sorted   = [...storages].sort((a, b) => (b.last_sync ?? '').localeCompare(a.last_sync ?? ''))
          const statuses = storages.map(s => s.status)
          return {
            projectId:    p.id,
            projectTitle: p.title,
            storages,
            taskCount:    p.task_number           ?? 0,
            labeledCount: p.finished_task_number  ?? 0,
            lastSync:     sorted[0]?.last_sync    ?? null,
            syncStatus:   statuses.includes('started')  ? 'started'
                        : statuses.includes('failed')   ? 'failed'
                        : statuses.every(s => s === 'completed') && statuses.length > 0 ? 'completed'
                        : 'unknown',
          }
        } catch {
          return {
            projectId: p.id, projectTitle: p.title, storages: [],
            taskCount: p.task_number ?? 0, labeledCount: p.finished_task_number ?? 0,
            lastSync: null, syncStatus: 'unknown',
          }
        }
      }))

      setDatasets(results)
    } catch (e: any) {
      showToast(`Error: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchDatasets() }, [fetchDatasets])

  const openInLS = useCallback(async (projectId: number) => {
    // Use unique name to avoid cross-origin reuse of a previous ls-N popup.
    const win = window.open('about:blank', `ls-${projectId}-${Date.now()}`)
    if (!win) return
    try {
      const creds = await fetchLSCreds()
      if (!creds?.password) {
        win.location.href = `${LS_BASE}/projects/${projectId}/settings/`
        return
      }
      // Step 1: GET login page via nginx proxy (same-origin) → sets csrftoken cookie for localhost
      const html = await fetch('/api/ls-login/').then(r => r.text())
      const csrf = html.match(/name="csrfmiddlewaretoken"\s+value="([^"]+)"/)?.[1]
      if (!csrf) throw new Error('CSRF not found')
      // Step 2: POST login via nginx proxy (same-origin) → sets sessionid cookie for localhost
      // Cookies are host-only for 'localhost' (port-agnostic per RFC 6265),
      // so sessionid will be sent to localhost:8085 as well.
      await fetch('/api/ls-login/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          csrfmiddlewaretoken: csrf,
          email: creds.email,
          password: creds.password,
        }).toString(),
      })
      // Step 3: Navigate popup to LS — sessionid cookie is valid for all localhost ports
      win.location.href = `${LS_BASE}/projects/${projectId}/settings/`
    } catch {
      win.location.href = `${LS_BASE}/projects/${projectId}/settings/`
    }
  }, [])

  const syncAll = async (ds: Dataset) => {
    if (ds.storages.length === 0) { showToast('No storage to sync', 'info'); return }
    setSyncing(prev => new Set([...prev, ds.projectId]))
    try {
      await Promise.all(ds.storages.map(s =>
        axios.post(`${LS_API}/storages/s3/${s.id}/sync`, {},
          { headers: { Authorization: `Token ${LS_TOKEN}` } }
        ).catch(() => {})
      ))
      showToast(`Syncing ${ds.storages.length} storage(s) for "${ds.projectTitle}"`)
      setTimeout(fetchDatasets, 3000)
    } finally {
      setSyncing(prev => { const s = new Set(prev); s.delete(ds.projectId); return s })
    }
  }

  const fmt = (d: string | null) =>
    d ? new Date(d).toLocaleString('th-TH', {
      day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
    }) : '—'

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner" />
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>Datasets</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            Label Studio annotation projects — datasets backed by MinIO S3
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={fetchDatasets}>
          <RefreshCw size={14} />Refresh
        </button>
      </div>

      {toast && (
        <div className={`toast ${toast.type === 'error' ? 'toast-error' : 'toast-success'} mb-4`}>
          {toast.msg}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {datasets.map((ds) => {
          const isSyncing = syncing.has(ds.projectId)
          return (
            <div key={ds.projectId} className="card" style={{ display: 'flex', flexDirection: 'column' }}>
              {/* Header */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1, marginRight: 10 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <Database size={15} style={{ color: 'var(--primary)' }} />
                  </div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', margin: 0 }}>
                    {ds.projectTitle}
                  </h3>
                </div>
                <StatusBadge status={ds.syncStatus} />
              </div>

              {/* Bucket chips — full width 2 column grid */}
              {ds.storages.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 10 }}>
                  {[...new Set(ds.storages.map(s => s.bucket))].map(b => (
                    <span key={b} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--primary-hover)', background: 'var(--primary-dim)', border: '1px solid var(--primary-border)', borderRadius: 4, padding: '3px 8px', overflow: 'hidden', minWidth: 0 }}>
                      <HardDrive size={9} style={{ flexShrink: 0 }} />
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{b}</span>
                    </span>
                  ))}
                </div>
              )}

              {/* Stats */}
              <div style={{ display: 'flex', gap: 20, paddingTop: 12, paddingBottom: 12, borderTop: '1px solid var(--border-subtle)', borderBottom: '1px solid var(--border-subtle)', marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{ds.taskCount.toLocaleString()}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Images</div>
                </div>
                <div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--success)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{ds.labeledCount.toLocaleString()}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Labeled</div>
                </div>
                {ds.taskCount > 0 && (
                  <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>
                      {Math.round((ds.labeledCount / ds.taskCount) * 100)}%
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Complete</div>
                  </div>
                )}
              </div>

              {/* Last sync */}
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, flex: 1 }}>
                Last sync: {fmt(ds.lastSync)}
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', gap: 8 }}>
                {ds.storages.length > 0 && (
                  <button
                    className="btn btn-primary btn-sm"
                    style={{ flex: 1 }}
                    onClick={() => syncAll(ds)}
                    disabled={isSyncing || ds.syncStatus === 'started'}
                  >
                    {isSyncing
                      ? <><Loader2 size={12} className="animate-spin" />Syncing…</>
                      : <><FolderSync size={12} />Sync</>
                    }
                  </button>
                )}
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ flex: 1 }}
                  onClick={() => openInLS(ds.projectId)}
                >
                  <ExternalLink size={12} />Open in LS
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {datasets.length === 0 && (
        <div style={{ textAlign: 'center', padding: '64px 0', color: 'var(--text-muted)' }}>
          <Database size={40} style={{ margin: '0 auto 16px', opacity: 0.4, display: 'block' }} />
          <p style={{ fontSize: 14 }}>No datasets yet — create a project to get started</p>
          <a href="/projects" style={{ color: 'var(--primary-hover)', fontSize: 13, marginTop: 8, display: 'inline-block' }}>
            Go to Projects →
          </a>
        </div>
      )}
    </div>
  )
}
