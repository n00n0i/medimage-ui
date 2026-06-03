import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import {
  Database, RefreshCw, ExternalLink, CheckCircle2, XCircle,
  AlertCircle, Loader2, HardDrive, FolderSync, Upload, FileText, Trash2,
  Search, Package,
} from 'lucide-react'

interface HFDataset {
  id: string
  author?: string
  downloads?: number
  tags?: string[]
  gated?: boolean | string
}

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

interface TextDataset {
  id: string
  name: string
  format: string
  row_count: number
  size_bytes: number
  created_at: number
  preview?: any[]
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
  const [tab, setTab] = useState<'image' | 'text'>('image')
  const [textDatasets, setTextDatasets] = useState<TextDataset[]>([])
  const [textLoading, setTextLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [showHFImport, setShowHFImport] = useState(false)

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

  const fetchTextDatasets = useCallback(async () => {
    setTextLoading(true)
    try {
      const res = await fetch('/api/text-datasets')
      const data = await res.json()
      setTextDatasets(data.datasets ?? [])
    } catch { /* ignore */ }
    setTextLoading(false)
  }, [])

  useEffect(() => {
    if (tab === 'text') fetchTextDatasets()
  }, [tab, fetchTextDatasets])

  const handleFileUpload = async (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!['jsonl', 'json', 'txt', 'csv'].includes(ext ?? '')) {
      showToast('รองรับเฉพาะ .jsonl, .json, .txt, .csv', 'error')
      return
    }
    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch('/api/text-datasets/upload', { method: 'POST', body: fd })
      if (!res.ok) throw new Error(await res.text())
      showToast(`✅ อัปโหลด "${file.name}" สำเร็จ`)
      fetchTextDatasets()
    } catch (e: any) {
      showToast(`❌ อัปโหลดล้มเหลว: ${e.message}`, 'error')
    }
    setUploading(false)
  }

  const deleteTextDataset = async (id: string, name: string) => {
    if (!confirm(`ลบ dataset "${name}"?`)) return
    try {
      await fetch(`/api/text-datasets/${id}`, { method: 'DELETE' })
      showToast(`ลบ "${name}" แล้ว`)
      fetchTextDatasets()
    } catch {
      showToast('ลบไม่สำเร็จ', 'error')
    }
  }

  const fmtBytes = (b: number) =>
    b < 1024 ? `${b} B` : b < 1024 * 1024 ? `${(b/1024).toFixed(1)} KB` : `${(b/1024/1024).toFixed(1)} MB`

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

  return (
    <div className="max-w-5xl mx-auto">
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>Datasets</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            {tab === 'image' ? 'Label Studio annotation projects — datasets backed by MinIO S3' : 'Text & Instruction datasets for LLM/VLM fine-tuning'}
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={tab === 'image' ? fetchDatasets : fetchTextDatasets}>
          <RefreshCw size={14} />Refresh
        </button>
        <button
          className="btn btn-primary btn-sm"
          style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#f97316', borderColor: '#f97316' }}
          onClick={() => setShowHFImport(true)}
        >
          <Package size={14} /> Import from HuggingFace
        </button>
      </div>

      {/* Tab switcher */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, padding: '4px', background: 'var(--bg-elevated)', borderRadius: 10, width: 'fit-content' }}>
        <button
          onClick={() => setTab('image')}
          style={{
            padding: '6px 16px', borderRadius: 8, fontSize: 13, fontWeight: 500, border: 'none', cursor: 'pointer',
            background: tab === 'image' ? 'var(--bg-surface)' : 'transparent',
            color: tab === 'image' ? 'var(--text-primary)' : 'var(--text-muted)',
            boxShadow: tab === 'image' ? '0 1px 3px rgba(0,0,0,0.15)' : 'none',
          }}
        >
          <Database size={13} style={{ display: 'inline', marginRight: 6 }} />
          Image Datasets
        </button>
        <button
          onClick={() => setTab('text')}
          style={{
            padding: '6px 16px', borderRadius: 8, fontSize: 13, fontWeight: 500, border: 'none', cursor: 'pointer',
            background: tab === 'text' ? 'var(--bg-surface)' : 'transparent',
            color: tab === 'text' ? 'var(--text-primary)' : 'var(--text-muted)',
            boxShadow: tab === 'text' ? '0 1px 3px rgba(0,0,0,0.15)' : 'none',
          }}
        >
          <FileText size={13} style={{ display: 'inline', marginRight: 6 }} />
          Text Datasets
        </button>
      </div>

      {toast && (
        <div className={`toast ${toast.type === 'error' ? 'toast-error' : 'toast-success'} mb-4`}>
          {toast.msg}
        </div>
      )}

      {/* ── Text Datasets Tab ── */}
      {tab === 'text' && (
        <div>
          {/* Upload zone */}
          <div
            className="card mb-6"
            style={{
              borderStyle: 'dashed', borderWidth: 2, cursor: 'pointer',
              borderColor: dragOver ? '#8b5cf6' : 'var(--border)',
              background: dragOver ? 'rgba(139,92,246,0.06)' : 'var(--bg-surface)',
              textAlign: 'center', padding: '32px 16px', transition: 'all 0.15s',
            }}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => {
              e.preventDefault()
              setDragOver(false)
              const file = e.dataTransfer.files[0]
              if (file) handleFileUpload(file)
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jsonl,.json,.txt,.csv"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            />
            {uploading ? (
              <><Loader2 size={28} className="animate-spin" style={{ color: '#8b5cf6', margin: '0 auto 8px' }} />
              <p style={{ fontSize: 14, color: 'var(--text-muted)' }}>กำลังอัปโหลด...</p></>
            ) : (
              <><Upload size={28} style={{ color: '#8b5cf6', margin: '0 auto 8px' }} />
              <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-primary)', marginBottom: 4 }}>คลิกหรือลาก .jsonl file มาวางที่นี่</p>
              <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>รองรับ .jsonl (Alpaca / ShareGPT), .json, .txt, .csv</p></>
            )}
          </div>

          {/* Format hints */}
          <div className="card mb-6" style={{ background: 'var(--bg-elevated)' }}>
            <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>รูปแบบ .jsonl ที่รองรับ</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Alpaca format</p>
                    <pre style={{ fontSize: 11, background: 'var(--bg-surface)', borderRadius: 6, padding: '8px', color: 'var(--text-secondary)', overflow: 'auto' }}>{`{"instruction": "Describe the defect",\n "input": "",\n "output": "Surface scratch on panel A"}`}</pre>
              </div>
              <div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>ShareGPT / ChatML format</p>
                <pre style={{ fontSize: 11, background: 'var(--bg-surface)', borderRadius: 6, padding: '8px', color: 'var(--text-secondary)', overflow: 'auto' }}>{`{"conversations": [\n  {"role": "user", "content": "..."},\n  {"role": "assistant", "content": "..."}\n]}`}</pre>
              </div>
            </div>
          </div>

          {/* Dataset list */}
          {textLoading ? (
            <div className="flex items-center justify-center h-32"><div className="loading-spinner" /></div>
          ) : textDatasets.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', padding: '40px 16px' }}>
              <FileText size={36} style={{ color: 'var(--text-muted)', margin: '0 auto 12px' }} />
              <p style={{ fontSize: 14, color: 'var(--text-muted)' }}>ยังไม่มี text dataset</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {textDatasets.map(ds => (
                <div key={ds.id} className="card">
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(139,92,246,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <FileText size={15} style={{ color: '#8b5cf6' }} />
                      </div>
                      <div>
                        <h3 style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)', margin: 0 }}>{ds.name}</h3>
                        <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                          <span style={{ fontSize: 11, background: 'rgba(139,92,246,0.15)', color: '#8b5cf6', borderRadius: 4, padding: '1px 6px', fontWeight: 600 }}>{ds.format}</span>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{ds.row_count.toLocaleString()} rows</span>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>·</span>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmtBytes(ds.size_bytes)}</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => deleteTextDataset(ds.id, ds.name)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 4 }}
                      title="ลบ dataset นี้"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  {/* Preview rows */}
                  {ds.preview && ds.preview.length > 0 && (
                    <div style={{ marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Preview (3 rows)</p>
                      {ds.preview.slice(0, 3).map((row, i) => (
                        <pre key={i} style={{ fontSize: 10, background: 'var(--bg-elevated)', borderRadius: 4, padding: '4px 6px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '100%', marginBottom: 2 }}>
                          {JSON.stringify(row).slice(0, 100)}{JSON.stringify(row).length > 100 ? '...' : ''}
                        </pre>
                      ))}
                    </div>
                  )}
                  <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                    <a href="/train" className="btn btn-primary btn-sm" style={{ fontSize: 11 }}>Use in Training →</a>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>{fmt(new Date(ds.created_at * 1000).toISOString())}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Image Datasets Tab ── */}
      {tab === 'image' && (
        <>
        {loading ? (
          <div className="flex items-center justify-center h-64"><div className="loading-spinner" /></div>
        ) : datasets.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '64px 0', color: 'var(--text-muted)' }}>
            <Database size={40} style={{ margin: '0 auto 16px', opacity: 0.4, display: 'block' }} />
            <p style={{ fontSize: 14 }}>No datasets yet — create a project to get started</p>
            <a href="/projects" style={{ color: 'var(--primary-hover)', fontSize: 13, marginTop: 8, display: 'inline-block' }}>
              Go to Projects →
            </a>
          </div>
        ) : (
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
      )}
      </>
      )}

      {showHFImport && (
        <HFDatasetImportModal
          onClose={() => setShowHFImport(false)}
          onDone={() => { setShowHFImport(false); fetchDatasets() }}
        />
      )}
    </div>
  )
}

// ─── HuggingFace Dataset Import Modal ────────────────────────────────────────

function HFDatasetImportModal({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [hfQuery, setHfQuery] = useState('')
  const [hfResults, setHfResults] = useState<HFDataset[]>([])
  const [hfSearching, setHfSearching] = useState(false)
  const [hfSearched, setHfSearched] = useState(false)
  const [selected, setSelected] = useState<HFDataset | null>(null)
  const [bucketName, setBucketName] = useState('')
  const [bucketError, setBucketError] = useState('')
  const [maxFiles, setMaxFiles] = useState(20)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<any>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  const suggestBucket = (id: string) => {
    const base = id.split('/').pop() ?? id
    return 'hf-' + base.toLowerCase().replace(/[^a-z0-9]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '').slice(0, 27)
  }

  const validateBucket = (name: string): string => {
    if (!name) return 'Required'
    if (!/^[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]$/.test(name))
      return '3–63 chars, lowercase letters/numbers/hyphens only, no leading/trailing hyphens'
    return ''
  }

  const searchHF = async () => {
    if (!hfQuery.trim()) return
    setHfSearching(true)
    setHfResults([])
    try {
      const res = await fetch(
        `https://huggingface.co/api/datasets?search=${encodeURIComponent(hfQuery.trim())}&limit=8&sort=downloads&direction=-1`,
        { signal: AbortSignal.timeout(8000) }
      )
      const data = await res.json()
      setHfResults(Array.isArray(data) ? data : [])
    } catch { setHfResults([]) }
    setHfSearching(false)
    setHfSearched(true)
  }

  const selectDataset = (ds: HFDataset) => {
    setSelected(ds)
    setHfResults([])
    setHfSearched(false)
    setHfQuery('')
    const suggested = suggestBucket(ds.id)
    setBucketName(suggested)
    setBucketError('')
  }

  const startImport = async () => {
    const err = validateBucket(bucketName)
    if (err) { setBucketError(err); return }
    if (!selected) return
    try {
      const res = await fetch('/api/datasets/import-hf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hf_dataset_id: selected.id, bucket_name: bucketName, max_files: maxFiles }),
      })
      const data = await res.json()
      if (!res.ok) { setBucketError(data.detail ?? 'Import failed'); return }
      setJobId(data.job_id)
      pollRef.current = setInterval(async () => {
        const r = await fetch(`/api/datasets/import-hf/${data.job_id}`)
        const d = await r.json()
        setJob(d)
        setTimeout(() => logRef.current?.scrollTo(0, logRef.current.scrollHeight), 30)
        if (d.status === 'completed' || d.status === 'error') {
          clearInterval(pollRef.current!)
        }
      }, 800)
    } catch (e: any) {
      setBucketError(e.message)
    }
  }

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const isRunning = job && (job.status === 'queued' || job.status === 'running')
  const isDone    = job?.status === 'completed'
  const isError   = job?.status === 'error'

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        style={{ maxWidth: 600, width: '95vw', padding: 28, maxHeight: '92vh', overflowY: 'auto' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 9, background: '#f9731620', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Package size={18} color="#f97316" />
            </div>
            <div>
              <h3 style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', margin: 0 }}>Import from HuggingFace</h3>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: 0 }}>Download dataset files → create MinIO bucket</p>
            </div>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={onClose} disabled={isRunning}>✕</button>
        </div>

        {/* ── Pre-import form ── */}
        {!jobId && (
          <>
            {/* Search */}
            <div style={{ marginBottom: 18 }}>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>
                Search HuggingFace Datasets
              </label>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  value={hfQuery}
                  onChange={e => setHfQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && searchHF()}
                  placeholder="e.g. imagenet, coco, squad, mnist..."
                  style={{ flex: 1, padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}
                />
                <button
                  onClick={searchHF}
                  disabled={hfSearching || !hfQuery.trim()}
                  style={{ padding: '8px 14px', borderRadius: 7, fontSize: 12, fontWeight: 600, cursor: 'pointer', border: '1px solid var(--border-default)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6, opacity: hfSearching || !hfQuery.trim() ? 0.5 : 1 }}
                >
                  {hfSearching ? <Loader2 size={13} className="animate-spin" /> : <Search size={13} />}
                  Search
                </button>
              </div>

              {/* Results */}
              {hfResults.length > 0 && (
                <div style={{ marginTop: 8, border: '1px solid var(--border-default)', borderRadius: 8, overflow: 'hidden', maxHeight: 240, overflowY: 'auto' }}>
                  {hfResults.map((ds, i) => (
                    <button
                      key={ds.id}
                      onClick={() => selectDataset(ds)}
                      style={{
                        width: '100%', display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
                        border: 'none', borderBottom: i < hfResults.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                        background: i % 2 === 0 ? 'var(--bg-surface)' : 'var(--bg-base)',
                        cursor: 'pointer', textAlign: 'left',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ds.id}</div>
                        <div style={{ display: 'flex', gap: 6, marginTop: 3, flexWrap: 'wrap' }}>
                          {ds.gated && <span style={{ fontSize: 10, background: '#f9731620', color: '#f97316', borderRadius: 4, padding: '1px 5px', fontWeight: 600 }}>Gated</span>}
                          {(ds.tags ?? []).slice(0, 3).map(t => (
                            <span key={t} style={{ fontSize: 10, background: 'var(--bg-elevated)', color: 'var(--text-muted)', borderRadius: 4, padding: '1px 5px' }}>{t}</span>
                          ))}
                        </div>
                      </div>
                      {ds.downloads !== undefined && (
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', flexShrink: 0, textAlign: 'right' }}>
                          ↓ {ds.downloads >= 1000 ? (ds.downloads / 1000).toFixed(0) + 'k' : ds.downloads}
                        </div>
                      )}
                      <span style={{ fontSize: 11, color: 'var(--primary)', fontWeight: 600, flexShrink: 0 }}>Select →</span>
                    </button>
                  ))}
                </div>
              )}
              {hfSearched && hfResults.length === 0 && !hfSearching && (
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>ไม่พบผลลัพธ์ ลองคำค้นอื่น</p>
              )}
            </div>

            {/* Selected dataset chip */}
            {selected && (
              <div style={{ marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 8, background: '#f9731614', border: '1px solid #f9731640' }}>
                <Package size={14} color="#f97316" style={{ flexShrink: 0 }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>{selected.id}</span>
                {selected.gated && <span style={{ fontSize: 10, background: '#f9731620', color: '#f97316', borderRadius: 4, padding: '2px 6px', fontWeight: 700 }}>Gated ⚠</span>}
                <button onClick={() => setSelected(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 16, lineHeight: 1 }}>×</button>
              </div>
            )}

            {/* Bucket name */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                MinIO Bucket Name <span style={{ color: 'var(--danger)' }}>*</span>
              </label>
              <input
                value={bucketName}
                onChange={e => { setBucketName(e.target.value); setBucketError(validateBucket(e.target.value)) }}
                placeholder="e.g. hf-squad"
                style={{ width: '100%', padding: '8px 10px', borderRadius: 7, fontSize: 13, border: `1px solid ${bucketError ? 'var(--danger)' : 'var(--border-default)'}`, background: 'var(--bg-surface)', color: 'var(--text-primary)', boxSizing: 'border-box' }}
              />
              {bucketError
                ? <p style={{ fontSize: 11, color: 'var(--danger)', marginTop: 4 }}>{bucketError}</p>
                : <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>lowercase letters, numbers, hyphens · 3–63 chars</p>
              }
            </div>

            {/* Max files */}
            <div style={{ marginBottom: 22 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                Max files to download
                <span style={{ marginLeft: 8, fontWeight: 400, color: 'var(--text-muted)' }}>(ใช้ขีดจำกัดเพื่อป้องกัน dataset ขนาดใหญ่)</span>
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="range" min={1} max={100} value={maxFiles}
                  onChange={e => setMaxFiles(Number(e.target.value))}
                  style={{ flex: 1, accentColor: '#f97316' }}
                />
                <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', minWidth: 30, textAlign: 'right' }}>{maxFiles}</span>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
              <button
                className="btn btn-primary"
                style={{ background: '#f97316', borderColor: '#f97316', display: 'flex', alignItems: 'center', gap: 7, opacity: !selected || !bucketName ? 0.6 : 1 }}
                onClick={startImport}
                disabled={!selected || !bucketName}
              >
                <Package size={14} /> Import Dataset
              </button>
            </div>
          </>
        )}

        {/* ── Progress ── */}
        {jobId && job && (
          <div>
            {/* Status row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <div style={{ width: 40, height: 40, borderRadius: 10, background: isDone ? '#10b98120' : isError ? '#ef444420' : '#f9731620', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                {isRunning ? <Loader2 size={20} color="#f97316" className="animate-spin" /> :
                 isDone    ? <CheckCircle2 size={20} color="#10b981" /> :
                             <XCircle size={20} color="#ef4444" />}
              </div>
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                  {isRunning ? `Importing ${job.hf_id}…` : isDone ? 'Import complete!' : 'Import failed'}
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0, marginTop: 2 }}>
                  Bucket: <code style={{ fontFamily: 'var(--font-mono)' }}>{job.bucket}</code>
                  {job.total_files > 0 && ` · ${job.done_files}/${job.total_files} files`}
                </p>
              </div>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{job.progress}%</span>
            </div>

            {/* Progress bar */}
            <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-elevated)', marginBottom: 16, overflow: 'hidden' }}>
              <div style={{ height: '100%', borderRadius: 3, transition: 'width 0.4s ease', width: `${job.progress}%`, background: isDone ? '#10b981' : isError ? '#ef4444' : '#f97316' }} />
            </div>

            {/* Log */}
            <pre ref={logRef} style={{
              background: 'var(--bg-base)', border: '1px solid var(--border-subtle)', borderRadius: 8,
              padding: '10px 14px', fontSize: 11, lineHeight: 1.6, color: 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)', maxHeight: '30vh', overflowY: 'auto',
              whiteSpace: 'pre-wrap', wordBreak: 'break-all', marginBottom: 20,
            }}>
              {job.log || 'Starting…'}
            </pre>

            {isError && (
              <div style={{ padding: '10px 14px', borderRadius: 8, background: '#ef444418', border: '1px solid #ef444440', marginBottom: 16, fontSize: 12, color: '#ef4444' }}>
                {job.error}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              {isDone && (
                <a href="http://localhost:9001" target="_blank" rel="noopener noreferrer" className="btn btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                  <ExternalLink size={12} /> MinIO Console
                </a>
              )}
              <button
                className="btn btn-primary"
                style={{ background: isDone ? '#10b981' : '#ef4444', borderColor: isDone ? '#10b981' : '#ef4444' }}
                onClick={isDone ? onDone : onClose}
                disabled={isRunning}
              >
                {isDone ? '✓ Done' : isRunning ? 'Importing…' : 'Close'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
